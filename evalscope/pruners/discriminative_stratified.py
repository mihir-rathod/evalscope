# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Discriminative Stratified Sampling (DSS) with metadata-augmented stratification.

Algorithm overview
------------------
Given per-sample scores from a set of *reference* models, DSS selects a
compact subset that preserves rank-order signal for any new model:

1. Compute each sample's **discrimination score** (Bernoulli variance
   ``p*(1-p)`` where ``p`` is the fraction of reference models that passed).
   Samples where all models agree (variance = 0) contribute zero rank-order
   information.

2. Partition samples into three **difficulty tiers** by average score:
   - Easy (avg ≥ EASY_THRESHOLD)         → anchor the ceiling
   - Discriminating (HARD_THRESHOLD < avg < EASY_THRESHOLD) → primary signal
   - Hard (avg ≤ HARD_THRESHOLD)         → anchor the floor

3. Allocate a **budget** to each tier (defaults: 15 % / 70 % / 15 %).

4. Within each tier, optionally **secondary-stratify** by a metadata field
   (e.g. ``contest_date`` recency for LCB, ``input_tokens`` quartile for
   AA-LCR).  This ensures the pruned set spans the full metadata space, not
   just a random slice of difficulty.

5. Within each (tier × secondary-bucket) cell, draw samples with probability
   proportional to their discrimination score.

Generalisation argument
-----------------------
Selection is driven by *problem difficulty* (a property of the item itself),
not by the specific spread between the three reference models.  A fourth model
with different capability will still encounter the same difficulty spectrum —
easy, medium, and hard items — because the tiers reflect intrinsic question
hardness rather than any model-specific pattern.
"""
import math
import random
from typing import Any, Dict, List, Optional

from evalscope.pruners.base import PruningStrategy, register_pruning_strategy
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_pruning_strategy('discriminative_stratified')
class DiscriminativeStratifiedPruner(PruningStrategy):
    """
    Discriminative Stratified Sampling pruner.

    Works identically for LCB (binary ``pass``) and AA-LCR (binary ``acc``).
    Secondary stratification is controlled via ``secondary_meta_key``:

    - Set to ``"contest_date_bucket"`` for LCB to spread across problem age.
    - Set to ``"token_bucket"`` for AA-LCR to spread across context lengths.
    - Leave as ``None`` to disable secondary stratification.

    Args:
        easy_threshold: avg-score threshold above which a sample is "easy".
        hard_threshold: avg-score threshold below which a sample is "hard".
        tier_budget: Dict mapping tier name → fraction of total budget.
            Must sum to 1.0.
        secondary_meta_key: Optional metadata field name to use for secondary
            stratification within each tier.
        n_secondary_buckets: Number of quantile buckets for numeric secondary
            fields.  Ignored for categorical fields.
    """

    TIER_EASY = 'easy'
    TIER_DISC = 'discriminating'
    TIER_HARD = 'hard'

    DEFAULT_EASY_THRESHOLD = 0.8
    DEFAULT_HARD_THRESHOLD = 0.2
    DEFAULT_TIER_BUDGET = {
        TIER_EASY: 0.15,
        TIER_DISC: 0.70,
        TIER_HARD: 0.15,
    }

    def __init__(
        self,
        easy_threshold: float = DEFAULT_EASY_THRESHOLD,
        hard_threshold: float = DEFAULT_HARD_THRESHOLD,
        tier_budget: Optional[Dict[str, float]] = None,
        secondary_meta_key: Optional[str] = None,
        n_secondary_buckets: int = 3,
    ):
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold
        self.tier_budget = tier_budget or dict(self.DEFAULT_TIER_BUDGET)
        self.secondary_meta_key = secondary_meta_key
        self.n_secondary_buckets = n_secondary_buckets

        # Validate budget sums to ~1.0
        total = sum(self.tier_budget.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f'tier_budget values must sum to 1.0, got {total:.4f}'
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select(
        self,
        samples: List[Dict[str, Any]],
        prune_ratio: float,
        seed: int = 42,
    ) -> List[Any]:
        """
        Select ``round(len(samples) * prune_ratio)`` samples.

        Returns:
            List of selected sample ``id`` values.
        """
        if not 0 < prune_ratio <= 1.0:
            raise ValueError(f'prune_ratio must be in (0, 1], got {prune_ratio}')

        rng = random.Random(seed)
        total_keep = max(1, round(len(samples) * prune_ratio))

        # Step 1: Partition into difficulty tiers
        tiers = self._partition_tiers(samples)

        # Step 2: Allocate budget per tier
        budgets = self._allocate_budgets(tiers, total_keep)

        # Step 3: Optionally compute secondary buckets
        if self.secondary_meta_key:
            secondary_map = self._build_secondary_buckets(samples)
        else:
            secondary_map = {}

        # Step 4: Sample within each tier (+ secondary bucket if enabled)
        selected_ids: List[Any] = []
        seen: set = set()

        for tier_name in [self.TIER_EASY, self.TIER_DISC, self.TIER_HARD]:
            tier_samples = tiers[tier_name]
            budget = budgets[tier_name]
            if not tier_samples or budget == 0:
                continue

            if self.secondary_meta_key and secondary_map:
                ids = self._sample_with_secondary(
                    tier_samples, budget, secondary_map, rng
                )
            else:
                ids = self._weighted_sample(tier_samples, budget, rng)

            for sid in ids:
                if sid not in seen:
                    seen.add(sid)
                    selected_ids.append(sid)

        logger.info(
            f'[DiscriminativeStratifiedPruner] '
            f'Kept {len(selected_ids)}/{len(samples)} samples '
            f'(ratio={prune_ratio}, seed={seed})'
        )
        return selected_ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _partition_tiers(
        self, samples: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        tiers: Dict[str, List[Dict[str, Any]]] = {
            self.TIER_EASY: [],
            self.TIER_DISC: [],
            self.TIER_HARD: [],
        }
        for s in samples:
            avg = s['avg_score']
            if avg >= self.easy_threshold:
                tiers[self.TIER_EASY].append(s)
            elif avg <= self.hard_threshold:
                tiers[self.TIER_HARD].append(s)
            else:
                tiers[self.TIER_DISC].append(s)
        return tiers

    def _allocate_budgets(
        self,
        tiers: Dict[str, List[Dict[str, Any]]],
        total_keep: int,
    ) -> Dict[str, int]:
        """
        Proportionally allocate the total budget across tiers.
        Any budget exceeding a tier's size is redistributed to the
        discriminating tier (most valuable signal).
        """
        raw = {
            name: round(total_keep * frac)
            for name, frac in self.tier_budget.items()
        }
        # Clamp to tier sizes and collect surplus
        surplus = 0
        budgets: Dict[str, int] = {}
        for name, b in raw.items():
            available = len(tiers[name])
            actual = min(b, available)
            surplus += b - actual
            budgets[name] = actual

        # Give surplus to discriminating tier (up to its size)
        if surplus > 0:
            disc_available = len(tiers[self.TIER_DISC]) - budgets[self.TIER_DISC]
            budgets[self.TIER_DISC] += min(surplus, disc_available)

        return budgets

    def _weighted_sample(
        self,
        candidates: List[Dict[str, Any]],
        k: int,
        rng: random.Random,
    ) -> List[Any]:
        """Sample k ids from candidates, weighted by discrimination score."""
        if k >= len(candidates):
            return [s['id'] for s in candidates]
        weights = [max(s['variance'], 1e-9) for s in candidates]
        chosen = rng.choices(candidates, weights=weights, k=k)
        # Deduplicate while preserving draw order
        seen: set = set()
        result: List[Any] = []
        for s in chosen:
            if s['id'] not in seen:
                seen.add(s['id'])
                result.append(s['id'])
        # If deduplication reduced the count, fill greedily from unchosen
        if len(result) < k:
            remaining = [s for s in candidates if s['id'] not in seen]
            remaining.sort(key=lambda s: s['variance'], reverse=True)
            for s in remaining:
                if len(result) >= k:
                    break
                result.append(s['id'])
        return result

    def _build_secondary_buckets(
        self, samples: List[Dict[str, Any]]
    ) -> Dict[Any, str]:
        """
        Map each sample id → secondary bucket label.

        For numeric fields (e.g. input_tokens), creates quantile buckets.
        For string fields (e.g. contest_date_bucket), uses the value as-is.
        """
        key = self.secondary_meta_key
        values = []
        for s in samples:
            v = s.get('metadata', {}).get(key)
            values.append((s['id'], v))

        # Detect numeric vs categorical
        numeric_values = [v for _, v in values if isinstance(v, (int, float))]
        if len(numeric_values) == len(values):
            # Compute quantile boundaries
            sorted_vals = sorted(numeric_values)
            n = len(sorted_vals)
            boundaries = [
                sorted_vals[int(i * n / self.n_secondary_buckets)]
                for i in range(1, self.n_secondary_buckets)
            ]
            bucket_map: Dict[Any, str] = {}
            for sid, v in values:
                bucket = 0
                for b in boundaries:
                    if v >= b:
                        bucket += 1
                bucket_map[sid] = f'bucket_{bucket}'
        else:
            # Categorical: use value directly, treat None as 'unknown'
            bucket_map = {sid: str(v) if v is not None else 'unknown'
                          for sid, v in values}
        return bucket_map

    def _sample_with_secondary(
        self,
        tier_samples: List[Dict[str, Any]],
        budget: int,
        secondary_map: Dict[Any, str],
        rng: random.Random,
    ) -> List[Any]:
        """Sample within a tier, distributed across secondary buckets."""
        # Group by secondary bucket
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for s in tier_samples:
            bucket = secondary_map.get(s['id'], 'unknown')
            buckets.setdefault(bucket, []).append(s)

        n_buckets = len(buckets)
        if n_buckets == 0:
            return []

        # Distribute budget roughly evenly across buckets
        base = budget // n_buckets
        remainder = budget % n_buckets
        result: List[Any] = []

        for i, (_, bucket_samples) in enumerate(sorted(buckets.items())):
            k = base + (1 if i < remainder else 0)
            result.extend(self._weighted_sample(bucket_samples, k, rng))

        return result
