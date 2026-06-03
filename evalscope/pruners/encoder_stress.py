# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Encoder-Stress Pruner for MMMU.

This strategy oversamples from MMMU questions whose correct answers require
fine-grained *spatial* or *structural* image understanding — the class of
questions where a degraded image encoder first shows its failures.

Background
----------
MMMU questions span 30 subjects and many image types.  Not all image types
stress the encoder equally:

- **Encoder-heavy types** (spatial/structural): circuit diagrams, molecular
  structures, geometric proofs, music sheets, medical imaging, scientific
  notation.  Language priors provide little help here; the encoder must parse
  precise spatial relationships.

- **Encoder-light types**: photographs, tables, natural scenes.  Models can
  often answer these from question phrasing and world knowledge alone, even
  with a degraded encoder.

A uniformly random probe of 5 % of ~12 K MMMU samples will be approximately
5 % encoder-heavy — not enough signal to detect encoder degradation reliably.
This strategy oversamples encoder-heavy items by a configurable weight
(default 3×), making encoder failures 3× more visible in the score.

Subject-level diversity
-----------------------
Within each encoder type, we stratify by subject to prevent the probe from
collapsing to a single discipline.  This keeps the probe broadly representative
while still being encoder-sensitive.
"""
import ast
import random
from typing import Any, Dict, List, Optional, Set

from evalscope.pruners.base import PruningStrategy, register_pruning_strategy
from evalscope.utils.logger import get_logger

logger = get_logger()

# Image types from the MMMU paper taxonomy that require fine-grained encoder work.
# Derived from MMMU dataset metadata field ``img_type``.
_DEFAULT_ENCODER_HEAVY_TYPES: Set[str] = {
    'Diagrams',
    'Chemical structure',
    'Music sheet',
    'Geometric shapes',
    'Scientific Notation',
    'Figures',
    'Medical images',
    'Scatter Plot',
    'Bar Chart',
    'Line Chart',        # charts require reading precise axis values
    'Tables',            # table cells with numeric content (not plain OCR)
}


@register_pruning_strategy('mmmu_encoder_stress')
class EncoderStressPruner(PruningStrategy):
    """
    Encoder-stress stratified sampler for MMMU.

    Oversamples questions that rely on encoder-heavy image types while
    ensuring subject-level diversity in the resulting probe set.

    Args:
        encoder_heavy_types: Set of ``img_type`` values considered
            encoder-stressing.  Defaults to ``_DEFAULT_ENCODER_HEAVY_TYPES``.
        encoder_heavy_weight: Sampling weight multiplier for encoder-heavy
            samples vs. encoder-light samples.  Default 3.0.
        difficulty_bonus: Additional weight multiplier for samples whose
            ``topic_difficulty`` is ``"Hard"``.  Default 1.5.
        ensure_subject_coverage: If True, guarantee at least one sample per
            MMMU subject (up to budget).  Default True.
    """

    def __init__(
        self,
        encoder_heavy_types: Optional[Set[str]] = None,
        encoder_heavy_weight: float = 3.0,
        difficulty_bonus: float = 1.5,
        ensure_subject_coverage: bool = True,
    ):
        self.encoder_heavy_types = encoder_heavy_types or _DEFAULT_ENCODER_HEAVY_TYPES
        self.encoder_heavy_weight = encoder_heavy_weight
        self.difficulty_bonus = difficulty_bonus
        self.ensure_subject_coverage = ensure_subject_coverage

    def select(
        self,
        samples: List[Dict[str, Any]],
        prune_ratio: float,
        seed: int = 42,
    ) -> List[Any]:
        if not 0 < prune_ratio <= 1.0:
            raise ValueError(f'prune_ratio must be in (0, 1], got {prune_ratio}')

        rng = random.Random(seed)
        total_keep = max(1, round(len(samples) * prune_ratio))

        # Phase 1: guarantee one sample per subject if budget allows
        selected_ids: List[Any] = []
        seen: set = set()

        if self.ensure_subject_coverage:
            subject_reps = self._pick_subject_representatives(samples, rng)
            for sid in subject_reps:
                if len(selected_ids) >= total_keep:
                    break
                if sid not in seen:
                    seen.add(sid)
                    selected_ids.append(sid)

        # Phase 2: fill remaining budget via weighted sampling
        remaining = total_keep - len(selected_ids)
        if remaining > 0:
            candidates = [s for s in samples if s['id'] not in seen]
            weights = [self._sample_weight(s) for s in candidates]
            extra = rng.choices(candidates, weights=weights, k=remaining * 2)
            for s in extra:
                if len(selected_ids) >= total_keep:
                    break
                if s['id'] not in seen:
                    seen.add(s['id'])
                    selected_ids.append(s['id'])

        logger.info(
            f'[EncoderStressPruner] '
            f'Kept {len(selected_ids)}/{len(samples)} samples '
            f'(ratio={prune_ratio}, seed={seed})'
        )
        return selected_ids

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_encoder_heavy(self, sample: Dict[str, Any]) -> bool:
        """Return True if the sample's img_type overlaps with encoder-heavy types."""
        img_type_raw = sample.get('metadata', {}).get('img_type', '[]')
        if isinstance(img_type_raw, str):
            try:
                img_types: List[str] = ast.literal_eval(img_type_raw)
            except (ValueError, SyntaxError):
                img_types = [img_type_raw]
        else:
            img_types = list(img_type_raw)
        return any(t in self.encoder_heavy_types for t in img_types)

    def _sample_weight(self, sample: Dict[str, Any]) -> float:
        """Compute the sampling weight for a single sample."""
        w = self.encoder_heavy_weight if self._is_encoder_heavy(sample) else 1.0
        difficulty = sample.get('metadata', {}).get('topic_difficulty', '')
        if difficulty == 'Hard':
            w *= self.difficulty_bonus
        # Also use variance as a tiebreaker (samples where the reference model
        # struggled are more diagnostic).
        w *= max(sample.get('variance', 0.0), 1e-6)
        return w

    def _pick_subject_representatives(
        self,
        samples: List[Dict[str, Any]],
        rng: random.Random,
    ) -> List[Any]:
        """Pick one encoder-heavy sample per subject (falling back to any sample)."""
        subjects: Dict[str, List[Dict[str, Any]]] = {}
        for s in samples:
            # Subject is encoded in the sample id, e.g. "validation_Electronics_11"
            sid = str(s.get('id', ''))
            parts = sid.split('_')
            subject = parts[1] if len(parts) >= 2 else 'Unknown'
            subjects.setdefault(subject, []).append(s)

        reps: List[Any] = []
        for subject, group in sorted(subjects.items()):
            # Prefer encoder-heavy, hard samples; fall back to anything
            heavy = [s for s in group if self._is_encoder_heavy(s)]
            pool = heavy if heavy else group
            # Pick the one with the highest weight (most diagnostic)
            best = max(pool, key=self._sample_weight)
            reps.append(best['id'])
        return reps
