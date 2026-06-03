# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
"""
AA-LCR Pruned Adapter.

Wraps ``AALCRAdapter`` and sub-selects samples before evaluation using the same
``DiscriminativeStratifiedPruner`` as LiveCodeBench.  This demonstrates the
*universal pruner* design: one strategy class, zero benchmark-specific changes.

Secondary stratification
------------------------
AA-LCR has an important extra dimension: context length (``input_tokens``).
The pruner is configured with ``secondary_meta_key='input_tokens'`` so the
pruned set spans the full context-length spectrum — not just random difficulty.

Judge noise note
----------------
AA-LCR uses an LLM judge (non-deterministic).  The discrimination scores in
``data/aa_lcr_item_scores.json`` reflect single-run scores, so variance
estimates are inflated by judge noise.  We document this limitation in
Handout A.  With access to multiple judge runs the scores could be corrected.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Set

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter, PROMPT_TEMPLATE
from evalscope.constants import Tags
from evalscope.pruners.base import get_pruning_strategy
from evalscope.utils.logger import get_logger

logger = get_logger()

_ITEM_SCORES_PATH: Path = (
    Path(__file__).parent   # aa_lcr_pruned/
    .parent                 # benchmarks/
    .parent                 # evalscope/ (package)
    .parent                 # repo root
    / 'data'
    / 'aa_lcr_item_scores.json'
)


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

A pruned version of AA-LCR that retains a compact, representative subset of
long-context retrieval questions.  Uses **Discriminative Stratified Sampling**
with secondary stratification by context length (``input_tokens``) to preserve
both difficulty spread and context-length diversity.

## When to use

Run `aa_lcr_pruned` for fast long-context capability screening.  Note: AA-LCR
is graded by an LLM judge (non-deterministic).  At prune_ratio=0.1 (10 samples)
judge noise is a non-trivial fraction of observed variance — results should be
interpreted directionally rather than as precise accuracy estimates.

## Key parameters

- ``pruning_strategy`` – strategy name (default: ``discriminative_stratified``)
- ``prune_ratio``      – fraction to keep (default: 0.1 → 10 samples)
- ``prune_seed``       – RNG seed (default: 42)
""",
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': 'Name of the pruning strategy.',
                'value': 'discriminative_stratified',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to retain.',
                'value': 0.1,
            },
            'prune_seed': {
                'type': 'int',
                'description': 'Random seed for the pruning strategy.',
                'value': 42,
            },
            # Inherited from AALCRAdapter
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory with AA-LCR text files; auto-download if null.',
                'value': None,
            },
        },
    )
)
class AALCRPrunedAdapter(AALCRAdapter):
    """
    AA-LCR adapter with dataset pruning.

    Same pruner interface as ``LiveCodeBenchPrunedAdapter``; the only
    benchmark-specific detail is loading from ``aa_lcr_item_scores.json``
    and using ``input_tokens`` for secondary stratification.
    """

    _PRUNER_IDX_KEY = '_pruner_idx'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pruning_strategy_name: str = self.extra_params.get(
            'pruning_strategy', 'discriminative_stratified'
        )
        self._prune_ratio: float = float(
            self.extra_params.get('prune_ratio', 0.1)
        )
        self._prune_seed: int = int(
            self.extra_params.get('prune_seed', 42)
        )
        self._selected_indices: Optional[Set[int]] = None
        # Sequential counter matching the 0-based index used in review files
        self._row_counter: int = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):
        self._build_selected_set()
        return super().load()

    def _build_selected_set(self) -> None:
        if not _ITEM_SCORES_PATH.exists():
            raise FileNotFoundError(
                f'Pre-computed AA-LCR item scores not found: {_ITEM_SCORES_PATH}\n'
                'Run:  python3 scripts/build_item_scores.py --evals-dir <Evals/> --out-dir data/'
            )

        with open(_ITEM_SCORES_PATH, encoding='utf-8') as f:
            item_scores = json.load(f)

        strategy_cls = get_pruning_strategy(self._pruning_strategy_name)
        # Configure secondary stratification by input_tokens (context length)
        strategy = strategy_cls(secondary_meta_key='input_tokens', n_secondary_buckets=3)
        selected = strategy.select(item_scores, self._prune_ratio, self._prune_seed)
        self._selected_indices = set(selected)

        logger.info(
            f'[AALCRPruned] Strategy={self._pruning_strategy_name}, '
            f'ratio={self._prune_ratio}, seed={self._prune_seed} → '
            f'{len(self._selected_indices)} samples selected'
        )

    # ------------------------------------------------------------------
    # sample_filter
    # ------------------------------------------------------------------

    def sample_filter(self, sample: Sample) -> bool:
        if self._selected_indices is None:
            return True
        idx = sample.metadata.get(self._PRUNER_IDX_KEY)
        if idx is None:
            return True
        return int(idx) in self._selected_indices

    # ------------------------------------------------------------------
    # record_to_sample — inject row index
    # ------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        sample = super().record_to_sample(record)
        sample.metadata[self._PRUNER_IDX_KEY] = self._row_counter
        self._row_counter += 1
        return sample
