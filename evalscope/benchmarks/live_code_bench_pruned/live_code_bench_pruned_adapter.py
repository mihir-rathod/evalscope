# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
"""
LiveCodeBench Pruned Adapter.

Wraps the upstream ``LiveCodeBenchAdapter`` and sub-selects samples before
evaluation using a pluggable ``PruningStrategy``.  All scoring, sandbox
execution, and reporting logic is inherited unchanged from the parent.

Extension point
---------------
Pruning is implemented via ``sample_filter``, the hook that ``DefaultDataAdapter``
passes to the ``DataLoader``.  This keeps the loading, caching, and formatting
pipeline intact — only the set of samples entering it changes.

Sample identification
---------------------
The pruner works with integer *row indices* (0-based position in the HF dataset)
that are stored in ``sample.metadata['_pruner_idx']`` by the overridden
``record_to_sample``.  The pre-computed file ``data/lcb_item_scores.json``
uses the same index scheme (derived from evalscope's sequential sample IDs).
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Set

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.constants import Tags
from evalscope.pruners.base import get_pruning_strategy
from evalscope.utils.logger import get_logger

logger = get_logger()

# Path to the pre-computed item scores bundled with this repo.
# Resolved relative to this file:  <repo>/evalscope/benchmarks/live_code_bench_pruned/
#                                                                       ↑ .parent ×3 → <repo>/evalscope/
#                                                                                       ×4 → <repo>/
_ITEM_SCORES_PATH: Path = (
    Path(__file__).parent  # live_code_bench_pruned/
    .parent                # benchmarks/
    .parent                # evalscope/ (package)
    .parent                # repo root
    / 'data'
    / 'lcb_item_scores.json'
)


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned)',
        tags=[Tags.CODING],
        description="""
## Overview

A pruned version of LiveCodeBench v5 that retains a small, statistically
representative subset of problems.  Uses **Discriminative Stratified Sampling**
to keep easy / discriminating / hard problems in calibrated proportions,
preserving rank-order accuracy at a fraction of evaluation cost.

## When to use

Run `live_code_bench_pruned` when you need a fast go/no-go signal on a new
model's coding capability.  Compare results with `live_code_bench` using the
`compare_runs` tool to validate that the pruned score matches the full score.

## Key parameters

- ``pruning_strategy`` – strategy name (default: ``discriminative_stratified``)
- ``prune_ratio``      – fraction of samples to keep (default: 0.1 → ~31 samples)
- ``prune_seed``       – RNG seed for reproducibility (default: 42)
""",
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=['release_v5'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        review_timeout=6,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': 'Name of the pruning strategy (see evalscope/pruners/).',
                'value': 'discriminative_stratified',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to retain.  Must be in (0, 1].',
                'value': 0.1,
            },
            'prune_seed': {
                'type': 'int',
                'description': 'Random seed passed to the pruning strategy.',
                'value': 42,
            },
            # Inherited from LiveCodeBenchAdapter
            'start_date': {
                'type': 'str | null',
                'description': 'Filter problems from this date (YYYY-MM-DD).',
                'value': None,
            },
            'end_date': {
                'type': 'str | null',
                'description': 'Filter problems up to this date (YYYY-MM-DD).',
                'value': None,
            },
            'debug': {
                'type': 'bool',
                'description': 'Enable verbose debug logging.',
                'value': False,
            },
        },
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {},
            },
        },
    )
)
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """
    LiveCodeBench adapter with dataset pruning.

    Adds three extra parameters on top of the parent adapter:
    ``pruning_strategy``, ``prune_ratio``, and ``prune_seed``.  Everything
    else — scoring, sandbox execution, date filtering, report generation —
    is delegated to ``LiveCodeBenchAdapter``.
    """

    # Metadata key used to carry the row index through the sample pipeline.
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
        # Populated by _build_selected_set() before loading starts
        self._selected_indices: Optional[Set[int]] = None

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load(self):
        """Build the index allow-list, then delegate to parent loading."""
        self._build_selected_set()
        return super().load()

    def _build_selected_set(self) -> None:
        """Load pre-computed scores and run the pruning strategy."""
        if not _ITEM_SCORES_PATH.exists():
            raise FileNotFoundError(
                f'Pre-computed LCB item scores not found: {_ITEM_SCORES_PATH}\n'
                'Run:  python3 scripts/build_item_scores.py --evals-dir <Evals/> --out-dir data/'
            )

        with open(_ITEM_SCORES_PATH, encoding='utf-8') as f:
            item_scores = json.load(f)

        strategy_cls = get_pruning_strategy(self._pruning_strategy_name)
        strategy = strategy_cls()
        selected = strategy.select(item_scores, self._prune_ratio, self._prune_seed)
        self._selected_indices = set(selected)

        logger.info(
            f'[LiveCodeBenchPruned] Strategy={self._pruning_strategy_name}, '
            f'ratio={self._prune_ratio}, seed={self._prune_seed} → '
            f'{len(self._selected_indices)} samples selected'
        )

    # ------------------------------------------------------------------
    # sample_filter — called by DataLoader after record_to_sample
    # ------------------------------------------------------------------

    def sample_filter(self, sample: Sample) -> bool:
        """
        Two-stage filter:
        1. Parent's date filter (start_date / end_date).
        2. Pruning allow-list based on row index.
        """
        # Stage 1: date filter from parent
        if not super().sample_filter(sample):
            return False

        # Stage 2: pruning filter
        if self._selected_indices is None:
            return True  # safety: allow all if index set not built yet

        idx = sample.metadata.get(self._PRUNER_IDX_KEY)
        if idx is None:
            return True  # unknown index: let through to avoid silent data loss

        return int(idx) in self._selected_indices

    # ------------------------------------------------------------------
    # record_to_sample — inject row index into metadata
    # ------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Delegate to parent, then inject the row index into metadata so
        ``sample_filter`` can use it.

        The row index is read from the ``__index_level_0__`` field that
        HuggingFace datasets expose when converted to list of dicts, falling
        back to a ``_row_idx`` counter if that field is absent.
        """
        sample = super().record_to_sample(record)

        # Prefer the HF row index field; fall back to sequential counter
        row_idx = record.get('__index_level_0__', record.get('_row_idx'))
        if row_idx is None:
            # Last resort: use contest_date + question hash as a proxy
            # (not needed when HF field is present, but safe fallback)
            row_idx = record.get('question_id', record.get('problem_id'))

        if row_idx is not None:
            sample.metadata[self._PRUNER_IDX_KEY] = int(row_idx)

        return sample
