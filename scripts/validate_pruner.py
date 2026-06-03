#!/usr/bin/env python3
# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Offline rank-order validation for the Discriminative Stratified Sampler.

Two validation modes are run:

1. SELF-CONSISTENCY (deployment proxy)
   Use all 3 reference models' scores to select the pruned subset, then
   check how well each model's score on the pruned set matches its true score
   on the full dataset.  This approximates the real deployment scenario where
   a new (4th) model is evaluated on the pre-selected pruned subset.

2. LEAVE-ONE-OUT (conservative worst case)
   Treat each of the 3 reference models as a holdout.  Build item scores from
   the remaining 2 models, select the subset, then estimate the holdout
   model's score on those samples.  This is harder than the real case
   (fewer reference models → coarser difficulty estimates) so it provides a
   lower-bound on the expected quality.

Both modes are repeated across N random seeds to get variance estimates.

Usage
-----
    python3 scripts/validate_pruner.py \\
        --reviews-dir  "/path/to/Evals/Part 1/reviews" \\
        --prune-ratio  0.1 \\
        --n-trials     50
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bernoulli_variance(p: float) -> float:
    return p * (1.0 - p)


def _rank_match(a: List[float], b: List[float]) -> bool:
    def order(lst): return sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
    return order(a) == order(b)


def _load_score_matrix(
    raw_review_dir: Path,
    benchmark_prefix: str,
    score_key: str,
    model_names: List[str],
) -> Dict[int, Dict[str, float]]:
    """Returns {sample_idx: {model: score}}."""
    matrix: Dict[int, Dict[str, float]] = {}
    for model in model_names:
        path = raw_review_dir / f'{benchmark_prefix}__{model}.jsonl'
        if not path.exists():
            print(f'  [WARN] Missing: {path.name}')
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                idx = int(rec['index'])
                sv = rec['sample_score']['score']['value']
                score = float(sv.get(score_key, 0.0))
                matrix.setdefault(idx, {})[model] = score
    return matrix


def _load_item_scores(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _fmt(v: float) -> str:
    return f'{v * 100:.1f}%'


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def self_consistency(
    item_scores: List[Dict[str, Any]],
    score_matrix: Dict[int, Dict[str, float]],
    full_accuracies: Dict[str, float],
    model_names: List[str],
    strategy_cls,
    prune_ratio: float,
    n_trials: int,
    strategy_kwargs: Dict = None,
) -> Dict[str, Any]:
    """
    All 3 models select → estimate each model's score on pruned subset.
    Approximates the deployment scenario.
    """
    strategy_kwargs = strategy_kwargs or {}
    errors = []
    rank_ok = 0

    for seed in range(n_trials):
        strategy = strategy_cls(**strategy_kwargs)
        selected = set(strategy.select(item_scores, prune_ratio, seed=seed))
        pruned = [i for i in sorted(score_matrix) if i in selected]
        if not pruned:
            continue

        pruned_accs = {
            m: sum(score_matrix[i].get(m, 0.0) for i in pruned) / len(pruned)
            for m in model_names
        }
        for m in model_names:
            errors.append(abs(pruned_accs[m] - full_accuracies[m]))

        full_vec   = [full_accuracies[m]  for m in model_names]
        pruned_vec = [pruned_accs[m]      for m in model_names]
        if _rank_match(full_vec, pruned_vec):
            rank_ok += 1

    return {
        'mae':           sum(errors) / max(1, len(errors)),
        'max_err':       max(errors, default=float('nan')),
        'within_3pct':   sum(1 for e in errors if e <= 0.03) / max(1, len(errors)),
        'within_5pct':   sum(1 for e in errors if e <= 0.05) / max(1, len(errors)),
        'rank_rate':     rank_ok / max(1, n_trials),
        'n_experiments': len(errors),
    }


def leave_one_out(
    item_scores: List[Dict[str, Any]],
    score_matrix: Dict[int, Dict[str, float]],
    full_accuracies: Dict[str, float],
    model_names: List[str],
    strategy_cls,
    prune_ratio: float,
    n_trials: int,
    strategy_kwargs: Dict = None,
) -> Dict[str, Any]:
    """
    2 models select → estimate the 3rd (holdout) model.
    Conservative lower bound.
    """
    strategy_kwargs = strategy_kwargs or {}

    # Pre-build lookup for metadata from item_scores
    meta_lookup = {s['id']: s.get('metadata', {}) for s in item_scores}

    errors = []
    rank_results_by_seed: Dict[int, List] = {}

    for holdout in model_names:
        refs = [m for m in model_names if m != holdout]
        for seed in range(n_trials):
            # Build item scores from refs only
            ref_items = []
            for idx in sorted(score_matrix):
                ref_scores = [score_matrix[idx].get(m, 0.0) for m in refs]
                avg = sum(ref_scores) / len(ref_scores)
                ref_items.append({
                    'id': idx,
                    'avg_score': avg,
                    'variance': _bernoulli_variance(avg),
                    'metadata': meta_lookup.get(idx, {}),
                })

            strategy = strategy_cls(**strategy_kwargs)
            selected = set(strategy.select(ref_items, prune_ratio, seed=seed))
            pruned = [i for i in sorted(score_matrix) if i in selected]
            if not pruned:
                continue

            pruned_acc = sum(score_matrix[i].get(holdout, 0.0) for i in pruned) / len(pruned)
            err = abs(pruned_acc - full_accuracies[holdout])
            errors.append(err)
            rank_results_by_seed.setdefault(seed, []).append({
                'holdout': holdout,
                'pruned_acc': pruned_acc,
            })

    # Rank preservation per seed (need all 3 holdout estimates)
    rank_ok = 0
    total_seeds = 0
    for seed, results in rank_results_by_seed.items():
        if len(results) != len(model_names):
            continue
        # Align: same ordering as model_names
        result_map = {r['holdout']: r['pruned_acc'] for r in results}
        full_vec   = [full_accuracies[m]  for m in model_names]
        pruned_vec = [result_map[m]        for m in model_names]
        if _rank_match(full_vec, pruned_vec):
            rank_ok += 1
        total_seeds += 1

    return {
        'mae':           sum(errors) / max(1, len(errors)),
        'max_err':       max(errors, default=float('nan')),
        'within_3pct':   sum(1 for e in errors if e <= 0.03) / max(1, len(errors)),
        'within_5pct':   sum(1 for e in errors if e <= 0.05) / max(1, len(errors)),
        'rank_rate':     rank_ok / max(1, total_seeds),
        'n_experiments': len(errors),
    }


def validate_benchmark(
    name: str,
    item_scores_path: Path,
    raw_review_dir: Path,
    benchmark_prefix: str,
    score_key: str,
    prune_ratio: float,
    n_trials: int,
    get_strategy,
    secondary_meta_key: Optional[str] = None,
) -> None:
    print(f'=== {name} ===')
    item_scores = _load_item_scores(item_scores_path)
    model_names = ['gpt-oss-120b', 'kimi-k2.5', 'minimax-m2.5']
    score_matrix = _load_score_matrix(raw_review_dir, benchmark_prefix, score_key, model_names)

    if not score_matrix:
        print('  ERROR: could not load any review data\n')
        return

    full_accs: Dict[str, float] = {
        m: sum(score_matrix[i].get(m, 0.0) for i in score_matrix) / len(score_matrix)
        for m in model_names
    }
    print('  Full benchmark accuracies:')
    for m, v in full_accs.items():
        print(f'    {m:<22} {_fmt(v)}')

    strategy_cls = get_strategy('discriminative_stratified')
    kwargs = {}
    if secondary_meta_key:
        kwargs['secondary_meta_key'] = secondary_meta_key

    # --- Self-consistency ---
    sc = self_consistency(
        item_scores, score_matrix, full_accs, model_names,
        strategy_cls, prune_ratio, n_trials, kwargs,
    )
    print(f'\n  [Self-consistency — deployment proxy, n={sc["n_experiments"]}]')
    print(f'    MAE on pruned vs full:  {_fmt(sc["mae"])}')
    print(f'    Max error:              {_fmt(sc["max_err"])}')
    print(f'    Within ±3%:             {_fmt(sc["within_3pct"])} of trials')
    print(f'    Within ±5%:             {_fmt(sc["within_5pct"])} of trials')
    print(f'    Rank order preserved:   {_fmt(sc["rank_rate"])} of seeds')

    # --- Leave-one-out ---
    loo = leave_one_out(
        item_scores, score_matrix, full_accs, model_names,
        strategy_cls, prune_ratio, n_trials, kwargs,
    )
    print(f'\n  [Leave-one-out — conservative bound, n={loo["n_experiments"]}]')
    print(f'    MAE (2 refs → holdout): {_fmt(loo["mae"])}')
    print(f'    Max error:              {_fmt(loo["max_err"])}')
    print(f'    Within ±3%:             {_fmt(loo["within_3pct"])} of trials')
    print(f'    Within ±5%:             {_fmt(loo["within_5pct"])} of trials')
    print(f'    Rank order preserved:   {_fmt(loo["rank_rate"])} of seeds')
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--lcb-scores',   default='data/lcb_item_scores.json')
    parser.add_argument('--aalcr-scores', default='data/aa_lcr_item_scores.json')
    parser.add_argument('--reviews-dir',  default=None)
    parser.add_argument('--prune-ratio',  type=float, default=0.1)
    parser.add_argument('--n-trials',     type=int,   default=50)
    args = parser.parse_args()

    if args.reviews_dir is None:
        candidates = [
            Path('../cerebras-assessment/ai-model-quality-challenge/Evals/Part 1/reviews'),
        ]
        for c in candidates:
            if c.exists():
                args.reviews_dir = str(c)
                break
    if args.reviews_dir is None or not Path(args.reviews_dir).exists():
        print('ERROR: Pass --reviews-dir /path/to/Evals/Part\\ 1/reviews/')
        return 1

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evalscope.pruners.base import get_pruning_strategy

    reviews_dir = Path(args.reviews_dir)
    print(f'Reviews dir:  {reviews_dir}')
    print(f'Prune ratio:  {args.prune_ratio} (~{round(315*args.prune_ratio)} LCB / ~{round(100*args.prune_ratio)} AA-LCR samples)')
    print(f'Seeds:        {args.n_trials}')
    print()

    validate_benchmark(
        name='LiveCodeBench v5',
        item_scores_path=Path(args.lcb_scores),
        raw_review_dir=reviews_dir,
        benchmark_prefix='live_code_bench_v5',
        score_key='pass',
        prune_ratio=args.prune_ratio,
        n_trials=args.n_trials,
        get_strategy=get_pruning_strategy,
    )

    validate_benchmark(
        name='AA-LCR',
        item_scores_path=Path(args.aalcr_scores),
        raw_review_dir=reviews_dir,
        benchmark_prefix='aa_lcr',
        score_key='acc',
        prune_ratio=args.prune_ratio,
        n_trials=args.n_trials,
        secondary_meta_key='input_tokens',
        get_strategy=get_pruning_strategy,
    )

    return 0


if __name__ == '__main__':
    exit(main())
