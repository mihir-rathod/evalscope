#!/usr/bin/env python3
# Copyright (c) Alibaba, Inc. and its affiliates.
"""
compare_runs — compare a full benchmark run against a pruned run.

Usage
-----
    python -m evalscope_ext.tools.compare_runs \\
        --full   ./results_full/ \\
        --pruned ./results_pruned/

The tool reads evalscope's standard JSON report files from both directories,
computes rank correlation and score error, and prints a summary table.

Exit codes
----------
0  Rank order preserved and mean absolute score error below threshold.
1  Rank order changed or score error above threshold.
2  Could not find or parse report files.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Report loading
# ---------------------------------------------------------------------------

def _find_report_files(results_dir: Path) -> List[Path]:
    """
    Locate evalscope report JSON files under results_dir.

    evalscope writes reports to:
        <work_dir>/reports/<benchmark>/<model>_<timestamp>.json
    """
    reports = []
    for root, _, files in os.walk(results_dir):
        for fname in files:
            if fname.endswith('.json') and not fname.startswith('.'):
                reports.append(Path(root) / fname)
    return sorted(reports)


def _load_report(path: Path) -> Optional[Dict]:
    """Load a single evalscope report JSON, returning None on failure."""
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'[WARN] Could not read {path}: {e}', file=sys.stderr)
        return None


def _extract_scores(report: Dict) -> Dict[str, float]:
    """
    Extract model → overall accuracy mapping from an evalscope report dict.

    evalscope report format (simplified):
        {
          "name": "live_code_bench",
          "model": "gpt-oss-120b",
          "metrics": {"acc": 0.765, ...},
          ...
        }
    Returns {model_name: score}.
    """
    model = report.get('model', report.get('model_id', 'unknown'))
    metrics = report.get('metrics', {})

    # Try common metric keys
    score = None
    for key in ('acc', 'pass@1', 'pass', 'score'):
        if key in metrics:
            score = float(metrics[key])
            break

    # Fallback: mean of all numeric metric values
    if score is None and metrics:
        numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
        score = sum(numeric) / len(numeric) if numeric else 0.0

    return {model: score or 0.0}


def _collect_scores_from_dir(results_dir: Path) -> Dict[str, float]:
    """
    Aggregate model → score from all reports in a directory.
    If multiple reports exist for the same model, the last one wins.
    """
    report_files = _find_report_files(results_dir)
    if not report_files:
        return {}

    scores: Dict[str, float] = {}
    for path in report_files:
        report = _load_report(path)
        if report is None:
            continue
        scores.update(_extract_scores(report))
    return scores


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _spearman_rho(a: List[float], b: List[float]) -> float:
    """Compute Spearman rank correlation between two equal-length sequences."""
    n = len(a)
    if n < 2:
        return float('nan')

    def ranks(lst):
        sorted_idx = sorted(range(n), key=lambda i: lst[i], reverse=True)
        r = [0] * n
        for rank, idx in enumerate(sorted_idx, 1):
            r[idx] = rank
        return r

    ra, rb = ranks(a), ranks(b)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def _mean_absolute_error(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n == 0:
        return float('nan')
    return sum(abs(x - y) for x, y in zip(a, b)) / n


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_RESET  = '\033[0m'
_GREEN  = '\033[92m'
_YELLOW = '\033[93m'
_RED    = '\033[91m'
_BOLD   = '\033[1m'


def _color(text: str, code: str) -> str:
    if sys.stdout.isatty():
        return f'{code}{text}{_RESET}'
    return text


def _fmt_pct(v: float) -> str:
    return f'{v * 100:.1f}%'


def _fmt_delta(d: float) -> str:
    sign = '+' if d >= 0 else ''
    s = f'{sign}{d * 100:.1f}%'
    if abs(d) > 0.05:
        return _color(s, _YELLOW)
    return s


def _print_table(
    rows: List[Tuple[str, float, float, int, int]],
    benchmark_full: str,
    benchmark_pruned: str,
) -> None:
    col_w = [20, 13, 14, 9, 9, 10]
    headers = ['Model', 'Full Score', 'Pruned Score', 'Δ Score', 'Rank (F)', 'Rank (P)']
    sep = '─' * (sum(col_w) + len(col_w) * 3 + 1)

    print(f'\nBenchmark (full):   {benchmark_full}')
    print(f'Benchmark (pruned): {benchmark_pruned}')
    print(sep)
    header_line = '  '.join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    print(_color(header_line, _BOLD))
    print(sep)

    for model, full_s, pruned_s, rank_f, rank_p in rows:
        delta = pruned_s - full_s
        rank_ok = rank_f == rank_p
        rank_f_str = str(rank_f)
        rank_p_str = str(rank_p) if rank_ok else _color(str(rank_p), _RED)
        cells = [
            model[:col_w[0]].ljust(col_w[0]),
            _fmt_pct(full_s).ljust(col_w[1]),
            _fmt_pct(pruned_s).ljust(col_w[2]),
            _fmt_delta(delta).ljust(col_w[3] + (len(_RED) + len(_RESET) if abs(delta) > 0.05 else 0)),
            rank_f_str.ljust(col_w[4]),
            rank_p_str,
        ]
        print('  '.join(cells))

    print(sep)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(full_dir: Path, pruned_dir: Path, mae_threshold: float = 0.05) -> int:
    """
    Compare full vs. pruned runs.  Returns exit code (0=OK, 1=WARN, 2=ERROR).
    """
    full_scores = _collect_scores_from_dir(full_dir)
    pruned_scores = _collect_scores_from_dir(pruned_dir)

    if not full_scores:
        print(f'[ERROR] No report files found under: {full_dir}', file=sys.stderr)
        return 2
    if not pruned_scores:
        print(f'[ERROR] No report files found under: {pruned_dir}', file=sys.stderr)
        return 2

    # Align on common models
    common = sorted(set(full_scores) & set(pruned_scores))
    if not common:
        print(
            f'[ERROR] No models in common between full ({sorted(full_scores.keys())}) '
            f'and pruned ({sorted(pruned_scores.keys())}) runs.',
            file=sys.stderr,
        )
        return 2

    full_vals   = [full_scores[m]   for m in common]
    pruned_vals = [pruned_scores[m] for m in common]

    # Compute ranks (1 = best)
    def rank_list(vals):
        indexed = sorted(enumerate(vals), key=lambda x: x[1], reverse=True)
        ranks = [0] * len(vals)
        for rank, (i, _) in enumerate(indexed, 1):
            ranks[i] = rank
        return ranks

    ranks_full   = rank_list(full_vals)
    ranks_pruned = rank_list(pruned_vals)

    rows = [
        (common[i], full_vals[i], pruned_vals[i], ranks_full[i], ranks_pruned[i])
        for i in range(len(common))
    ]
    rows.sort(key=lambda r: r[3])  # sort by full rank

    _print_table(rows, str(full_dir), str(pruned_dir))

    rho = _spearman_rho(full_vals, pruned_vals)
    mae = _mean_absolute_error(full_vals, pruned_vals)
    rank_match = all(r[3] == r[4] for r in rows)

    print(f'\nSpearman rank correlation (ρ):  {rho:.3f}')
    print(f'Mean absolute score error:       {_fmt_pct(mae)}')

    if rank_match and mae <= mae_threshold:
        verdict = _color(
            f'✅  Rank order preserved. Score error within ±{mae_threshold*100:.0f}% threshold.',
            _GREEN,
        )
        print(f'\nVerdict: {verdict}')
        return 0
    elif not rank_match:
        verdict = _color('❌  Rank order CHANGED — pruned results disagree with full run.', _RED)
        print(f'\nVerdict: {verdict}')
        return 1
    else:
        verdict = _color(
            f'⚠️   Rank order preserved but score error ({_fmt_pct(mae)}) '
            f'exceeds ±{mae_threshold*100:.0f}% threshold.',
            _YELLOW,
        )
        print(f'\nVerdict: {verdict}')
        return 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--full',   required=True,  help='Path to full benchmark results directory.')
    parser.add_argument('--pruned', required=True,  help='Path to pruned benchmark results directory.')
    parser.add_argument(
        '--mae-threshold', type=float, default=0.05,
        help='Maximum acceptable mean absolute score error (default: 0.05 = 5%%).',
    )
    args = parser.parse_args()

    exit_code = run(
        Path(args.full),
        Path(args.pruned),
        mae_threshold=args.mae_threshold,
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
