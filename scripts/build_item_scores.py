#!/usr/bin/env python3
# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Build pre-computed item-score files used by the pruned benchmark adapters.

This script reads the pre-evaluated review JSONL files (from the Cerebras
assessment Evals/ directory) and produces three JSON files:

    data/lcb_item_scores.json        -- LiveCodeBench v5
    data/aa_lcr_item_scores.json     -- AA-LCR
    data/mmmu_stress_scores.json     -- MMMU (glm-4.5v-fp8 reference)

Each file is a list of dicts with schema:
    {
        "id":        <stable sample identifier>,
        "avg_score": <mean score across reference models>,
        "variance":  <Bernoulli variance p*(1-p)>,
        "metadata":  { ...benchmark-specific fields... }
    }

Usage
-----
    python scripts/build_item_scores.py \\
        --evals-dir /path/to/Evals/ \\
        --out-dir   data/

The Evals directory is expected to have the structure shipped in the
Cerebras assessment repo:
    Evals/
    ├── Part 1/
    │   └── reviews/
    │       ├── live_code_bench_v5__gpt-oss-120b.jsonl
    │       ├── live_code_bench_v5__kimi-k2.5.jsonl
    │       ├── live_code_bench_v5__minimax-m2.5.jsonl
    │       ├── aa_lcr__gpt-oss-120b.jsonl
    │       ├── aa_lcr__kimi-k2.5.jsonl
    │       └── aa_lcr__minimax-m2.5.jsonl
    └── MMMU/
        └── reviews/
            └── glm-4.5v-fp8/
                └── mmmu_<Subject>.jsonl  (one per subject)
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _bernoulli_variance(p: float) -> float:
    """Bernoulli variance: p*(1-p).  Maximum at p=0.5, zero at p=0 or p=1."""
    return p * (1.0 - p)


def _date_recency_bucket(date_str: str, n_buckets: int = 4) -> str:
    """
    Bucket a contest date string (YYYY-MM-DD or ISO format) into one of
    n_buckets recency categories.  Newer problems → higher bucket index.
    Returns a string label like 'recency_0' .. 'recency_3'.
    """
    try:
        # Take just the date portion (handles both 'YYYY-MM-DD' and ISO)
        date_part = str(date_str)[:10]
        year, month, _ = date_part.split('-')
        # Convert to sortable int: YYYYMM
        ym = int(year) * 100 + int(month)
        return str(ym)          # Will be quantile-bucketed by the pruner
    except Exception:
        return '0'


# ---------------------------------------------------------------------------
# LiveCodeBench v5
# ---------------------------------------------------------------------------

def build_lcb_scores(reviews_dir: Path, out_dir: Path) -> None:
    """
    Aggregate per-sample pass/fail scores across the three LCB reference models
    and write data/lcb_item_scores.json.

    The stable sample identifier is the integer ``index`` from the review file,
    which corresponds to the dataset row position used by evalscope's evaluator.
    """
    model_files = {
        'gpt-oss-120b':   reviews_dir / 'live_code_bench_v5__gpt-oss-120b.jsonl',
        'kimi-k2.5':      reviews_dir / 'live_code_bench_v5__kimi-k2.5.jsonl',
        'minimax-m2.5':   reviews_dir / 'live_code_bench_v5__minimax-m2.5.jsonl',
    }

    # Check all files exist
    for model, path in model_files.items():
        if not path.exists():
            print(f'[WARN] Missing LCB review file for {model}: {path}', file=sys.stderr)

    # Collect scores per sample index
    sample_scores: Dict[int, List[float]] = defaultdict(list)
    sample_metadata: Dict[int, Dict] = {}

    for model, path in model_files.items():
        if not path.exists():
            continue
        records = _load_jsonl(str(path))
        for rec in records:
            idx = int(rec['index'])
            score_val = rec['sample_score']['score']['value']
            # LCB uses 'pass' key; fall back to 'acc'
            score = float(score_val.get('pass', score_val.get('acc', 0.0)))
            sample_scores[idx].append(score)

            # Capture metadata from the first model that has it
            if idx not in sample_metadata:
                meta = rec.get('sample_score', {}).get('sample_metadata', {}) or {}
                # Also capture contest_date for secondary stratification
                contest_date = meta.get('contest_date', '')
                sample_metadata[idx] = {
                    'contest_date': contest_date,
                    'contest_date_bucket': _date_recency_bucket(contest_date),
                }

    # Build output records
    output = []
    for idx in sorted(sample_scores.keys()):
        scores = sample_scores[idx]
        avg = sum(scores) / len(scores) if scores else 0.0
        output.append({
            'id':        idx,
            'avg_score': round(avg, 6),
            'variance':  round(_bernoulli_variance(avg), 6),
            'metadata':  sample_metadata.get(idx, {}),
        })

    out_path = out_dir / 'lcb_item_scores.json'
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f'[OK] LCB: wrote {len(output)} items → {out_path}')

    # Quick sanity summary
    easy = sum(1 for r in output if r['avg_score'] >= 0.8)
    disc = sum(1 for r in output if 0.2 < r['avg_score'] < 0.8)
    hard = sum(1 for r in output if r['avg_score'] <= 0.2)
    print(f'     Tiers → easy={easy}, discriminating={disc}, hard={hard}')


# ---------------------------------------------------------------------------
# AA-LCR
# ---------------------------------------------------------------------------

def build_aa_lcr_scores(reviews_dir: Path, out_dir: Path) -> None:
    """
    Aggregate per-sample accuracy scores across the three AA-LCR reference
    models and write data/aa_lcr_item_scores.json.

    Secondary metadata: ``input_tokens`` (context length) from sample_metadata.
    """
    model_files = {
        'gpt-oss-120b':   reviews_dir / 'aa_lcr__gpt-oss-120b.jsonl',
        'kimi-k2.5':      reviews_dir / 'aa_lcr__kimi-k2.5.jsonl',
        'minimax-m2.5':   reviews_dir / 'aa_lcr__minimax-m2.5.jsonl',
    }

    for model, path in model_files.items():
        if not path.exists():
            print(f'[WARN] Missing AA-LCR review file for {model}: {path}', file=sys.stderr)

    sample_scores: Dict[int, List[float]] = defaultdict(list)
    sample_metadata: Dict[int, Dict] = {}

    for model, path in model_files.items():
        if not path.exists():
            continue
        records = _load_jsonl(str(path))
        for rec in records:
            idx = int(rec['index'])
            score_val = rec['sample_score']['score']['value']
            score = float(score_val.get('acc', 0.0))
            sample_scores[idx].append(score)

            if idx not in sample_metadata:
                meta = rec.get('sample_score', {}).get('sample_metadata', {}) or {}
                sample_metadata[idx] = {
                    'question':     meta.get('question', ''),
                    'input_tokens': meta.get('input_tokens', 0),
                }

    output = []
    for idx in sorted(sample_scores.keys()):
        scores = sample_scores[idx]
        avg = sum(scores) / len(scores) if scores else 0.0
        output.append({
            'id':        idx,
            'avg_score': round(avg, 6),
            'variance':  round(_bernoulli_variance(avg), 6),
            'metadata':  sample_metadata.get(idx, {}),
        })

    out_path = out_dir / 'aa_lcr_item_scores.json'
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f'[OK] AA-LCR: wrote {len(output)} items → {out_path}')

    easy = sum(1 for r in output if r['avg_score'] >= 0.8)
    disc = sum(1 for r in output if 0.2 < r['avg_score'] < 0.8)
    hard = sum(1 for r in output if r['avg_score'] <= 0.2)
    print(f'     Tiers → easy={easy}, discriminating={disc}, hard={hard}')


# ---------------------------------------------------------------------------
# MMMU
# ---------------------------------------------------------------------------

def build_mmmu_scores(mmmu_reviews_dir: Path, out_dir: Path) -> None:
    """
    Build data/mmmu_stress_scores.json from the glm-4.5v-fp8 MMMU reviews.

    The stable sample identifier is the ``id`` field from sample_metadata
    (e.g. ``"validation_Electronics_11"``).
    """
    model_dir = mmmu_reviews_dir / 'glm-4.5v-fp8'
    if not model_dir.exists():
        print(f'[WARN] MMMU reviews directory not found: {model_dir}', file=sys.stderr)
        return

    output = []
    for jsonl_file in sorted(model_dir.glob('*.jsonl')):
        records = _load_jsonl(str(jsonl_file))
        for rec in records:
            ss = rec.get('sample_score', {})
            score_val = ss.get('score', {}).get('value', {})
            score = float(score_val.get('acc', 0.0))
            meta = ss.get('sample_metadata', {}) or {}

            sample_id = meta.get('id', rec.get('index', ''))
            output.append({
                'id':        str(sample_id),
                'avg_score': round(score, 6),
                # Only one reference model → variance is 0 for binary outcomes.
                # We set it to the mid-point to avoid all-zero weights.
                'variance':  round(_bernoulli_variance(score) if score not in (0.0, 1.0)
                                   else 0.25, 6),
                'metadata': {
                    'img_type':        meta.get('img_type', '[]'),
                    'subfield':        meta.get('subfield', ''),
                    'topic_difficulty': meta.get('topic_difficulty', ''),
                },
            })

    out_path = out_dir / 'mmmu_stress_scores.json'
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f'[OK] MMMU: wrote {len(output)} items → {out_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--evals-dir', required=True,
        help='Path to the Evals/ directory from the Cerebras assessment repo.'
    )
    parser.add_argument(
        '--out-dir', default='data',
        help='Output directory for the generated JSON files (default: data/).'
    )
    args = parser.parse_args()

    evals_dir = Path(args.evals_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    part1_reviews = evals_dir / 'Part 1' / 'reviews'
    mmmu_reviews  = evals_dir / 'MMMU'   / 'reviews'

    print('Building LCB item scores...')
    build_lcb_scores(part1_reviews, out_dir)

    print('\nBuilding AA-LCR item scores...')
    build_aa_lcr_scores(part1_reviews, out_dir)

    print('\nBuilding MMMU stress scores...')
    build_mmmu_scores(mmmu_reviews, out_dir)

    print('\nDone.  Run the pruned benchmarks to verify registration:')
    print('  python -c "from evalscope.api.registry import BENCHMARK_REGISTRY; '
          'print([k for k in BENCHMARK_REGISTRY if \'pruned\' in k])"')


if __name__ == '__main__':
    main()
