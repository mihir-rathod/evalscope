# Handout B — Why This Matters and How to Use It
### Task 2: Benchmark Compression · For Sales, Test Engineers, and PMs

---

## What Changes for the Customer Conversation

Before this work, if a customer asked us, "Is this model good enough for our specific workload?", getting them an answer meant running 315 coding problems and 100 long-context questions. That process takes hours of compute time and costs real money. That kind of timeline kills momentum in a sales cycle.

After deploying these pruners, you can get that exact same go/no-go answer in **minutes**, using roughly **10% of the compute**. 

This means a sales engineer can run the benchmark *live* during a proof-of-concept call, rather than setting it up overnight. The customer gets a concrete number on the screen before the meeting ends, not in a follow-up email two days later. 

---

## How to Actually Run This Tomorrow

I built these pruners directly into the existing `evalscope` framework, so there is no weird custom tooling to learn. 

**Step 1 — Install the fork**
```bash
git clone <your-fork-url> && cd evalscope
pip install -e ".[dev]"
```

**Step 2 — Run the fast version**
Just use the standard `evalscope eval` command you are already used to, but call the `_pruned` dataset:
```bash
# This will take ~minutes instead of hours
evalscope eval --model <model-endpoint> \
    --datasets live_code_bench_pruned \
    --dataset-args '{"live_code_bench_pruned": {"extra_params": {"prune_ratio": 0.1}}}' \
    --work-dir ./results_pruned/
```

**Step 3 — If you want to verify against the full run**
If you ever want to prove to a customer that the fast version works, you can run both and use the comparison tool I built:
```bash
evalscope eval --model <model-endpoint> \
    --datasets live_code_bench --work-dir ./results_full/

python -m evalscope_ext.tools.compare_runs \
    --full ./results_full/ --pruned ./results_pruned/
```

The `compare_runs` tool will print a table showing if the fast run ranked the models in the exact same order as the expensive full run. If you see a green checkmark, the quick answer matches the full answer. That is your definitive go/no-go.

---

## What the Multimodal Probe Gives That Random Sampling Cannot

If your customer says they might want to use vision capabilities next quarter, you need to know if the model's image *encoder* is actually good. 

If you just run a standard random sample of MMMU questions, you will miss the failure mode that actually matters. A model with a broken image encoder can still "pass" random text-heavy visual questions by guessing from the context. 

My `mmmu_pruned` probe specifically forces the model to look at **circuit diagrams, molecular structures, geometry proofs, and medical scans**. A model that fails these specifically has a broken image encoder. That is a completely different conversation with engineering, and a different procurement decision for the customer. A random 5% sample dilutes this signal 3-to-1. My probe concentrates it.

---

## Why a PM Should Care

* **Speed changes what is possible in a sales cycle.** A test that takes two hours cannot be run during a meeting. A test that takes ten minutes can. That changes evaluation from a "homework assignment" into a live, interactive demonstration.
* **The probe tells you *why* a model fails, not just *that* it fails.** Hearing "This model scores 60%" isn't actionable. Hearing "This model scores 60% on structured image understanding, but 80% on everything else" tells engineering exactly where to look, and tells the customer whether their specific use case (like parsing technical documentation) is affected.
* **It is already inside evalscope.** Your team doesn't have to install new software or maintain bespoke scripts. It runs with the exact same command-line interface they already use every day.
