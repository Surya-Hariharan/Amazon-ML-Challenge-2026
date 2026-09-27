# experiments/

Standalone experiment drivers that run the unmodified pipeline and record results. They
never modify `code/src/` or write to `output/`.

| Script | Purpose | Output |
| --- | --- | --- |
| [`scripts/run_baseline_experiment.py`](scripts/run_baseline_experiment.py) | Runs `run_pipeline.valid_run` on a seeded S1 sample and dumps every metric plus extra diagnostics (feature separability, threshold sweep, error budget, resource usage) | `results/<tag>.json` |
| [`scripts/source_recall.py`](scripts/source_recall.py) | Blocking recall split by candidate source (S2 vs S3) | `results/n10000_source_recall.json` |
| [`scripts/make_figures.py`](scripts/make_figures.py) | Plots from the JSON results (requires `matplotlib`) | `reports/figures/*.png` |

```bash
python experiments/scripts/run_baseline_experiment.py 10000 n10000   # <n_s1> <tag>
python experiments/scripts/make_figures.py
```

`logs/`, `results/` and `reports/` hold generated files and are not versioned. A summary
of the results lives in [`docs/results.md`](../docs/results.md); in-pipeline diagnostics
(`python -m src.diagnose`) write to `code/artifacts/diagnostics/` instead.
