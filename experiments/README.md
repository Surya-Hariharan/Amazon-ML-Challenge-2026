# experiments/

Scaffolding for human-curated experiment records, separate from the machine-generated
`experiments.csv` (see [`../docs/experiments/README.md`](../docs/experiments/README.md)
for how the two relate). All four subdirectories are currently empty (each holds only a
`.gitkeep` placeholder) — nothing has been run into them yet as of this reorganisation.

| Directory | Purpose | Tracked in git? |
| --- | --- | --- |
| `configs/` | Saved `config.py`-equivalent snapshots (e.g. a JSON/YAML dump of the tunables in `config.TUNABLES`) for a run worth reproducing later. Small, human-readable. | Yes |
| `reports/` | Short written summaries of an experiment round (e.g. the CP6 error-analysis writeup, a LOCO robustness summary). Small, human-readable. | Yes |
| `logs/` | Raw run logs / stdout captures from pipeline invocations. Can grow large and is regenerated freely. | No — gitignored (`experiments/logs/*`) |
| `results/` | Exported metric tables, plots, or larger result dumps tied to a specific experiment. Can grow large. | No — gitignored (`experiments/results/*`) |

`configs/` and `reports/` are kept trackable because they are expected to stay small and
are meant to be read by teammates; `logs/` and `results/` are ignored because they are
regenerable and can grow without bound — consistent with how `artifacts/` (caches,
trained models, OOF predictions) and `experiments.csv` are already gitignored per
`CLAUDE.md` §3/§5.
