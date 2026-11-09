# Reproducibility of the HFDRL charging experiments

This guide accompanies Section 6.2 of *Communication–performance trade-offs in hierarchical federated reinforcement learning for electric vehicle charging*. It gives two routes to the results. Route A rebuilds the summaries from the supplied per-run records. Route B reruns training and evaluation. The two campaigns use separate training budgets.

## Inputs and environment

| Input | Purpose |
| --- | --- |
| `HFDRL_Comparaison_1000_episodes.zip` | Section 6.2.1, Tables 6 and 7. Includes 15 learned-policy runs, the immediate-charging rule, communication records and the LoRA initialization audit. |
| `HFDRL_Evaluation_Ablation_Reward.zip` | Section 6.2.2, Tables 8 and 9. Includes the corrected 200-episode pilot in `results_feasible/` and the initial-simulator diagnostic in `results/`. |
| `PFE-FDRL-EV` | Public simulator and tariff CSV at Git commit `12a4ff452f157d57b554f1a618ed6f1d6f004e0f`. |

The runs used Linux x86-64, Python 3.12.14 and PyTorch 2.7.1 CPU. The pinned pandapower distribution is 2.14.11. Its internal version string is 2.14.10, which is what the archived JSON records report. The two archives contain the same pinned direct dependencies.

### Setup

Start in a directory containing both ZIP files. The commands below use Bash, Git, unzip and Python 3.12. Keep this shell open for either route.

~~~bash
set -euo pipefail
ROOT="$(pwd)"
COMMIT=12a4ff452f157d57b554f1a618ed6f1d6f004e0f

sha256sum -c <<'SHA'
82a044d71ba7d9dbc46fdcf47c3f955cb566a8a3f920c31e2897e02f7ddb66cf  HFDRL_Comparaison_1000_episodes.zip
aef7d2610ac399c7f872d294c9e270907988e703aa7af5a409f80886b55c75f4  HFDRL_Evaluation_Ablation_Reward.zip
SHA

unzip -q HFDRL_Comparaison_1000_episodes.zip -d "$ROOT/reference"
unzip -q HFDRL_Evaluation_Ablation_Reward.zip -d "$ROOT/pilot"

git clone https://github.com/LIMAMMohamedlimam/PFE-FDRL-EV.git "$ROOT/PFE-FDRL-EV"
git -C "$ROOT/PFE-FDRL-EV" checkout --detach "$COMMIT"
printf '%s  %s\n' \
  '61a6e7c23942bd6a09e0c351adf5e2726bbace28231eccdbe2d33ac4ff83402a' \
  "$ROOT/PFE-FDRL-EV/data/iso_ne_prices_real.csv" | sha256sum -c -

python3.12 -m venv "$ROOT/.venv"
. "$ROOT/.venv/bin/activate"
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "$ROOT/reference/requirements.txt"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~~~

## Route A · Rebuild the manuscript summaries

This route uses the raw JSON from the two archives. It writes to separate directories and leaves the extracted records intact.

~~~bash
cd "$ROOT"
mkdir -p rebuilt/reference rebuilt/pilot rebuilt/diagnostic
cp reference/results/*.json rebuilt/reference/
cp pilot/results_feasible/*.json rebuilt/pilot/
cp pilot/results/*.json rebuilt/diagnostic/

python reference/summarize_reference.py \
  --repo "$ROOT/PFE-FDRL-EV" --directory "$ROOT/rebuilt/reference"

python pilot/summarize_grid_reward_study.py --directory "$ROOT/rebuilt/pilot"
python pilot/summarize_grid_reward_study.py --directory "$ROOT/rebuilt/diagnostic"

for version in pilot diagnostic; do
  python pilot/audit_energy_balance.py --repo "$ROOT/PFE-FDRL-EV" \
    --directory "$ROOT/rebuilt/$version"
done

python pilot/report_tables.py \
  --directory "$ROOT/rebuilt/pilot" --output "$ROOT/rebuilt/pilot/tables.md"
python pilot/report_tables.py --original \
  --directory "$ROOT/rebuilt/diagnostic" --output "$ROOT/rebuilt/diagnostic/tables.md"
~~~

The base checkout suffices in this route because the summary and energy-audit scripts only use it to reconstruct the deterministic vehicle profiles. Boundary-test records and the LoRA initialization audit are read from the archives.

| Manuscript output | Rebuilt data |
| --- | --- |
| Table 6, final charging outcomes | `rebuilt/reference/summary.json` → `estimates`. The two attainment columns use `nominal` and `stress`. The remaining columns use `stress`. Paired contrasts are under `full_minus_reference`. |
| Table 7, communication | `rebuilt/reference/threshold.csv`. Use the mean final vehicle upload in GiB and the medians of episode and upload MiB among the first observed crossings. One MiB is 2²⁰ bytes and one GiB is 2³⁰ bytes. |
| Table 8, corrected pilot outcomes | `rebuilt/pilot/tables.md`, increased-load panel. The numeric source is `rebuilt/pilot/summary.csv`. |
| Table 9, corrected pilot contrasts | `rebuilt/pilot/tables.md`, paired contrasts under increased load. The numeric source is `rebuilt/pilot/paired_differences.csv`. |

The following check extracts the main results directly from the rebuilt reference summary.

~~~bash
python - <<'PY'
import json
from pathlib import Path
from statistics import mean, median

s = json.loads(Path("rebuilt/reference/summary.json").read_text())
assert s["trained_runs"] == 15
assert s["test_episodes_including_heuristic"] == 960
assert s["failed_test_power_flow_steps"] == 0
assert s["maximum_absolute_energy_residual_kwh"] < 1e-7

for mode, methods in s["estimates"].items():
    assert all(row["target_met_pct"]["mean"] == 100 for row in methods.values()), mode

def first_crossings(method):
    rows = [r for r in s["threshold"] if r["method"] == method]
    assert len(rows) == 5 and all(r["reached"] for r in rows)
    return {r["seed"]: r for r in rows}

full = first_crossings("hfdrl_full")
dense = first_crossings("hfdrl_no_lora")
key = "first_observed_vehicle_uplink_tensor_bytes"
savings = 100 * mean(1 - full[seed][key] / dense[seed][key] for seed in sorted(full))
assert round(savings, 1) == 82.8
print(f"Paired upload reduction at the 80% threshold: {savings:.1f}%")
for label, rows in [("HFDRL with LoRA", full), ("HFDRL without LoRA", dense)]:
    mib = median(r[key] for r in rows.values()) / 2**20
    print(f"{label}: median first-crossing payload {mib:.1f} MiB")
PY
~~~

Expected medians are 125.2 MiB with LoRA and 764.7 MiB without LoRA. The 82.8% reduction is computed from the relative differences within seed pairs.

For the corrected pilot, verify 20 runs and 1,200 evaluation episodes, zero attained targets for full HFDRL and 31.8% mean attainment for HFDRL with the explicit grid term removed, rounded to one decimal. For a direct battery-boundary test, use the patched simulator in Route B.

## Route B · Rerun the experiments

Start from the completed setup. Create separate worktrees from the pinned commit. Apply the patches in order.

~~~bash
cd "$ROOT"
git -C PFE-FDRL-EV worktree add --detach "$ROOT/sim-reference" "$COMMIT"
git -C sim-reference apply "$ROOT/reference/pipeline.patch"
git -C sim-reference apply "$ROOT/reference/battery_feasibility.patch"
git -C sim-reference apply "$ROOT/reference/round_accounting.patch"

git -C PFE-FDRL-EV worktree add --detach "$ROOT/sim-pilot" "$COMMIT"
git -C sim-pilot apply "$ROOT/pilot/pipeline.patch"
git -C sim-pilot apply "$ROOT/pilot/battery_feasibility.patch"
~~~

### Reference budget

Use a fresh output directory. The campaign script reuses completed files from its own `results/` directory, so running it from the original extracted folder would reuse the archived runs. The reference campaign took about 90 minutes with eight CPU workers in the original environment.

~~~bash
cd "$ROOT"
mkdir -p reference-rerun
cp reference/*.py reference/run_reference.sh reference-rerun/
WORKERS=8 PYTHON="$ROOT/.venv/bin/python" \
  bash reference-rerun/run_reference.sh "$ROOT/sim-reference"

python reference-rerun/audit_lora_initialization.py \
  --repo "$ROOT/sim-reference" \
  --output "$ROOT/reference-rerun/results/lora_initialization_audit.json"
python reference-rerun/summarize_reference.py \
  --repo "$ROOT/sim-reference" --directory "$ROOT/reference-rerun/results"
~~~

### Pilot and initial-simulator diagnostic

The corrected pilot uses four configurations and five seeds. To reproduce the initial-simulator diagnostic as a separate campaign, apply the instrumentation patch alone.

~~~bash
cd "$ROOT"
HFDRL_RESULTS_DIR="$ROOT/pilot-rerun/corrected" WORKERS=4 PYTHON="$ROOT/.venv/bin/python" \
  bash pilot/run_campaign.sh "$ROOT/sim-pilot"

git -C PFE-FDRL-EV worktree add --detach "$ROOT/sim-initial" "$COMMIT"
git -C sim-initial apply "$ROOT/pilot/pipeline.patch"

HFDRL_RESULTS_DIR="$ROOT/pilot-rerun/initial" WORKERS=4 PYTHON="$ROOT/.venv/bin/python" \
  bash pilot/run_campaign.sh "$ROOT/sim-initial"

python pilot/report_tables.py \
  --directory "$ROOT/pilot-rerun/corrected" --output "$ROOT/pilot-rerun/corrected/tables.md"
python pilot/report_tables.py --original \
  --directory "$ROOT/pilot-rerun/initial" --output "$ROOT/pilot-rerun/initial/tables.md"
~~~

Reduce `WORKERS` on machines with fewer cores or less memory. Route A reproduces the calculations from the saved outputs. Numerical differences can occur when training is rerun on different CPU and software configurations.

## Protocol and interpretation

| Setting | Reference comparison | Reward-ablation pilot |
| --- | --- | --- |
| Learned configurations | FedAvg-SAC, HFDRL without LoRA, full HFDRL | Same three, plus full HFDRL with the explicit grid term removed during training |
| Episodes per seed | 1,000 | 200 |
| Local replay warm-up | 2,000 transitions | 512 transitions |
| Local SAC update interval | 1 transition | 4 transitions |
| Minibatch | 256 | 256 |
| Grid term weight | 0.3 | 0.3 for the standard policies, 0 for the ablation during training, 0.3 for the common evaluation reward |
| Non-learning policy | Immediate charging capped at the requested SOC | Battery-only attainability check |

- **Shared design.** The five training seeds are 42, 123, 456, 789 and 1234. Both campaigns use an IEEE 33-bus feeder, 20 synthetic vehicle profiles per episode and 24 hourly steps. The scenario seed is 830515. Each of the 30 final test days is evaluated under nominal load and with an extra 0.8 MW of background load.
- **Price data.** The pinned `data/iso_ne_prices_real.csv` has 85,527 rows and nine entries per timestamp. The code averages the nine entries and drops the incomplete day. It fits the scaler and the hour-of-day tariff profile on 316 days ending 2022-05-13. The 79 subsequent days form the holdout. The final test and the communication monitor draw disjoint dates from this holdout. Realized normalized prices enter the cost index. The training-only hour-of-day profile supplies the price input to the agents.
- **Battery correction.** The patch bounds the signed action before the power-flow solve and uses the same power in the battery update. The feasible battery interval is `[-z C/(eta dt), (1-z) C/(eta dt)]` intersected with the charger limits, with SOC `z`, battery capacity `C`, efficiency `eta` and step duration `dt`. The check follows the existing symmetric efficiency convention. The initial-simulator outputs document the boundary defect and are kept separate.
- **Statistical unit.** First average the scenarios within each learned run and load condition. Then aggregate over seeds. The paired 95% Student intervals use within-seed differences. The deterministic immediate-charging rule is evaluated once per scenario and has direct access to the individual SOC target.
- **Service and voltage.** A departure succeeds when the SOC reaches the vehicle-specific target. The deficit includes all departures and is expressed in SOC percentage points. The voltage metric is the proportion of bus-hours outside [0.90, 1.10] p.u. Solver failures are reported separately. The cost index uses imported energy and normalized tariffs.
- **Traffic to a common target.** The monitor evaluates 12 held-out days under increased load every 100 training episodes after aggregation. Its scenario seed is 122783. The 80% threshold is a first observed crossing at a scheduled check. Final testing uses the episode-1,000 policies. Cumulative vehicle upload counts the byte sizes of the tensors sent by participating vehicles. Wire framing, retransmissions and downstream traffic are outside this measure.
- **Limits of the record.** The archives contain the per-run results, configuration and analysis code. Rerunning training is the route to learned policies. The model measurements apply to the implemented LoRA variant, whose critic bases remain vehicle-specific. Historical PFE results in the manuscript are separately attributed and have a different provenance.
