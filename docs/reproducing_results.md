# Results Reproduction Guide

This document maps each paper table and figure to the exact command, expected output location, and the metric source in code.

---

## Prerequisites

```bash
source venv/bin/activate
python -c "from env.GridEnv import GridEnv; print('OK')"   # smoke test
```

All commands below use Training mode (`python main.py 1 ...`). Replace `1` with `2` for a quick dev-mode sanity check first.

---

## Table 1 — Main Performance Comparison

**Paper content:** Test reward, charging cost, final SOC, voltage violation rate, communication overhead — HFDRL vs all baselines, mean ± CI95 over 10 seeds.

**Command:**

```bash
python main.py 1 --simulation 16
# Select: 10 seeds, all methods
```

**Output location:**

```
results/multi_seed/<timestamp>/aggregated/tables/performance_table.{tex,csv}
```

**Expected values (to be filled after full run):**

| Method | Test reward | Cost ($/ep) | Final SOC | Volt. viol. (%) |
|--------|------------|------------|-----------|-----------------|
| HFDRL | — ± — | — ± — | — ± — | — ± — |
| SWIFT-SAC | — ± — | — ± — | — ± — | — ± — |
| FedAvg-SAC | — ± — | — ± — | — ± — | — ± — |
| SAC Local-Only | — ± — | — ± — | — ± — | — ± — |
| SAC Centralized Oracle | — ± — | — ± — | — ± — | — ± — |
| Random | — ± — | — ± — | — ± — | — ± — |
| Greedy | — ± — | — ± — | — ± — | — ± — |
| Price-Aware | — ± — | — ± — | — ± — | — ± — |

**Metric source in code:**

| Column | `EvalMetrics` method | `metrics.json` key |
|--------|--------------------|--------------------|
| Test reward | `log_episode(r, mode='test')` | `mean_test_reward` |
| Cost | `log_cost(c)` | `mean_test_cost` |
| Final SOC | `log_final_soc(soc_list)` | `final_soc_mean` |
| Voltage violations | `log_voltage_violations(n)` | `voltage_violation_rate` |
| Comm. overhead | `log_comm_overhead(ms)` | `comm_overhead_ms` |

**Statistical tests:** `results/multi_seed/<timestamp>/aggregated/statistics/significance_report.json`

---

## Table 2 — Significance Tests

**Paper content:** p-values for paired t-test and Wilcoxon signed-rank test, HFDRL vs each baseline.

**Command:** Same run as Table 1.

**Output location:**

```
results/multi_seed/<timestamp>/aggregated/tables/significance_table.{tex,csv}
```

**Tolerance:** A result is considered significant at p < 0.05. Both tests must agree for a strong claim.

---

## Figure 1 — Reward Convergence Curves

**Paper content:** Mean ± CI95 training reward over episodes, HFDRL vs FedAvg-SAC vs SWIFT-SAC.

**Command:** Same run as Table 1.

**Output location:**

```
results/multi_seed/<timestamp>/aggregated/plots/reward_convergence.{pdf,png}
```

**Metric source:** `reward_curve.npy` in each `<Method>/seed_N/` directory.

---

## Figure 2 — Dwell-Time Sensitivity

**Paper content:** Test reward vs dwell-time window (1h, 2h, 4h, 6h, 8h) for FedAvg-SAC, SWIFT-SAC, HFDRL.

**Command:**

```bash
python main.py 1 --simulation 17
# Select: Full study, 10 seeds, all 5 dwell windows, all 3 methods
```

**Output location:**

```
results/dwell_time_study/<timestamp>/
```

The study runner aggregates across seeds per (method, dwell) combination and produces comparison plots.

**Expected pattern:** HFDRL and SWIFT-SAC should degrade less than FedAvg-SAC as dwell windows shrink, demonstrating SWIFT's benefit under time constraints.

---

## Figure 3 — Communication Overhead vs Bandwidth

**Paper content:** Communication overhead (MB/round or ms/round) vs available bandwidth for HFDRL vs HFDRL+LoRA across 5 BW scenarios.

**Command:**

```bash
python main.py 1 --simulation 18
# Select: Full study, 10 seeds, both methods
```

**Output location:**

```
results/lora_network_study/<timestamp>/
  <method>/seed_N/bw_sweep.json   ← per-scenario overhead
```

The study has two phases:
1. **RL training phase** — trains both methods to convergence and saves RL metrics.
2. **Analytical BW sweep** — loads trained models, measures parameter dict sizes, computes transfer time across 5 BW scenarios without re-running training.

**Expected pattern:** HFDRL+LoRA achieves the same test reward as HFDRL (no LoRA) but with ~94 % lower parameter transmission per FL round.

---

## Figure 4 — Robustness to Forecast Error

**Paper content:** Test reward degradation vs price forecast noise σ ∈ {0, 0.02, 0.05, 0.10, 0.20} for HFDRL vs FedAvg-SAC vs SWIFT-SAC.

**Command:**

```bash
python main.py 1 --simulation 19
# Select: Forecast Error only, 5 seeds, all 3 methods
```

Or via CLI:

```bash
for sigma in 0.0 0.02 0.05 0.10 0.20; do
  for method in FedAvg-SAC SWIFT-SAC HFDRL; do
    for seed in 0 1 2 42 123; do
      python main.py 1 --simulation stressTest \
        --sub-study forecast_error \
        --scenario $sigma \
        --method $method \
        --seed $seed
    done
  done
done
```

**Output location:**

```
results/stress_test_study/<timestamp>/forecast_error/<sigma>/seed_N/metrics.json
```

**Expected pattern:** HFDRL should be more robust than FedAvg-SAC at high noise levels due to the adaptive client selection in SWIFT.

---

## Figure 5 — Robustness to Non-IID Data

**Paper content:** Test reward vs Dirichlet heterogeneity α ∈ {1000, 10, 1.0, 0.5, 0.1}.

**Command:**

```bash
python main.py 1 --simulation 19
# Select: Non-IID only, 5 seeds, all 3 methods, NHTS archetypes
```

**Output location:**

```
results/stress_test_study/<timestamp>/non_iid/<alpha>/seed_N/metrics.json
```

**Expected pattern:** HFDRL's diversity utility in SWIFT should mitigate performance degradation as data heterogeneity increases (lower α = more heterogeneous).

---

## Reproducing a Single Number

To reproduce a single cell from Table 1 (e.g., HFDRL test reward, seed 42):

```bash
python main.py 1 --simulation 10   # Federated + SWIFT + LoRA
# Choose: SAC, FedAvg, then wait for training to complete
# The test reward is printed at the end and saved in metrics.json
```

For exact reproducibility, you must also:
1. Use Training mode (not Dev mode)
2. Use the same `configs/training.yaml` (1 000 episodes, 20 agents, real prices)
3. Fix the random seed — the multi-seed runner does this automatically, but standalone modes do not set a seed

For fully seeded reproduction of any result, use mode 16 (multi-seed) which seeds `random`, `numpy`, `torch`, and `PYTHONHASHSEED` before each run.

---

## Run Time Estimates

| Mode | Config | Estimated time |
|------|--------|---------------|
| Single method, training mode | 1 000 ep, 20 agents | 30–90 min (CPU) |
| Mode 16, 5 seeds, filtered methods | training mode | ~4–8 hours |
| Mode 16, 10 seeds, all methods | training mode | ~24–48 hours |
| Mode 17 (dwell study), 10 seeds | 5 dwell × 3 methods | ~24 hours |
| Mode 19 (stress test), 5 seeds | 5 σ × 3 methods | ~8–12 hours |

Times depend strongly on hardware. GPU acceleration (CUDA) reduces SAC training time significantly; pandapower power flow runs on CPU regardless.
