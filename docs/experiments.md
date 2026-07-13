# Experiment Tracking & Outputs

## 1. Run Directory Layout

Every simulation writes to a timestamped directory under `results/`. The exact sub-path depends on the mode:

```
results/
│
├── <run_name>_<timestamp>/          ← modes 1–15 (standalone / pipeline)
│   ├── metrics.png                  6-panel dashboard
│   ├── metrics.json                 Scalar summary for this run
│   └── [optional] episodes.csv      Per-episode reward, cost, SOC
│
├── multi_seed/<timestamp>/          ← mode 16 (multi-seed statistical evaluation)
│   ├── reproducibility_notes.json   RNG versions, CI formula, seed list
│   ├── aggregated_raw.json          All method × seed scalar metrics
│   ├── <Method_Name>/
│   │   └── seed_N/
│   │       ├── reward_curve.npy     Episode rewards (numpy, shape [n_episodes])
│   │       ├── cost_curve.npy       Episode costs   (numpy, shape [n_episodes])
│   │       └── metrics.json         Per-seed scalar summary
│   └── aggregated/
│       ├── tables/
│       │   ├── performance_table.tex   LaTeX table (mean ± CI95)
│       │   ├── performance_table.csv
│       │   ├── significance_table.tex  p-values vs HFDRL
│       │   └── significance_table.csv
│       ├── plots/                   8 figures: PDF + PNG
│       └── statistics/
│           ├── full_stats.json      Mean, std, CI95 per method per metric
│           └── significance_report.json  t-test + Wilcoxon results
│
├── dwell_time_study/<timestamp>/    ← mode 17
│   └── <method>/<dwell>h/seed_N/   same per-seed structure as multi_seed
│
├── lora_network_study/<timestamp>/  ← mode 18
│   └── <method>/seed_N/
│       ├── metrics.json             RL phase metrics
│       └── bw_sweep.json            Analytical BW sweep results per scenario
│
├── stress_test_study/<timestamp>/   ← mode 19
│   └── <sub_study>/<scenario>/seed_N/
│
└── trained_models/                  Model checkpoints (saved by SACAgent)
```

---

## 2. `metrics.json` Schema

Every per-run `metrics.json` (modes 1–15) contains:

```json
{
  "run_name":         "SAC_HFedAvg_20260630_142301",
  "method":           "SAC HFedAvg SWIFT LoRA",
  "final_test_reward": 14.72,
  "mean_test_reward":  13.85,
  "mean_train_cost":   2.34,
  "mean_test_cost":    2.18,
  "grid_stability":    0.042,
  "mean_satisfaction": 0.94,
  "voltage_violation_rate": 0.003,
  "comm_overhead_ms":  null,
  "config": { ... }
}
```

Per-seed `metrics.json` (modes 16–19) additionally has:

```json
{
  "seed":              42,
  "convergence_episode": 312,
  "final_soc_mean":    0.91,
  "final_soc_std":     0.04,
  ...
}
```

---

## 3. `reproducibility_notes.json` Schema

Written at the start of every multi-seed run:

```json
{
  "seeds":          [0, 1, 2, 3, 4, 42, 123, 456, 789, 999],
  "python_version": "3.11.2",
  "torch_version":  "2.2.0",
  "numpy_version":  "1.26.4",
  "ci_z_score":     1.96,
  "ci_formula":     "mean ± z * std / sqrt(n_seeds)",
  "rng_control":    ["random.seed", "numpy.seed", "torch.seed", "PYTHONHASHSEED"],
  "timestamp":      "2026-06-30T14:23:01Z"
}
```

---

## 4. Metrics Logged by EvalMetrics

**Code location:** `utils/EvalMetrics.py` — class `EvalMetrics`.

| Method | What is logged | Stored in |
|--------|---------------|-----------|
| `log_episode(r, mode)` | Episode reward | `episode_rewards` / `test_rewards` |
| `log_cost(c)` | Energy cost ($/episode) | `episode_costs` |
| `log_step(power_mw)` | Instantaneous grid power | internal list for σ_g |
| `log_satisfaction(sats)` | SOC_final / SOC_req per agent | `satisfaction_history` |
| `log_voltage_violations(n)` | Count of buses outside [V_min, V_max] | `voltage_violations` |
| `log_comm_overhead(ms)` | Communication time per FL round | `comm_overhead_ms` |
| `log_final_soc(soc_list)` | Final SOC of each agent in test episode | `final_soc_per_test` |
| `compute_stability_metric()` | σ_g = std of grid power changes | returned, not stored |

### 6-panel dashboard (`metrics.png`)

| Panel | Content |
|-------|---------|
| 1 | Episode reward (train) — convergence curve |
| 2 | Energy cost per episode |
| 3 | Client satisfaction (SOC_final / SOC_req) |
| 4 | Grid stability — σ_g over training |
| 5 | Train vs test reward — boxplot |
| 6 | Summary table (scalar metrics) |

---

## 5. Statistical Evaluation Outputs (Mode 16)

`utils/StatisticalAnalysis.py` computes:

**Confidence intervals:**

```
CI95(metric) = mean ± 1.96 × std / √n_seeds
```

**Significance tests (HFDRL vs each baseline):**

- Paired t-test (`scipy.stats.ttest_rel`)
- Wilcoxon signed-rank test (`scipy.stats.wilcoxon`)

Both tests are paired because the same seeds are used for all methods — the pairing controls for seed-to-seed variance.

The output `significance_report.json` reports t-statistic, p-value, and whether p < 0.05 for each baseline comparison on each metric.

**8 AAAI figures (in `aggregated/plots/`):**

| Figure | Content |
|--------|---------|
| `reward_convergence.{pdf,png}` | Mean ± CI95 reward curves (all methods) |
| `test_reward_bar.{pdf,png}` | Test reward bar chart with CI95 error bars |
| `cost_bar.{pdf,png}` | Charging cost comparison |
| `soc_bar.{pdf,png}` | Final SOC comparison |
| `voltage_bar.{pdf,png}` | Voltage violation rate |
| `comm_overhead.{pdf,png}` | Communication overhead (ms/round) |
| `significance_heatmap.{pdf,png}` | p-value heatmap vs HFDRL |
| `convergence_speed.{pdf,png}` | Episode at convergence per method |

---

## 6. Checkpoints

**Saved by:** `SACAgent.save_trained_model(directory, agent_id)`.

**Full model checkpoint** (no LoRA):

```
results/trained_models/sac_agent_<id>.pth
  Keys: actor (state_dict), critic (state_dict),
        critic_target (state_dict), log_alpha, alpha
```

**LoRA checkpoint:**

```
results/trained_models/sac_agent_lora_<id>_<timestamp>.pth
  Keys: actor.<layer>.lora_A, actor.<layer>.lora_B,
        critic.<layer>.lora_A, critic.<layer>.lora_B
```

**Loading a checkpoint:**

```python
agent = SACAgent(input_dim=13, action_dim=1)
agent.load_trained_model('results/trained_models/sac_agent_0.pth')
```

The `load_trained_model()` method auto-detects the checkpoint format (full vs LoRA) from the key structure.

---

## 7. Cross-Run Comparison

`results/simulation_registry.json` accumulates one entry per completed run (modes 1–15). Each entry records the run name, method, config, and scalar metrics. This lets you compare runs without re-reading individual `metrics.json` files:

```bash
python -c "
import json
with open('results/simulation_registry.json') as f:
    registry = json.load(f)
for run in registry:
    print(run['run_name'], run['final_test_reward'])
"
```

For mode 16 outputs, use `aggregated_raw.json` instead, which has the full method × seed × metric matrix.
