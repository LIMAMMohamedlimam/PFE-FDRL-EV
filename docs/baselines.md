# Baselines & Comparison

HFDRL is compared against three categories of baselines. Each category isolates a different design choice.

---

## Category 1 — Heuristic Baselines

Rule-based agents with no learning. They implement the `BaseAgent` interface with no-op FL methods (`get_parameters` returns `{}`, `set_parameters` is a no-op).

**Code location:** `agents/HeuristicAgents.py`

| Name | Class | Policy |
|------|-------|--------|
| Random | `RandomAgent` | Uniform random action ∈ [−1, 1] |
| Greedy | `GreedyAgent` | Charge at P_max when SOC < soc_req, else idle |
| EDF | `EarliestDeadlineFirst` | Charge proportional to urgency = (soc_req − SOC) / t_remaining |
| Price-Aware | `PriceAwareAgent` | Full charge when price < low_threshold; linear interpolation; idle when price > high_threshold |
| Simple MPC | `SimpleMPCAgent` | 1-step lookahead over {−1, 0, +1}: minimises cost_weight × price × |a| − soc_weight × ΔSOC |

**What they isolate:** A lower bound on performance and an indicator of how much learned policy adds over simple rules.

**How to run:**

```bash
python main.py 1 --simulation 12   # interactive heuristic selection
```

Or from the baseline suite (all heuristics run together):

```bash
python main.py 1 --simulation 11   # reads methods from training.yaml
```

The `methods` list in `configs/training.yaml` includes `Random`, `Greedy`, `EDF`, `Price-Aware`, `Simple MPC`.

---

## Category 2 — RL Standalone Baselines

RL agents without federation, or with a degenerate (centralized) federation.

### SAC Local-Only

Each agent trains independently with no parameter sharing. This measures the cost of distributing training without any communication.

```bash
python main.py 1 --simulation 14
```

**Code path:** `ComparisonPipeline.run_single_experiment(policy='sac', aggregation='none')`

### SAC Centralized Oracle

All agents share a single network and a single replay buffer. This is an upper bound — it requires sharing all raw data (not just model parameters), which is infeasible for privacy reasons in practice.

```bash
python main.py 1 --simulation 15
```

**Code path:** `ComparisonPipeline.run_single_experiment(policy='sac', aggregation='none', centralized=True)`

### PPO (standalone)

Standard PPO without federation.

```bash
python main.py 1 --simulation 1
```

### Q-Learning (standalone)

Tabular Q-Learning with discrete actions {idle, half-charge, full-charge}. A weak baseline that cannot represent the continuous action space.

```bash
python main.py 1 --simulation 2
```

**What they isolate:** The benefit of federated learning (Local-Only vs any federated method) and the performance ceiling (Centralized Oracle).

---

## Category 3 — Federated Aggregation Variants

SAC with different FL aggregation strategies. All use `num_agents = 20` and `num_edges = 2` (hierarchical). No SWIFT, no LoRA.

**Code location:** `training/FederatedServer.py`

| Name | Strategy | Key difference from FedAvg |
|------|---------|---------------------------|
| FedAvg-SAC | `fedavg` | Weighted average; no server momentum |
| FedProx | `fedprox` | Adds proximal penalty μ/2‖w−w_g‖² client-side |
| FedAvgM | `fedavgm` | Server-side momentum on the pseudo-gradient |
| FedAdam | `fedadam` | Server-side Adam on the pseudo-gradient |

**How to run interactively:**

```bash
python main.py 1 --simulation 13   # FedProx / FedAvgM / FedAdam with optional LoRA + SWIFT
```

**How to run all from config (baseline suite):**

```bash
python main.py 1 --simulation 11   # reads training.yaml → methods
```

The `methods` block in `configs/training.yaml` (Category 2 entries):

```yaml
- name: "SAC FedProx"
  agent: sac
  federated:
    enabled: true
    aggregation: fedprox
    mu_fedprox: 0.01

- name: "SAC FedAvgM"
  agent: sac
  federated:
    enabled: true
    aggregation: fedavgm
    beta_momentum: 0.9

- name: "SAC FedAdam"
  agent: sac
  federated:
    enabled: true
    aggregation: fedadam
    adam_lr: 0.01
```

**What they isolate:** The choice of FL aggregation strategy. HFDRL uses `fedavg` at both edge and cloud level; the variants test whether more sophisticated server-side optimisation helps in the EV charging setting.

---

## HFDRL Ablation Variants

The three AAAI study methods (modes 17 and 18) form an ablation ladder:

| Method | Components | Ablates |
|--------|-----------|---------|
| `FedAvg-SAC` | SAC + FedAvg, no SWIFT, no LoRA | − SWIFT and − LoRA |
| `SWIFT-SAC` | SAC + FedAvg + SWIFT, no LoRA | − LoRA |
| `HFDRL` | SAC + FedAvg + SWIFT + LoRA | full system |

This isolates the contribution of each component: SWIFT client selection (SWIFT-SAC vs FedAvg-SAC) and LoRA compression (HFDRL vs SWIFT-SAC).

**How to run the full ablation (dwell-time study):**

```bash
python main.py 1 --simulation 17   # interactive: methods, dwell windows, seeds
```

---

## Reading Comparison Results

### From the multi-seed statistical evaluation (mode 16)

After running mode 16, the performance table is at:

```
results/multi_seed/<timestamp>/aggregated/tables/performance_table.csv
```

Columns: `Method`, `test_reward_mean`, `test_reward_ci95`, `cost_mean`, `cost_ci95`, `soc_mean`, `soc_ci95`, `voltage_violation_rate_mean`, `comm_overhead_ms_mean`.

The significance table (`significance_table.csv`) shows which baselines are significantly worse than HFDRL (paired t-test + Wilcoxon, p < 0.05).

### From the results gallery

```bash
python app.py   # → http://127.0.0.1:5000
```

All run PNGs are shown sorted by timestamp. Each 6-panel dashboard can be compared visually.

### From `simulation_registry.json`

```bash
python -c "
import json, sys
with open('results/simulation_registry.json') as f:
    runs = json.load(f)
for r in sorted(runs, key=lambda x: x.get('final_test_reward', 0), reverse=True):
    print(f\"{r['run_name']:50s}  reward={r.get('final_test_reward','?'):.2f}\")
"
```
