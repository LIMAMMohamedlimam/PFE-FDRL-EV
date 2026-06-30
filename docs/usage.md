# Running HFDRL — Train, Evaluate, Reproduce

## Modes & Entry Point

`main.py` is the single entry point. It presents an interactive two-step menu:

1. **Mode** — Training (full config) or Development (fast config, 20 ep, 5 agents)
2. **Simulation** — one of 19 options

```bash
python main.py          # fully interactive
python main.py 1        # Training mode, then choose simulation interactively
python main.py 2        # Development mode, then choose simulation interactively
python main.py 1 --simulation 16   # Training mode, simulation 16 directly
```

---

## All 19 Simulation Modes

### Standalone agents (1–3)

```bash
python main.py 1 --simulation 1   # PPO (continuous)
python main.py 1 --simulation 2   # Q-Learning (discrete)
python main.py 1 --simulation 3   # SAC (off-policy, continuous)
```

### Federated training (4–10)

```bash
python main.py 1 --simulation 4   # Federated: interactive policy + aggregation choice
python main.py 1 --simulation 5   # Full comparison pipeline (all baselines)
python main.py 1 --simulation 6   # SAC + LoRA (standalone)
python main.py 1 --simulation 7   # PPO + LoRA (standalone)
python main.py 1 --simulation 8   # Federated + LoRA
python main.py 1 --simulation 9   # Federated + SWIFT client selection
python main.py 1 --simulation 10  # Federated + SWIFT + LoRA  ← full HFDRL
```

### Baselines (11–15)

```bash
python main.py 1 --simulation 11  # Baseline suite (reads methods from training.yaml)
python main.py 1 --simulation 12  # Heuristic baselines (Random/Greedy/EDF/Price-Aware/MPC)
python main.py 1 --simulation 13  # Federated variants (FedProx / FedAvgM / FedAdam)
python main.py 1 --simulation 14  # SAC Local-Only (no federation)
python main.py 1 --simulation 15  # SAC Centralized Oracle (upper bound)
```

### AAAI statistical studies (16–19)

```bash
python main.py 1 --simulation 16  # Multi-seed statistical evaluation
python main.py 1 --simulation 17  # SWIFT dwell-time study
python main.py 1 --simulation 18  # LoRA network constraints study
python main.py 1 --simulation 19  # Robustness stress tests
```

---

## Common Workflows

### Quick sanity check (< 1 min)

```bash
python main.py 2 --simulation 3   # Dev mode → SAC, 20 ep, 5 agents
```

### Run the main HFDRL method

```bash
python main.py 1 --simulation 10  # Federated + SWIFT + LoRA
```

The interactive prompts will ask for the RL policy (SAC) and aggregation strategy (FedAvg). For non-interactive use, add the SWIFT + LoRA flags programmatically via `ComparisonPipeline.run_single_experiment(policy='sac', aggregation='fedavg', use_swift=True, use_lora=True)`.

### Run all baselines for comparison (mode 11)

```bash
python main.py 1 --simulation 11
```

This runs every entry in the `methods` list of `configs/training.yaml`. Results land in `results/` with a timestamp prefix.

### Run the multi-seed statistical evaluation

```bash
# Full (10 seeds, all methods — takes several hours)
python main.py 1 --simulation 16

# Filtered (HFDRL + key baselines, 5 seeds — faster smoke test)
python main.py 2 --simulation 16   # Dev mode for ultra-fast check
```

### Run the stress-test study (non-interactive CLI)

```bash
# Forecast-error sub-study, HFDRL, σ=0.05, seed=0
python main.py 1 --simulation stressTest \
  --sub-study forecast_error \
  --scenario 0.05 \
  --method HFDRL \
  --seed 0

# Non-IID sub-study, FedAvg-SAC, α=0.5, NHTS archetypes, seed=42
python main.py 1 --simulation stressTest \
  --sub-study non_iid \
  --scenario 0.5 \
  --method FedAvg-SAC \
  --seed 42 \
  --archetype nhts
```

---

## Output Layout

Every simulation run writes outputs under `results/`. The exact sub-directory depends on the mode:

| Mode | Output location |
|------|----------------|
| Standalone / pipeline (1–10) | `results/<run_name>_<timestamp>/` |
| Baseline suite (11–15) | `results/<run_name>_<timestamp>/` per method |
| Multi-seed (16) | `results/multi_seed/<timestamp>/` |
| Dwell-time study (17) | `results/dwell_time_study/<timestamp>/` |
| LoRA network study (18) | `results/lora_network_study/<timestamp>/` |
| Stress-test study (19) | `results/stress_test_study/<timestamp>/` |

### Per-run files (modes 1–15)

```
results/<run_name>_<timestamp>/
├── metrics.png          6-panel dashboard (reward, cost, SOC, grid, train vs test)
├── metrics.json         Scalar summary (final test reward, cost, SOC, stability)
└── simulation_registry.json  Cross-run registry entry
```

### Multi-seed run (mode 16)

```
results/multi_seed/<timestamp>/
├── reproducibility_notes.json
├── aggregated_raw.json
├── <Method>/seed_N/
│   ├── reward_curve.npy
│   ├── cost_curve.npy
│   └── metrics.json
└── aggregated/
    ├── tables/
    │   ├── performance_table.tex
    │   ├── performance_table.csv
    │   ├── significance_table.tex
    │   └── significance_table.csv
    ├── plots/           8 PDF + PNG figures
    └── statistics/
        ├── full_stats.json
        └── significance_report.json
```

---

## Viewing Results

```bash
python app.py
# Open http://127.0.0.1:5000
```

The Flask gallery lists all PNG files under `results/` sorted by timestamp. Use it to compare runs side by side.

---

## Resuming / Using Checkpoints

Model checkpoints are saved to `results/trained_models/`. To load a checkpoint into a SACAgent:

```python
from agents.SACAgent import SACAgent
import torch

agent = SACAgent(input_dim=13, action_dim=1)
state = torch.load('results/trained_models/<run>.pt', map_location='cpu')
agent.actor.load_state_dict(state['actor'])
agent.critic.load_state_dict(state['critic'])
```

---

## Configuration Overrides

All tunable values live in `configs/`. For a one-off experiment, edit the relevant YAML before launching. For example, to run 2 000 episodes with 30 agents:

```yaml
# configs/training.yaml
num_episodes: 2000
num_agents: 30
```

See [`docs/configuration.md`](configuration.md) for the complete parameter reference.

---

## Development Mode

All modes support `python main.py 2 --simulation <N>` (mode=2 = Development). This loads `configs/training_dev.yaml` instead of `configs/training.yaml`:

| Parameter | Training | Dev |
|-----------|---------|-----|
| `num_episodes` | 1 000 | 20 |
| `num_test_episodes` | 200 | 1 |
| `num_agents` | 20 | 5 |
| `fl_rounds` | 6 | 1 |
| Price data | `iso_ne_prices_real.csv` | `iso_ne_prices_dev_test.csv` |

Always test a new config or code change in Dev mode first.
