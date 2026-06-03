# FDRL-EV — Federated Deep Reinforcement Learning for EV Charging Optimization

> A simulation framework implementing **Federated Deep Reinforcement Learning (FDRL)** to optimize electric vehicle (EV) charging across a distribution network, balancing energy costs, driver satisfaction, and grid stability.

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
cd pfe_imp
python -m venv venv
source venv/bin/activate
pip install -r requirement.txt
pip install questionary flask
```

### Run a Simulation

```bash
# Interactive mode selector:
python main.py

# Or directly:
python main.py 1    # PPO (continuous actions)
python main.py 2    # Q-Learning (discrete actions)
python main.py 3    # SAC (continuous, off-policy)
python main.py 4    # Federated training (interactive policy + aggregation choice)
python main.py 5    # Full comparison pipeline (all 9 combos)
```

### View Results Gallery

```bash
python app.py
# Open http://127.0.0.1:5000
```

---

## Project Structure

```
pfe_imp/
├── configs/               # YAML configuration files (env, sac, reward, training)
├── data/                  # Real market price datasets (CSV)
│   └── iso_ne_prices.csv  # 30-day hourly ISO-NE style prices
├── env/                   # Environments (GridEnv, EVClientEnv)
├── agents/                # RL Agents (Base, PPO, QLearning, SAC)
├── training/              # Orchestrators (Pipeline, Servers, Edge, Runners)
├── utils/                 # Helpers (DataLoader, DriverBehaviorModel, Metrics, config_loader, rewards, MarketPriceLoader)
├── main.py                # Entry point — Interactive selector / CLI runner
├── app.py                 # Flask web gallery for result visualization
├── requirement.txt        # Python dependencies
├── templates/
│   └── gallery.html       # HTML template for results gallery
├── results/               # Saved plots (PNG) and simulation registry (JSON)
├── SPEC.md                # Formal mathematical specification
├── ARCHITECTURE.md        # System architecture & component diagrams
├── REWARD_DESIGN.md       # Detailed breakdown of the structured reward
├── CONFIG_GUIDE.md        # Guide to configuring parameters via YAML files
└── CHANGELOG.md           # Version updates and large refactors
```

---

## Architecture

The system models a **3-tier hierarchy**:

| Tier | Role | Implementation |
|------|------|---------------|
| **Cloud** | Global orchestration, FL aggregation | `main.py`, `FederatedServer.py` |
| **Edge** | Grid physics, intermediate aggregation | `GridEnv.py`, `EdgeAggregator.py` |
| **Vehicle** | Local RL policy learning | `EVClientEnv.py` + Agent (`PPO` / `Q-Learning` / `SAC`) |

### Agents

| Agent | Action Space | Algorithm | Best For |
|-------|-------------|-----------|----------|
| **PPO** | Continuous [-1, 1] → scaled to kW | Proximal Policy Optimization (Actor-Critic) | Fine-grained power control |
| **Q-Learning** | Discrete {Discharge, Idle, Charge} | Tabular Q-Learning (ε-greedy) | Baseline comparison |
| **SAC** | Continuous [-1, 1] → scaled to kW | Soft Actor-Critic (twin Q, auto-entropy) | Sample-efficient off-policy learning |

All agents implement the `BaseAgent` interface with `get_parameters()` / `set_parameters()` for federated aggregation.

### Federated Learning

| Strategy | Description | Implementation |
|----------|------------|---------------|
| **FedAvg** | Weighted average of edge params | `FederatedServer(strategy='fedavg')` |
| **FedOpt** | Server-side momentum | `FederatedServer(strategy='fedopt')` |
| **Edge (FHDP)** | Vehicle → Edge intermediate aggregation | `EdgeAggregator` |

---

## Comparison Pipeline

Mode 5 runs all 9 combinations and produces a comparative figure:

| Policy | Standalone | FedAvg | FedOpt |
|--------|-----------|--------|--------|
| Q-Learning | ✅ | ✅ | ✅ |
| PPO | ✅ | ✅ | ✅ |
| SAC | ✅ | ✅ | ✅ |

Output: `results/comparison_*.png` — 6-panel dashboard with reward convergence, cost, satisfaction, grid stability, test distributions, and a summary table.

---

## Evaluation Metrics

Each simulation run produces a **6-panel dashboard** saved to `results/`:

1. **Convergence** — Episode reward over training
2. **Energy Cost** — Cost minimization trend
3. **Client Satisfaction** — `SOC_final / SOC_required` ratio
4. **Grid Stability** — Power ramp standard deviation (σ_g)
5. **Generalization** — Train vs Test reward boxplot

Run configurations are logged to `results/simulation_registry.json` for cross-run comparison.

---

## Configuration System

The project is driven entirely by YAML configuration files in the `configs/` directory. **There are no hardcoded magic numbers**.

- `env.yaml`: EV battery capacity, physics, boundary constraints.
- `reward.yaml`: Weights for tracking, progress, cost, grid, and terminal penalties.
- `sac.yaml`: SAC specific hyper parameters (gamma, tau, lr, buffer, entropy).
- `training.yaml`: Federation loops, episodes, agents, core runner params, **and real-price data toggle**.

### Real vs Synthetic Prices

By default the pipeline uses a synthetic ISO-NE style price generator.  
To switch to **real market data**, set `use_real_prices: true` in `training.yaml` and point `real_prices_csv` to your CSV file (default: `data/iso_ne_prices.csv`).  
The CSV must contain columns `timestamp` and `price`. Prices are automatically min-max normalized to [0, 1].

Refer to the [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for deeper details on tuning the system.
---

## Documentation

| Document | Contents |
|----------|----------|
| [SPEC.md](SPEC.md) | Mathematical formulation, MDP definition, reward function, FL equations. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System diagrams, class hierarchy, data flow, gap analysis. |
| [REWARD_DESIGN.md](REWARD_DESIGN.md) | Reward function specification, component breakdown. |
| [CONFIG_GUIDE.md](CONFIG_GUIDE.md) | Guide to the four YAML configuration files `env, reward, sac, training`. |
| [CHANGELOG.md](CHANGELOG.md) | Release notes for structure refactors and feature drops. |

---

## Reward Design (v2)

The EV charging reward uses a **5-component structured function** designed to make SOC ≥ 0.9 the clear primary objective:

| Component | Formula | Weight | Purpose |
|-----------|---------|--------|---------|
| Distance penalty | `−w·max(0, soc_req−soc)` | `2.0` | Continuous pull toward target at every step |
| SOC progress | `+w·Δsoc` (if Δsoc>0 & soc<soc_req) | `5.0` | Positive reinforcement for charging |
| Cost penalty | `−w·max(0,energy)·price` | `0.5` | Moderate economic disincentive |
| Grid penalty | `−w·|energy|·grid_signal` | `0.3` | Mild congestion constraint |
| Terminal bonus | `+15.0` if SOC ≥ soc_req | — | Strong goal achievement signal |
| Terminal penalty | `−25·(soc_req−soc)` if SOC < soc_req | — | Proportional miss penalty |

**Expected reward range:** −1.5 to +0.8 per step; +10 to +20 per successful episode.

### SAC Training Stability

The reward redesign is paired with updated SAC hyperparameters:

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `hidden_dim` | 64 | **128** | More capacity for richer reward landscape |
| `buffer_capacity` | 50k | **100k** | Larger diverse replay before updates |
| `warmup_steps` | 500 | **2,000** | Diverse buffer before first gradient step |
| `target_entropy` | −1.0 | **−0.5** | More exploitation, less exploration for goal task |

### Expected Learning Curve

```
Episode reward
 +15 |                                         ╭──────────
 +10 |                                   ╭────╯
  +5 |                             ╭────╯
   0 |               ╭────────────╯
  −5 |       ╭───────╯
 −15 |───────╯
     └──────────────────────────────────────────────────▶
       ep 0    ep 100   ep 200   ep 300   ep 500+
```

See [REWARD_DESIGN.md](REWARD_DESIGN.md) for full mathematical formulation.

---

## Roadmap

- [x] **SAC Agent** — Soft Actor-Critic with twin Q-networks and auto entropy
- [x] **Federated Aggregation** — FedAvg and FedOpt via `FederatedServer`
- [x] **FHDP Pipeline** — Edge-level intermediate aggregation via `EdgeAggregator`
- [x] **Comparison Pipeline** — All 9 combos with comparative plots
- [x] **Real Price Data** — `MarketPriceLoader` loads real ISO-NE style CSV prices (toggle via `use_real_prices`)
- [ ] **LoRA Integration** — Freeze base weights, train only low-rank A/B matrices
- [x] **SWIFT Scheduling** — Client selection under time-of-stay constraints (modes 9/10)
- [ ] **DQN Agent** — Deep Q-Network as alternative to tabular Q-Learning
- [ ] **Real Driver Profiles** — Replace synthetic NHTS profile generator with real NHTS survey data

---

## License

Academic research project — PFE (Projet de Fin d'Études).
