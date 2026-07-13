# HFDRL — Hierarchical Federated Deep Reinforcement Learning for EV Charging

A simulation framework implementing **Hierarchical Federated Deep Reinforcement Learning (HFDRL)** to optimize electric vehicle (EV) charging across an IEEE 33-bus distribution network. HFDRL combines Soft Actor-Critic (SAC) with federated learning (HFedAvg), SWIFT client selection, and LoRA model compression to simultaneously minimize charging cost, preserve grid stability, and maximize driver satisfaction — while cutting communication overhead.

---

## Headline Results

| Metric | FedAvg-SAC | SWIFT-SAC | **HFDRL** |
|--------|-----------|-----------|-----------|
| Mean test reward | — | — | **—** |
| Charging cost ($/ep) | — | — | **—** |
| Voltage violations (%) | — | — | **—** |
| Comm. overhead (MB/round) | — | — | **—** |
| Final SOC ≥ 0.9 (%) | — | — | **—** |

> Numbers will be populated from the multi-seed statistical evaluation (`python main.py` → option 16). See [`docs/reproducing_results.md`](docs/reproducing_results.md) (P1).

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirement.txt

# 3. Launch the interactive menu (Training mode)
python main.py

# 4. Run a specific simulation directly
python main.py 1 --simulation 16      # Multi-seed statistical evaluation (AAAI)

# 5. Browse results
python app.py   # → open http://127.0.0.1:5000
```

See [`docs/installation.md`](docs/installation.md) for full setup and GPU notes, and [`docs/usage.md`](docs/usage.md) for all 19 simulation modes.

---

## Repository Structure

```
pfe_imp/
├── main.py                    # Entry point — interactive selector + CLI
├── app.py                     # Flask gallery for result visualisation
├── requirement.txt            # Python dependencies
│
├── configs/                   # YAML configuration files (no hardcoded numbers)
│   ├── training.yaml          # Episodes, agents, FL rounds, baselines registry
│   ├── training_dev.yaml      # Fast dev mode (20 ep, 5 agents)
│   ├── env.yaml               # Battery physics, SOC bounds, CC-CV profile
│   ├── sac.yaml               # SAC hyperparameters (γ, τ, lr, buffer, entropy)
│   ├── reward.yaml            # Reward function weights
│   ├── lora.yaml              # LoRA rank, alpha, target modules
│   ├── swift.yaml             # SWIFT eligibility and utility weights
│   ├── multiseed.yaml         # Seeds, CI z-score, plot formatting
│   ├── dwell_time_study.yaml  # Dwell-time study parameters
│   ├── lora_network_study.yaml# LoRA network constraints study
│   └── stress_test_study.yaml # Forecast-error & non-IID robustness study
│
├── env/                       # RL environments
│   ├── GridEnv.py             # IEEE 33-bus power-flow simulation (pandapower)
│   └── EVClientEnv.py         # Per-EV battery MDP (state, reward, SOC dynamics)
│
├── agents/                    # RL agents (all FL-ready via BaseAgent interface)
│   ├── BaseAgent.py           # Abstract base: get_action / update / get_parameters
│   ├── SACAgent.py            # Soft Actor-Critic with twin-Q and auto entropy
│   ├── PPOAgent.py            # Proximal Policy Optimisation (Actor-Critic)
│   ├── QLearningAgent.py      # Tabular Q-Learning with ε-greedy
│   └── HeuristicAgents.py     # Random, Greedy, EDF, Price-Aware, Simple MPC
│
├── training/                  # Orchestration and study runners
│   ├── SimulationRunner.py    # Reusable agent-agnostic simulation loop
│   ├── ComparisonPipeline.py  # Unified multi-method comparison
│   ├── FederatedServer.py     # Global FL aggregation (FedAvg/FedProx/FedAvgM/FedAdam)
│   ├── EdgeAggregator.py      # Intermediate edge aggregation (FHDP)
│   ├── SWIFTScheduler.py      # SWIFT client selection (eligibility + utility)
│   ├── MultiSeedRunner.py     # 10-seed statistical pipeline (AAAI)
│   ├── DwellTimeStudy.py      # FedAvg vs SWIFT vs HFDRL × dwell windows
│   ├── LoRANetworkStudy.py    # HFDRL vs HFDRL+LoRA × BW scenarios
│   ├── StressTestStudy.py     # Forecast error & non-IID robustness study
│   └── BaseStudy.py           # Shared base class for all study runners
│
├── utils/                     # Shared utilities
│   ├── config_loader.py       # Singleton YAML loader
│   ├── reward_functions.py    # Pure reward computation function
│   ├── EvalMetrics.py         # Metrics logging, plots, JSON registry
│   ├── StatisticalAnalysis.py # CI95, t-test, Wilcoxon, LaTeX tables
│   ├── DataLoader.py          # Synthetic price & driver profile generation
│   ├── MarketPriceLoader.py   # Real ISO-NE CSV price loader
│   ├── DriverBehaviorModel.py # 3-archetype driver behaviour model
│   ├── lora.py                # LoRA layer, apply_lora(), FL-scoped state dicts
│   ├── device_utils.py        # CUDA / MPS discovery
│   └── constants.py           # Shared constants
│
├── network_sim/               # Non-intrusive communication overhead simulator
│   ├── network_simulator.py   # Node hierarchy, transfer cost model
│   ├── wrappers.py            # Instrumented wrappers (agent / edge / server)
│   ├── simulation_runner.py   # Cloud-only vs hierarchical comparison
│   └── config.py              # Default BW / latency parameters
│
├── data/                      # Market price datasets
│   ├── iso_ne_prices_real.csv # 30-day real ISO-NE hourly prices
│   └── iso_ne_prices_dev_test.csv # Shorter dataset for dev mode
│
├── results/                   # Runtime outputs (git-ignored)
│   ├── multi_seed/            # Multi-seed statistical evaluation outputs
│   ├── dwell_time_study/      # Dwell-time study outputs
│   ├── lora_network_study/    # LoRA network study outputs
│   ├── stress_test_study/     # Stress-test study outputs
│   └── trained_models/        # Saved model checkpoints
│
├── docs/                      # Extended documentation (this project)
│   ├── installation.md        # Full setup guide
│   ├── architecture.md        # Module map + system diagrams
│   ├── configuration.md       # Complete hyperparameter reference
│   └── usage.md               # All 19 modes, commands, output layout
│
└── paper/                     # LaTeX manuscript
```

---

## Architecture Overview

HFDRL follows a 3-tier Cloud → Edge → Vehicle hierarchy:

```
┌─────────────────────────────────────────────────────────────────┐
│  CLOUD  FederatedServer  ←  global FL aggregation (HFedAvg)     │
│         SWIFTScheduler   ←  client selection (utility-based)    │
├─────────────────────────────────────────────────────────────────┤
│  EDGE   EdgeAggregator   ←  intermediate FHDP aggregation       │
│         GridEnv          ←  IEEE 33-bus power flow (pandapower)  │
├─────────────────────────────────────────────────────────────────┤
│  VEHICLE  EVClientEnv × N  ←  battery MDP, reward               │
│           SACAgent × N     ←  SAC + optional LoRA adapters      │
└─────────────────────────────────────────────────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the full system and data-flow diagrams.

---

## Simulation Modes

`python main.py` launches an interactive menu. The same modes are reachable via `--simulation <N>`:

| # | Mode | Description |
|---|------|-------------|
| 1 | PPO Training | Standalone PPO (continuous action) |
| 2 | Q-Learning Training | Standalone tabular Q-Learning |
| 3 | SAC Training | Standalone SAC |
| 4 | Federated Training | SAC/PPO/Q-Learning + FedAvg or FedOpt |
| 5 | Full Comparison | All baseline combinations |
| 6 | SAC + LoRA | SAC with LoRA compression |
| 7 | PPO + LoRA | PPO with LoRA compression |
| 8 | Federated + LoRA | Federated training with LoRA |
| 9 | Federated + SWIFT | Federated with SWIFT client selection |
| 10 | Federated + SWIFT + LoRA | Full HFDRL stack |
| 11 | Baseline Suite | All baselines from `training.yaml` |
| 12 | Heuristic Baseline | Random / Greedy / EDF / Price-Aware / MPC |
| 13 | Federated Variant | FedProx / FedAvgM / FedAdam |
| 14 | SAC Local-Only | No federation |
| 15 | SAC Centralized Oracle | Shared network (upper bound) |
| 16 | **Multi-Seed Eval (AAAI)** | 10-seed CI95 + significance tests |
| 17 | **Dwell-Time Study (AAAI)** | FedAvg vs SWIFT vs HFDRL × dwell windows |
| 18 | **LoRA Network Study (AAAI)** | HFDRL vs HFDRL+LoRA × BW scenarios |
| 19 | **Robustness Stress Tests (AAAI)** | Forecast error + Non-IID data |

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/installation.md`](docs/installation.md) | Full setup, dependencies, GPU, smoke test |
| [`docs/architecture.md`](docs/architecture.md) | Module map, system diagram, data flow |
| [`docs/configuration.md`](docs/configuration.md) | Every hyperparameter and where it lives |
| [`docs/usage.md`](docs/usage.md) | Commands for all 19 modes, output layout |
| [`SPEC.md`](SPEC.md) | Mathematical MDP formulation |
| [`REWARD_DESIGN.md`](REWARD_DESIGN.md) | Reward function breakdown |
| [`CONFIG_GUIDE.md`](CONFIG_GUIDE.md) | YAML config quick reference |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Detailed class hierarchy and flow diagrams |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

---

## Citation

```bibtex
@misc{limam2026hfdrl,
  title  = {Hierarchical Federated Deep Reinforcement Learning for EV Charging Optimisation},
  author = {Limam, Mohamed},
  year   = {2026},
  note   = {PFE — Projet de Fin d'Études}
}
```

---

## License

Academic research project — PFE (Projet de Fin d'Études).
