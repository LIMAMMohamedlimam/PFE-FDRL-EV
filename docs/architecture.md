# Repository Architecture

## 1. System Overview

HFDRL follows a strict **Cloud → Edge → Vehicle** hierarchy. No layer communicates with a non-adjacent layer directly.

```
┌──────────────────────────────────────────────────────────────────────┐
│  CLOUD                                                               │
│  FederatedServer ── HFedAvg / FedProx / FedAvgM / FedAdam           │
│  SWIFTScheduler  ── eligibility filter + utility-based top-k        │
│  ComparisonPipeline ── orchestrates multi-method training            │
├──────────────────────────────────────────────────────────────────────┤
│  EDGE  (× num_edges, default 2)                                      │
│  EdgeAggregator ── FHDP intermediate aggregation                    │
│  GridEnv        ── IEEE 33-bus power flow (pandapower / Newton-Raphson)│
├──────────────────────────────────────────────────────────────────────┤
│  VEHICLE  (× num_agents, default 20)                                 │
│  EVClientEnv ── battery MDP: SOC dynamics, CC-CV profile, reward     │
│  SACAgent    ── SAC with twin-Q, auto-entropy, optional LoRA adapters│
└──────────────────────────────────────────────────────────────────────┘
```

The **LoRA** module (`utils/lora.py`) freezes the base weights W₀ of each SAC network and trains only the low-rank adapters A, B. Only these small matrices are transmitted during FL rounds, reducing communication overhead.

---

## 2. Module Map

### Entry points

| File | Role |
|------|------|
| `main.py` | Interactive CLI menu (19 modes) + argument dispatch |
| `app.py` | Flask gallery — serves result images from `results/` |

### `env/`

| File | Responsibility |
|------|---------------|
| `GridEnv.py` | IEEE 33-bus AC power-flow simulation via pandapower. Accepts per-bus EV load injections, runs Newton-Raphson, returns grid congestion signal λ and voltage metrics. |
| `EVClientEnv.py` | Per-EV battery MDP. Tracks SOC, enforces CC-CV power limit, builds 13-dim state vector, computes structured reward. |

### `agents/`

| File | Responsibility |
|------|---------------|
| `BaseAgent.py` | Abstract interface: `get_action`, `update`, `get_parameters`, `set_parameters` — every FL-ready agent implements this. |
| `SACAgent.py` | Soft Actor-Critic with GaussianPolicy actor, TwinQNetwork critic, auto-entropy tuning. Supports LoRA via `use_lora=True`. |
| `PPOAgent.py` | Proximal Policy Optimisation with shared ActorCritic network. Continuous action ∈ [−1, 1]. |
| `QLearningAgent.py` | Tabular Q-Learning, discrete actions {idle, half-charge, full-charge}. |
| `HeuristicAgents.py` | Rule-based baselines: Random, Greedy (charge when SOC < req), EDF (earliest deadline first), Price-Aware (threshold), Simple MPC (1-step lookahead). |

### `training/`

| File | Responsibility |
|------|---------------|
| `SimulationRunner.py` | Reusable agent-agnostic training + test loop. Called by ComparisonPipeline. |
| `ComparisonPipeline.py` | Runs one or many methods from config; logs FL round timing and voltage violations. |
| `FederatedServer.py` | Global FL aggregation. Strategies: `fedavg`, `fedopt`, `fedprox`, `fedavgm`, `fedadam`. |
| `EdgeAggregator.py` | Intermediate FHDP aggregation: vehicle → edge → cloud. |
| `SWIFTScheduler.py` | Client selection: eligibility filter (minimum dwell) + utility scoring (staleness, SOC gap, diversity). |
| `BaseStudy.py` | Shared base for all AAAI study runners: seeding, output directory setup, progress logging. |
| `MultiSeedRunner.py` | 10-seed statistical pipeline: seeds all RNGs, runs all methods, collects per-seed metrics. |
| `DwellTimeStudy.py` | FedAvg-SAC vs SWIFT-SAC vs HFDRL across dwell-time scenarios [1h, 2h, 4h, 6h, 8h]. |
| `LoRANetworkStudy.py` | Two-phase study: RL training phase then analytical bandwidth sweep (5 scenarios). |
| `StressTestStudy.py` | Forecast-error sub-study (Gaussian noise on price forecast) + Non-IID sub-study (Dirichlet α). |

### `utils/`

| File | Responsibility |
|------|---------------|
| `config_loader.py` | Singleton loader: `get_config('sac')` returns parsed dict from `configs/sac.yaml`. |
| `reward_functions.py` | Pure function computing the 5-component EV reward (no side effects). |
| `EvalMetrics.py` | Metrics logging (reward, cost, SOC, voltage violations, comm overhead), 6-panel plots, JSON registry. |
| `StatisticalAnalysis.py` | Mean/std/CI95, paired t-test + Wilcoxon vs HFDRL, LaTeX + CSV tables. |
| `DataLoader.py` | Synthetic ISO-NE price generator + NHTS driver profile sampler. |
| `MarketPriceLoader.py` | Loads real ISO-NE CSV, min-max normalises to [0,1], train/test split. |
| `DriverBehaviorModel.py` | 3-archetype behaviour model (commuter, flexible, night_charger) with Gaussian parameters. |
| `lora.py` | `LoRALinear` layer, `apply_lora()`, FL-scoped state dict helpers (LoRA-only get/set). |
| `device_utils.py` | Auto-discovers CUDA, Apple MPS, or CPU. |
| `constants.py` | Shared numeric constants. |

### `network_sim/`

Non-intrusive communication overhead simulator — wraps real agents/aggregators without modifying them.

| File | Responsibility |
|------|---------------|
| `network_simulator.py` | `NetworkSimulator` log + `AgentNode`/`EdgeNode`/`CloudNode` with bandwidth/latency parameters. |
| `wrappers.py` | `InstrumentedAgent`, `InstrumentedEdge`, `InstrumentedServer` — transparent adapters that intercept `get_parameters` / `set_parameters` / `aggregate` to measure byte sizes and transfer time. |
| `simulation_runner.py` | Runs Cloud-Only vs Hierarchical simulation modes and produces comparison metrics. |
| `config.py` | Default network parameters (bandwidth, latency per node type). |

---

## 3. System Architecture Diagram

```
                          ┌─────────────────────────┐
                          │      main.py (CLI)       │
                          │   mode 1–19 dispatcher   │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   ComparisonPipeline     │
                          │   (or Study runners)     │
                          └──┬──────────────────┬───┘
                             │                  │
              ┌──────────────▼──┐          ┌────▼──────────────────┐
              │ FederatedServer  │          │   SimulationRunner    │
              │ HFedAvg / FedOpt │◄────────►│  (training loop)      │
              │ FedProx / FedAdam│          └──────────┬────────────┘
              └──────┬──────────┘                      │
                     │  broadcast /                    │  per episode × 24 hours
                     │  aggregate                      │
              ┌──────▼──────────┐          ┌──────────▼────────────┐
              │  EdgeAggregator  │          │      GridEnv           │
              │  (FHDP, × edges) │          │  IEEE 33-bus / pandapp │
              └──────┬──────────┘          │  → λ_grid, voltages    │
                     │                     └──────────┬────────────┘
                     │  collect                        │  per agent
              ┌──────▼──────────────────────────────── ▼────────────┐
              │              EVClientEnv × N agents                  │
              │    SOC dynamics · CC-CV · 13-dim state · reward      │
              │                   SACAgent (+ LoRA)                  │
              └──────────────────────────────────────────────────────┘
```

---

## 4. Data-Flow Diagram (one timestep)

```
DataGenerator / MarketPriceLoader
  ├─ price(hour)            $/kWh, normalised to [0,1]
  └─ price_forecast[5]      5-hour ahead window

  For each active agent i:
  ┌─────────────────────────────────────────────────────┐
  │  EVClientEnv.get_state()                            │
  │    state[0]     = SOC                               │
  │    state[1]     = sin(2π·t/24)  time encoding      │
  │    state[2]     = cos(2π·t/24)                     │
  │    state[3]     = (t_dep − t) / t_dep  (normalised)│
  │    state[4]     = λ_grid  (congestion, prev step)  │
  │    state[5]     = voltage_dev  (prev step)          │
  │    state[6]     = ev_total_mw  (prev step)          │
  │    state[7]     = delta_ev_mw  (prev step)          │
  │    state[8:13]  = price_forecast / 0.5              │
  │                                              dim=13  │
  │  SACAgent.get_action(state)                         │
  │    → raw_action ∈ [−1, 1]  (tanh-squashed Gaussian)│
  │    → P_kw = raw_action × P_max(SOC)                 │
  │    → P_mw = P_kw / 1000                             │
  └─────────────────────────────────────────────────────┘
          │  grid_injections_mw[bus_i] += P_mw
          ▼
  ┌─────────────────────────────────────────────────────┐
  │  GridEnv.step(grid_injections_mw, base_load_mw)     │
  │    AC power flow  (Newton-Raphson, IEEE 33-bus)      │
  │    → λ_grid  (line congestion signal)               │
  │    → grid_info {max_voltage, min_voltage, …}        │
  └─────────────────────────────────────────────────────┘
          │
          ▼
  For each active agent i:
  ┌─────────────────────────────────────────────────────┐
  │  EVClientEnv.step(P_kw, λ_grid, voltage_dev, price) │
  │    SOC += (P_kw · η · dt) / C                       │
  │    reward = r_track + r_progress + r_cost + r_grid  │
  │             + terminal_bonus_or_penalty              │
  │    → (reward, done, soc, energy_cost)               │
  │                                                     │
  │  SACAgent.update(s_t, a_t, r_t, s_{t+1}, done)     │
  │    → store in replay buffer                         │
  │    → if buffer ≥ warmup_steps: gradient update      │
  └─────────────────────────────────────────────────────┘

  End of episode (FL enabled):
  ┌─────────────────────────────────────────────────────┐
  │  SWIFTScheduler.select(agents)                      │
  │    eligibility: t_dep − t_current ≥ min_stay_hours  │
  │    utility: w_staleness·U_s + w_soc·U_g + w_div·U_d│
  │    → selected_ids (top-k eligible)                  │
  │                                                     │
  │  EdgeAggregator.collect(selected) → .aggregate()   │
  │    → edge_params (weighted FedAvg per edge)         │
  │                                                     │
  │  FederatedServer.aggregate(edge_params)             │
  │    → global_params  (cloud-level aggregation)       │
  │                                                     │
  │  FederatedServer.broadcast(all_agents)              │
  │    → set_parameters(global_params) on each agent    │
  └─────────────────────────────────────────────────────┘
```

---

## 5. LoRA Compression in the FL Loop

Without LoRA, `get_parameters()` returns the full weight dict (~MB per agent per round). With LoRA:

```
SACAgent
  ├─ base weights W₀  (frozen, never transmitted)
  └─ LoRA adapters A, B  (trainable, transmitted)
       W_eff = W₀ + B · A · (alpha / rank)
```

Only the A, B matrices (rank × hidden_dim each) are aggregated. For `rank=4` and `hidden_dim=128`, this is ~6 % of the original parameter count per layer.

The `utils/lora.py` helpers `lora_state_dict()` and `load_lora_state_dict()` extract/inject only the LoRA parameters so the FL aggregation code is unchanged.

---

## 6. Output Directory Layout

```
results/
├── multi_seed/<timestamp>/
│   ├── reproducibility_notes.json   ← RNG control, versions, CI formula
│   ├── aggregated_raw.json          ← scalar metrics for all methods × seeds
│   ├── <Method_Name>/seed_N/
│   │   ├── reward_curve.npy
│   │   ├── cost_curve.npy
│   │   └── metrics.json
│   └── aggregated/
│       ├── tables/    ← performance_table.{tex,csv}, significance_table.{tex,csv}
│       ├── plots/     ← 8 figures in PDF + PNG
│       └── statistics/← full_stats.json, significance_report.json
│
├── dwell_time_study/<timestamp>/
│   └── <method>/<dwell_h>h/seed_N/ ← same per-seed structure
│
├── lora_network_study/<timestamp>/
│   └── <method>/seed_N/            ← RL metrics + analytical BW sweep results
│
├── stress_test_study/<timestamp>/
│   └── <sub_study>/<scenario>/seed_N/
│
└── trained_models/                 ← saved model checkpoints
```
