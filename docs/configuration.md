# Configuration Reference

All hyperparameters live in YAML files under `configs/`. There are no hardcoded numbers in the training or agent code — every tunable value is loaded via `utils/config_loader.get_config('<name>')`.

To change a value: edit the relevant YAML file, no code change required.

---

## File Overview

| File | Loaded by | Controls |
|------|-----------|---------|
| `configs/training.yaml` | `main.py`, `ComparisonPipeline` | Episodes, agents, FL rounds, baselines registry |
| `configs/training_dev.yaml` | Dev mode (mode=2) | Same keys, smaller values for fast iteration |
| `configs/env.yaml` | `EVClientEnv`, `GridEnv` | Battery physics, SOC bounds, CC-CV profile |
| `configs/sac.yaml` | `SACAgent` | SAC hyperparameters |
| `configs/reward.yaml` | `reward_functions.py` | Reward component weights |
| `configs/lora.yaml` | `SACAgent`, `PPOAgent` | LoRA rank, alpha, target modules |
| `configs/swift.yaml` | `SWIFTScheduler` | Client selection policy |
| `configs/multiseed.yaml` | `MultiSeedRunner` | Seeds, CI z-score, plot style |
| `configs/dwell_time_study.yaml` | `DwellTimeStudy` | Dwell-time study parameters |
| `configs/lora_network_study.yaml` | `LoRANetworkStudy` | BW scenarios, RL + analytical phases |
| `configs/stress_test_study.yaml` | `StressTestStudy` | Noise levels, Dirichlet α values |

---

## `configs/training.yaml` — Training & Federation

| Key | Default | Unit | Description |
|-----|---------|------|-------------|
| `num_episodes` | 1000 | episodes | Training episodes per method |
| `num_test_episodes` | 200 | episodes | Test episodes per method |
| `simulation_hours` | 24 | hours | Episode length (= 24 decision steps) |
| `num_agents` | 20 | agents | Number of EVs in the simulation |
| `grid_type` | `"case33bw"` | — | pandapower network topology |
| `update_every` | 10 | steps | Agent weight update frequency |
| `num_edges` | 2 | — | Number of edge aggregators (FHDP) |
| `fl_rounds_per_episode` | 1 | — | FL aggregation rounds per episode |
| `fl_rounds` | 6 | — | Total FL rounds over training |
| `fl_fraction` | 1.0 | — | Fraction of clients per FL round (overridden by SWIFT) |
| `use_real_prices` | `true` | — | `true` = load CSV; `false` = synthetic generator |
| `real_prices_csv` | `"data/iso_ne_prices_real.csv"` | path | ISO-NE price CSV (must have `timestamp`, `price` columns) |
| `progress.enabled` | `true` | — | Show tqdm progress bars |
| `progress.level` | `"episode"` | — | Progress granularity |
| `methods` | see file | list | Baseline methods registry for `run_methods_from_config()` |

**Dev mode** (`configs/training_dev.yaml`) uses `num_episodes=20`, `num_test_episodes=1`, `num_agents=5`, `fl_rounds=1`. Switch with mode=2.

---

## `configs/env.yaml` — Environment & Battery Physics

| Key | Default | Unit | Paper symbol | Description |
|-----|---------|------|-------------|-------------|
| `battery_capacity` | 60.0 | kWh | Cᵢ | EV battery capacity |
| `eta` | 0.95 | — | η | Charging efficiency |
| `dt` | 1.0 | hours | Δt | Timestep duration |
| `max_power` | 11.0 | kW | ū | Maximum charging power (Level 2 AC) |
| `soc_min` | 0.0 | — | SOC_min | Hard lower SOC bound |
| `soc_max` | 1.0 | — | SOC_max | Hard upper SOC bound |
| `soc_target` | 0.9 | — | SOC_req | Target SOC at departure |
| `alpha_constraint` | 0.05 | — | α | CC-CV taper coefficient: P_max = ū · (1 − α · SOC) |
| `driver_behavior.enabled` | `false` | — | — | Enable location-aware driver schedule |
| `driver_behavior.types` | `[commuter, flexible, night_charger]` | — | — | Driver archetypes to sample |
| `paper_plots.enabled` | `true` | — | — | Auto-generate research-quality figures |

**State of Charge update (per timestep):**

```
SOC_{t+1} = SOC_t + (P_kw · η · Δt) / C
```

**CC-CV power limit:**

```
P_max(SOC) = ū · (1 − α · SOC)
```

---

## `configs/sac.yaml` — Soft Actor-Critic

| Key | Default | Description |
|-----|---------|-------------|
| `gamma` | 0.99 | Discount factor γ |
| `tau` | 0.005 | Target network soft update coefficient τ |
| `lr` | 3e-4 | Learning rate for actor, critic, and log-α |
| `batch_size` | 256 | Mini-batch size for gradient updates |
| `buffer_capacity` | 100 000 | Replay buffer capacity (diverse off-policy experience) |
| `warmup_steps` | 2 000 | Random steps before first gradient update |
| `update_every` | 1 | Update networks every N environment steps |
| `hidden_dim` | 128 | Hidden layer width for GaussianPolicy and TwinQNetwork |
| `alpha_init` | 0.2 | Initial entropy temperature α |
| `target_entropy_scale` | −0.5 | Target entropy = `action_dim × target_entropy_scale` |
| `federated.aggregation` | `"fedavg"` | Default FL strategy (overridden per-method by `training.yaml`) |
| `federated.mu_fedprox` | 0.01 | FedProx proximal coefficient μ |
| `federated.beta_momentum` | 0.9 | FedAvgM server-side momentum β |
| `federated.adam_lr` | 0.01 | FedAdam server-side learning rate |

**Network architecture:**

```
GaussianPolicy: Linear(13,128) → ReLU → Linear(128,128) → ReLU
                → μ-head Linear(128,1), log_std-head Linear(128,1)
TwinQNetwork:   [Linear(14,128) → ReLU → Linear(128,128) → ReLU → Linear(128,1)] × 2
```

---

## `configs/reward.yaml` — Reward Weights

The reward has 5 components. See `REWARD_DESIGN.md` for the full mathematical formulation.

| Key | Default | Paper symbol | Component |
|-----|---------|-------------|-----------|
| `w_target_tracking` | 2.0 | w_A | Continuous distance penalty: −w · max(0, SOC_req − SOC) |
| `w_progress` | 5.0 | w_B | SOC progress bonus: +w · ΔSOC (when SOC < SOC_req and ΔSOC > 0) |
| `w_cost` | 0.5 | w_C | Energy cost penalty: −w · max(0, energy_kWh) · price |
| `w_grid` | 0.3 | w_D | Grid congestion penalty: −w · |energy_kWh| · λ_grid |
| `terminal_success_bonus` | 15.0 | — | Fixed bonus if SOC ≥ SOC_req at t_dep |
| `terminal_failure_weight` | 25.0 | — | Proportional penalty: −w · (SOC_req − SOC) if SOC < SOC_req at t_dep |

**Expected range:** −1.5 to +0.8 per step; +10 to +20 per successful episode.

---

## `configs/lora.yaml` — LoRA Compression

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Master switch. Also overridable with env var `USE_LORA=true`. |
| `rank` | 4 | Low-rank dimension r. Typical range: 2–16. |
| `alpha` | 8 | Scaling factor. Effective scale = alpha / rank. |
| `target_modules` | see file | List of `nn.Linear` name substrings to wrap with LoRA. |

**Parameter count reduction:** For `hidden_dim=128` and `rank=4`, each wrapped layer contributes 2 × (128 × 4) = 1 024 parameters instead of 128 × 128 = 16 384 — a 94 % reduction per layer.

**Effective weight:** W_eff = W₀ + B · A · (alpha / rank), where W₀ is frozen, A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}.

---

## `configs/swift.yaml` — SWIFT Client Selection

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable SWIFT scheduling (overrides `fl_fraction`). |
| `min_stay_hours` | 2 | Minimum remaining parking time (hours) for eligibility. |
| `fraction` | 0.6 | Fraction of eligible clients selected per round. |
| `utility_weights.staleness` | 0.4 | Weight for staleness utility Uₛ (reward unseen agents). |
| `utility_weights.soc_gap` | 0.4 | Weight for SOC-gap utility U_g (reward agents far from target). |
| `utility_weights.diversity` | 0.2 | Weight for diversity utility U_d (reward minority driver types). |
| `staleness_cap` | 15 | Normalisation ceiling for staleness score. |
| `force_select_after` | 20 | Force-select any agent unseen for this many consecutive rounds. |

**Selection rule:**

```
U(i) = w_s · U_s(i) + w_g · U_g(i) + w_d · U_d(i)
eligible = {i : t_dep(i) − t_current ≥ min_stay_hours}
selected = top-k(eligible, key=U, k=ceil(fraction · |eligible|))
```

---

## `configs/multiseed.yaml` — Statistical Evaluation

| Key | Default | Description |
|-----|---------|-------------|
| `seeds` | `[0,1,2,3,4,42,123,456,789,999]` | Seeds for multi-seed runs |
| `ci_z_score` | 1.96 | z-score for 95 % confidence intervals |
| `significance_alpha` | 0.05 | p-value threshold for hypothesis tests |
| `convergence.window` | 50 | Rolling window to detect convergence |
| `convergence.threshold` | 0.01 | Relative improvement threshold for convergence |

---

## `configs/stress_test_study.yaml` — Robustness Study

| Key | Default | Description |
|-----|---------|-------------|
| `forecast_error.noise_levels` | `[0.0, 0.02, 0.05, 0.10, 0.20]` | Gaussian σ added to price forecast ($/kWh) |
| `non_iid.alpha_values` | `[1000.0, 10.0, 1.0, 0.5, 0.1]` | Dirichlet α for data heterogeneity |
| `methods` | `[FedAvg-SAC, SWIFT-SAC, HFDRL]` | Methods to evaluate |
| `seeds` | `[0,1,2,42,123]` | Default seeds (5); overridable interactively |

---

## Network Parameters (not YAML — hardcoded in `network_sim/config.py`)

These default values were used in the LoRA network study. Verify against the actual simulation config before publishing in a paper.

| Node type | Bandwidth (Mbps) | Latency (ms) | Rationale |
|-----------|-----------------|--------------|-----------|
| Agent (vehicle) | 10 | 20 | Typical LTE uplink |
| Edge server | 100 | 5 | Low-latency MEC |
| Cloud server | 1 000 | 50 | Cross-region backbone |

Transfer time: `t = (size_MB × 8 / bw_Mbps) + (latency_ms / 1000)`
