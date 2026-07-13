# API Reference

Public interfaces for every class and function that external code (scripts, notebooks, new baselines) may call. Internal helpers prefixed with `_` are omitted.

Style: Google docstring conventions. Shapes use the notation `(dim,)` for 1-D arrays.

---

## `env.EVClientEnv`

### `EVClientEnv(override_env_config=None)`

Per-EV battery MDP. Reads base config from `configs/env.yaml` and `configs/reward.yaml`; `override_env_config` merges episode-specific values (e.g. `soc_req`, `t_dep`).

**Args:**

| Param | Type | Description |
|-------|------|-------------|
| `override_env_config` | dict or None | Keys from `env.yaml` to override for this instance (e.g. `initial_soc`, `soc_req`, `t_dep`, `capacity`, `max_power`) |

---

### `EVClientEnv.get_state(grid_signal, voltage_dev, price_forecast, ev_total_mw=0.0, delta_ev_mw=0.0) → np.ndarray`

Build and return the normalised state vector.

**Args:**

| Param | Type | Unit | Description |
|-------|------|------|-------------|
| `grid_signal` | float | — | Grid congestion λ ∈ [0, 2] from `GridEnv.step()` previous timestep |
| `voltage_dev` | float | p.u. | `max_voltage − 1.0` from `grid_info` previous timestep |
| `price_forecast` | list[float] | $/kWh | 5 future prices (unnormalised) |
| `ev_total_mw` | float | MW | Aggregate EV load previous timestep (default 0) |
| `delta_ev_mw` | float | MW | Change in EV load previous timestep (default 0) |

**Returns:** `np.ndarray` of shape `(13,)` (or `(16,)` with driver behavior enabled), `dtype=float32`, all values in [−1, 1] or [0, 1].

---

### `EVClientEnv.step(action_power, grid_signal, voltage_dev, price_current) → (float, bool, float, float)`

Execute one MDP step.

**Args:**

| Param | Type | Unit | Description |
|-------|------|------|-------------|
| `action_power` | float | kW | Requested charging power; positive = charge, negative = discharge (V2G). Clipped to [−P_max(SOC), +P_max(SOC)]. |
| `grid_signal` | float | — | Current grid congestion λ |
| `voltage_dev` | float | p.u. | Current `max_voltage − 1.0` |
| `price_current` | float | $/kWh | Current electricity price (unnormalised) |

**Returns:** `(total_reward, done, soc, energy_cost)`

| Return | Type | Description |
|--------|------|-------------|
| `total_reward` | float | RL reward (sum of 5 components from `reward_functions.compute_reward`) |
| `done` | bool | True when `current_step >= t_dep` |
| `soc` | float | Updated SOC ∈ [0, 1] |
| `energy_cost` | float | $ cost for this step = max(0, energy_kWh) × price_current |

---

## `env.GridEnv`

### `GridEnv(network_type='case33bw')`

IEEE 33-bus power-flow simulation. Wraps pandapower's `case33bw()` network.

**Args:** `network_type` — only `'case33bw'` is fully supported; any other value falls back to `case14()`.

---

### `GridEnv.reset() → np.ndarray`

Reset all load injections to zero. Returns bus voltage array (p.u.).

---

### `GridEnv.step(ev_loads_dict, base_load_mw) → (float, dict)`

Apply EV and base loads; run AC power flow (Newton-Raphson).

**Args:**

| Param | Type | Unit | Description |
|-------|------|------|-------------|
| `ev_loads_dict` | dict[int, float] | MW | `{bus_index: power_mw}` — aggregate EV load per bus |
| `base_load_mw` | float | MW | Background (non-EV) load, distributed uniformly across all load buses |

**Returns:** `(lambda_grid, info)`

| Return | Type | Description |
|--------|------|-------------|
| `lambda_grid` | float | Congestion signal = `clip(max_voltage_deviation / VOLTAGE_BAND, 0, 2)`. 0 = no stress; 2 = convergence failure |
| `info` | dict | `{'converged': bool, 'voltage_violations': int, 'min_voltage': float, 'max_voltage': float}` (p.u.) |

---

## `agents.BaseAgent`

Abstract interface that every FL-compatible agent must implement.

### `get_action(state, eval_mode=False) → float or int`

Select an action given the current state.

**Args:** `state` — observation array of shape `(obs_dim,)`. `eval_mode` — if True, use a deterministic policy (no exploration).

**Returns:** Continuous action ∈ [−1, 1] (SAC/PPO) or discrete index ∈ {0, 1, 2} (Q-Learning).

---

### `update(state, action, reward, next_state, done=False)`

Store the transition and, if applicable, perform a gradient update.

---

### `get_parameters() → dict[str, np.ndarray]`

Return model parameters as a dict of CPU numpy arrays for FL aggregation. Keys are prefixed: `'actor.<name>'`, `'critic.<name>'`. When LoRA is active, returns only LoRA adapter weights.

---

### `set_parameters(parameters: dict[str, np.ndarray])`

Load aggregated parameters from the FL server into the local model.

---

## `agents.SACAgent`

### `SACAgent(input_dim, action_dim=1, alpha_init=0.2, use_lora=False)`

**Args:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `input_dim` | int | — | State dimension (13 base, 16 with driver behavior) |
| `action_dim` | int | 1 | Number of action dimensions (always 1 for a single EV) |
| `alpha_init` | float | 0.2 | Initial entropy temperature |
| `use_lora` | bool | False | Enable LoRA adapters (also controlled by `lora.yaml → enabled` and `USE_LORA` env var) |

**Key attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `actor` | `GaussianPolicy` | Stochastic policy network |
| `critic` | `TwinQNetwork` | Twin Q-value networks |
| `buffer` | `ReplayBuffer` | Experience replay (capacity from `sac.yaml`) |
| `alpha` | float | Current entropy temperature (auto-tuned) |
| `device` | `torch.device` | Auto-detected (CUDA → MPS → CPU) |

---

### `SACAgent.save_trained_model(directory, agent_id)`

Save checkpoint to `<directory>/sac_agent[_lora]_<agent_id>[_<timestamp>].pth`.

Full checkpoints include actor, critic, critic_target, log_alpha, alpha. LoRA checkpoints include only adapter weights.

---

### `SACAgent.load_trained_model(model_path)`

Load checkpoint auto-detecting full vs LoRA format from key names.

---

### `SACAgent.set_fedprox_global(global_params, mu=0.01)`

Store global model snapshot for FedProx proximal penalty. Call once after each FL aggregation round, before local training resumes.

---

## `training.FederatedServer`

### `FederatedServer(strategy='fedavg', server_lr=1.0, beta=0.9, adam_beta1=0.9, adam_beta2=0.99, adam_eps=1e-3)`

Cloud-level aggregation server. Strategy must be one of `{'fedavg', 'fedprox', 'fedavgm', 'fedadam', 'fedopt'}`.

---

### `FederatedServer.initialize(params: dict)`

Set the initial global model. Must be called before the first `aggregate()`.

---

### `FederatedServer.aggregate(edge_updates: list[dict]) → dict`

Aggregate edge-level updates and apply the server-side strategy.

**Args:** List of `{'params': dict[str, np.ndarray], 'n_samples': int}`.

**Returns:** Updated global parameter dict.

---

### `FederatedServer.broadcast() → dict`

Return a copy of the current global parameters for distribution to agents.

---

### `FederatedServer.collect_selected(agents, selected_indices) → list[dict]`

Collect parameters from a subset of agents (used when SWIFT is active instead of collecting from all agents).

---

## `training.EdgeAggregator`

### `EdgeAggregator(edge_id, vehicle_ids=None)`

Edge-level intermediate aggregator. Manages `vehicle_ids` assigned to this edge.

---

### `EdgeAggregator.collect(vehicle_id, params, n_samples)`

Buffer one vehicle's update. `params` is a `dict[str, np.ndarray]` from `agent.get_parameters()`. `n_samples` is the number of transitions trained on (used as aggregation weight).

---

### `EdgeAggregator.aggregate() → (dict or None, int)`

Weighted FedAvg across buffered updates. Clears the buffer after aggregation.

**Returns:** `(aggregated_params, total_samples)`. Returns `(None, 0)` if no updates collected.

---

### `EdgeAggregator.collect_selected(agents, agent_bus_map, selected_indices) → list[dict]`

Filter collection to only `selected_indices`, grouping by bus, and return per-edge updates for `FederatedServer.aggregate()`.

---

## `training.SWIFTScheduler`

### `SWIFTScheduler(n_agents, config)`

**Args:** `n_agents` — total number of EV agents. `config` — dict from `get_config('swift')`.

---

### `SWIFTScheduler.select_clients(agents, envs, driver_profiles, current_round) → list[int]`

Run the full 5-step SWIFT selection algorithm and return sorted selected agent indices.

**Args:**

| Param | Type | Description |
|-------|------|-------------|
| `agents` | list | All RL agents (used for indexing only) |
| `envs` | list[EVClientEnv] | SOC, t_dep, current_step are read from each |
| `driver_profiles` | list[dict] | Per-agent dicts with `'driver_type'` key |
| `current_round` | int | Current FL round (episode index) |

**Returns:** Sorted list of selected agent indices.

---

### `SWIFTScheduler.get_round_stats(selected_indices, driver_profiles) → dict`

Return logging stats for the most recent selection round. Keys: `n_selected`, `n_eligible`, `avg_soc_gap`, `type_counts`.

---

### `SWIFTScheduler.reset()`

Reset staleness counters. Call between full experiment reruns.

---

## `utils.EvalMetrics`

### `EvalMetrics(run_name='metrics_plot', config=None)`

Metrics logger for one simulation run.

---

### Logging methods

| Method | Arg | Unit | What it logs |
|--------|-----|------|-------------|
| `log_episode(reward, mode='train')` | float | — | Episode total reward (train or test) |
| `log_cost(cost)` | float | $ | Total energy cost for the episode |
| `log_step(total_grid_load)` | float | MW | Instantaneous grid power (for σ_g computation) |
| `log_satisfaction(agent_sats)` | list[float] | — | Per-agent SOC_final / SOC_req |
| `log_voltage_violation(violated)` | bool | — | 1 if any bus outside [V_min, V_max] |
| `log_comm_overhead(ms)` | float | ms | Communication time for one FL round |
| `log_final_soc(mean_soc)` | float | — | Mean SOC across agents at end of test episode |
| `log_swift_selection(ep, indices, stats)` | — | — | SWIFT selection log for one round |

---

### `EvalMetrics.compute_stability_metric() → float`

Return σ_g = standard deviation of per-step grid power changes (MW). Lower = more stable.

---

### `EvalMetrics.plot_metrics()`

Save the 6-panel dashboard PNG and update `results/simulation_registry.json`. Automatically calls `save_csv()`.

---

## `utils.reward_functions`

### `compute_reward(soc, prev_soc, soc_req, t_dep, current_step, energy_transfer, price_current, grid_signal, reward_config) → (float, bool)`

Pure function computing the 5-component EV reward. No side effects.

**Args:**

| Param | Type | Unit | Description |
|-------|------|------|-------------|
| `soc` | float | — | Current SOC ∈ [0, 1] after physics update |
| `prev_soc` | float | — | SOC before this step |
| `soc_req` | float | — | Required SOC at departure |
| `t_dep` | int | steps | Departure timestep |
| `current_step` | int | steps | Current timestep (after increment) |
| `energy_transfer` | float | kWh | `action_power × dt`; positive = charging |
| `price_current` | float | $/kWh | Unnormalised electricity price |
| `grid_signal` | float | — | λ_grid from `GridEnv.step()` |
| `reward_config` | dict | — | Loaded from `configs/reward.yaml` |

**Returns:** `(total_reward: float, done: bool)`

---

## `utils.lora`

### `apply_lora(model, rank=4, alpha=8.0, target_modules=None) → nn.Module`

Walk `model` and replace matching `nn.Linear` layers with `LoRALinear` wrappers in-place.

**Args:** `target_modules` — list of substrings to match against layer names. Empty list = wrap all `nn.Linear` layers.

**Returns:** Same model (modified in-place).

---

### `get_lora_state_dict(model, prefix='') → dict[str, np.ndarray]`

Extract only LoRA weights from model state dict. Returns CPU numpy arrays for FL compatibility.

---

### `load_lora_state_dict(model, state_dict, prefix='', device=None)`

Load LoRA weights into model. Non-LoRA keys in `state_dict` are silently ignored.

---

### `get_lora_parameters(model) → Iterator[nn.Parameter]`

Yield only LoRA parameters (those with `'lora_'` in their name and `requires_grad=True`). Pass to `optim.Adam()` when LoRA is active.

---

### `count_parameters(model, trainable_only=True) → int`

Count parameters in model. `trainable_only=True` counts only those with `requires_grad=True`.

---

## `utils.DataLoader.DataGenerator`

### `DataGenerator.get_iso_ne_price(hour, mode='train') → float`

Return electricity price in $/kWh for the given hour. Uses real CSV data when `use_real_prices: true` in `training.yaml`, otherwise synthetic ISO-NE profile.

**Args:** `hour` ∈ [0, 23]; `mode` ∈ `{'train', 'test'}`.

---

### `DataGenerator.get_nhts_profile(n_agents) → list[dict]`

Return `n_agents` driver profiles sampled from NHTS 2017 distributions.

**Returns:** List of dicts with keys: `soc_init` (float), `soc_req` (float), `duration` (int, hours), `driver_type` (str).

---

## `utils.config_loader`

### `get_config(name: str) → dict`

Load and cache `configs/<name>.yaml`. Singleton — subsequent calls with the same name return the cached dict.

**Args:** `name` — YAML file stem (e.g. `'sac'`, `'env'`, `'training'`).

**Returns:** Parsed YAML as a Python dict.

---

## `utils.constants`

All constants used for normalisation, voltage bounds, and statistical tests. Import directly:

```python
from utils.constants import PRICE_NORM_SCALE, VOLTAGE_MIN, VOLTAGE_MAX, CI_Z_95
```

| Constant | Value | Description |
|----------|-------|-------------|
| `PRICE_NORM_SCALE` | 0.50 | Divides raw $/kWh in state[8:13] |
| `EV_TOTAL_NORM_SCALE` | 0.25 | Clips EV total load (MW) to [0, 1] |
| `EV_DELTA_NORM_SCALE` | 0.10 | Clips EV load delta (MW) to [−1, 1] |
| `VOLTAGE_MIN` | 0.95 | IEEE 1547 lower voltage limit (p.u.) |
| `VOLTAGE_MAX` | 1.05 | IEEE 1547 upper voltage limit (p.u.) |
| `VOLTAGE_BAND` | 0.05 | Half-band for λ_grid scaling |
| `CI_Z_95` | 1.96 | z-score for 95 % CI |
| `ALPHA_SIGNIFICANCE` | 0.05 | p-value threshold for significance tests |
| `CONVERGENCE_WINDOW` | 10 | Rolling window for convergence detection |
| `CONVERGENCE_THRESHOLD` | 0.90 | Fraction of final mean that counts as converged |
| `EPSILON_MIN` | 0.05 | Q-Learning minimum exploration rate |
| `EPSILON_DECAY` | 0.95 | Q-Learning per-episode ε decay |
