# Environment & Data

## 1. IEEE 33-Bus Distribution Network

### Topology

The grid environment is built on the **IEEE 33-bus radial distribution network** (`case33bw`), a standard benchmark for distribution system power-flow studies.

| Property | Value |
|---------|-------|
| Buses | 33 |
| Branches | 32 (radial topology) |
| Base voltage | 12.66 kV |
| Slack bus | Bus 0 (infinite source) |
| Load buses | 32 (buses 1–32) |
| Pandapower network | `pandapower.networks.case33bw()` |

### Code location

`env/GridEnv.py` — class `GridEnv`.

### Power flow

At each timestep the grid runs an **AC power flow** (Newton-Raphson) via pandapower:

```python
pp.runpp(self.net, algorithm='nr')
```

EV load injections are added to the existing base load at each bus before the power flow is solved. If the solver fails to converge (overloaded network), a maximum congestion signal λ = 2.0 is returned.

### Grid congestion signal λ

After power flow, the congestion signal is derived from the worst bus voltage deviation:

```
max_deviation = max |V_i − 1.0|   (in p.u., across all buses)
λ_grid = clip(max_deviation / VOLTAGE_BAND, 0.0, 2.0)
```

Where `VOLTAGE_BAND` is defined in `utils/constants.py`. λ ∈ [0, 2]:
- λ = 0 → no voltage stress
- λ = 1 → voltage at the allowed band boundary
- λ = 2 → convergence failure (maximum penalty)

### Voltage constraints

| Limit | Value | Key |
|-------|-------|-----|
| V_min | `VOLTAGE_MIN` | `utils/constants.py` |
| V_max | `VOLTAGE_MAX` | `utils/constants.py` |

Voltage violations are counted for the statistical evaluation (metric `voltage_violation_rate`).

---

## 2. EV Charging Model (EVClientEnv)

### Code location

`env/EVClientEnv.py` — class `EVClientEnv`.

### Battery dynamics

State of charge update per timestep:

```
SOC_{t+1} = clip(SOC_t + (η · P_act · Δt) / C,  SOC_min, SOC_max)
```

| Symbol | Config key | Default | Unit |
|--------|-----------|---------|------|
| C | `battery_capacity` | 60.0 | kWh |
| η | `eta` | 0.95 | — |
| Δt | `dt` | 1.0 | hours |
| SOC_min | `soc_min` | 0.0 | — |
| SOC_max | `soc_max` | 1.0 | — |

Negative P_act (discharge / V2G) is physically supported — the action space ∈ [−1, 1] maps to [−P_max(SOC), +P_max(SOC)].

### CC-CV charging profile

Real EV chargers taper power as the battery fills. HFDRL models this with:

```
P_max(SOC) = ū · (1 − α · SOC)
```

| Symbol | Config key | Default | Unit |
|--------|-----------|---------|------|
| ū | `max_power` | 11.0 | kW |
| α | `alpha_constraint` | 0.05 | — |

At SOC = 0: P_max = 11.0 kW. At SOC = 1: P_max = 10.45 kW (~5 % reduction).

### Episode horizon

Each episode represents one 24-hour day: `simulation_hours = 24` decision steps, one per hour.

The EV departs at `t_dep` (hours). The done flag is set when `current_step >= t_dep`. After departure, a terminal reward is applied based on whether `SOC >= soc_req`.

---

## 3. State Vector

`EVClientEnv.get_state()` returns a normalised float32 vector.

### Base state (dim = 13)

| Index | Symbol | Description | Normalisation |
|-------|--------|-------------|---------------|
| 0 | SOC | State of charge | raw value ∈ [0, 1] |
| 1 | t_sin | Cyclic hour encoding | sin(2π·h/24) ∈ [−1, 1] |
| 2 | t_cos | Cyclic hour encoding | cos(2π·h/24) ∈ [−1, 1] |
| 3 | t_rem | Remaining time to departure | (t_dep − step) / 24 |
| 4 | λ | Grid congestion signal | clip(λ_grid, 0, 1) |
| 5 | V_dev | Voltage deviation | clip(voltage_dev × 10, −1, 1) |
| 6 | P_ev | Aggregate EV load | clip(ev_total_mw / EV_TOTAL_NORM_SCALE, 0, 1) |
| 7 | ΔP_ev | EV load delta (ramp) | clip(delta_ev_mw / EV_DELTA_NORM_SCALE, −1, 1) |
| 8–12 | p_0..4 | 5-hour price forecast | price / PRICE_NORM_SCALE (from `utils/constants.py`) |

### Extended state (dim = 16, driver behavior enabled)

When `env.yaml → driver_behavior.enabled = true`, three one-hot location features are appended:

| Index | Feature | Value |
|-------|---------|-------|
| 13 | loc_home | 1.0 if at home, else 0 |
| 14 | loc_office | 1.0 if at office, else 0 |
| 15 | loc_driving | 1.0 if driving, else 0 |

Charging is forced to 0 when `loc_driving = 1`.

---

## 4. Reward Function

**Code location:** `utils/reward_functions.py` — function `compute_reward()`.

The reward has five components, each with a weight from `configs/reward.yaml`:

### Components

```
r_A = −w_A · max(0, SOC_req − SOC)                     # distance penalty
r_B = +w_B · ΔSOC   if (ΔSOC > 0 and SOC < SOC_req)    # SOC progress
      0              otherwise
r_C = −w_C · max(0, energy_kWh) · price                 # economic cost
r_D = −w_D · |energy_kWh| · λ_grid                      # grid congestion
r_E = +terminal_success_bonus  if done and SOC ≥ SOC_req # terminal
      −terminal_failure_weight · (SOC_req − SOC)  otherwise
```

### Weight defaults (from `configs/reward.yaml`)

| Component | Key | Default |
|-----------|-----|---------|
| r_A | `w_target_tracking` | 2.0 |
| r_B | `w_progress` | 5.0 |
| r_C | `w_cost` | 0.5 |
| r_D | `w_grid` | 0.3 |
| r_E success | `terminal_success_bonus` | 15.0 |
| r_E failure | `terminal_failure_weight` | 25.0 |

**Expected range:** −1.5 to +0.8 per step; −25 to +20 for the terminal term.

The high `w_B` (5.0) versus `w_C` (0.5) means SOC achievement is 10× more important than cost minimisation, which aligns with the objective that every EV must reach SOC ≥ 0.9 before departure.

---

## 5. Driver Behavior Model

### Code location

`utils/DriverBehaviorModel.py` — class `DriverBehaviorModel`.

Three archetypes model heterogeneous EV usage:

| Archetype | Home departure | Office hours | Typical soc_init | soc_req |
|-----------|---------------|-------------|-----------------|---------|
| Commuter | 7–9 AM | 9 AM – 5 PM | 0.2–0.4 | 0.9 |
| Flexible | variable | variable | 0.3–0.6 | 0.8 |
| Night charger | evening / night | overnight | 0.1–0.3 | 0.95 |

Parameters are sampled from Gaussian distributions around the archetype means. The schedule is pre-generated for `schedule_days` days (default 5) and cycled across episodes.

When `driver_behavior.enabled = false` (default), all EVs are assumed to always be at home and use the `t_dep`, `soc_init`, `soc_req` values drawn from `DataGenerator.get_nhts_profile()` (NHTS 2017-derived profiles).

---

## 6. Market Price Data

### Data files

| File | Rows | Use |
|------|------|-----|
| `data/iso_ne_prices_real.csv` | ~720 (30 days) | Training mode |
| `data/iso_ne_prices_dev_test.csv` | shorter slice | Dev mode |

Both files must have columns `timestamp` and `price` ($/kWh). Prices are min-max normalised to [0, 1] by `utils/MarketPriceLoader.py` before use.

### Price access

**Real prices (default, `use_real_prices: true`):**

`utils/MarketPriceLoader.py` loads and normalises the CSV, then provides `get_price(hour, mode)` with train/test splits.

**Synthetic prices (`use_real_prices: false`):**

`utils/DataLoader.DataGenerator.get_iso_ne_price(hour, mode)` generates ISO-NE style prices from a parametric model. Used as a fallback or for ablations.

### 5-hour forecast

At each decision step, the agent receives a 5-step ahead price forecast:

```python
price_forecast = [get_iso_ne_price((hour + h) % 24, mode='train') for h in range(5)]
```

This is stored normalised in `state[8:13]` (divided by `PRICE_NORM_SCALE` from `utils/constants.py`).
