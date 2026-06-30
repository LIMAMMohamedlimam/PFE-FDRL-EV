# Method Documentation — HFDRL

HFDRL combines four components: **SAC** (local RL), **HFedAvg** (hierarchical federation), **SWIFT** (client selection), and **LoRA** (model compression). This document maps each component to the paper section and the implementing code.

---

## 1. Soft Actor-Critic (SAC)

### Role in the system

Each EV agent runs a local SAC policy. SAC is chosen for its sample efficiency (off-policy replay buffer) and automatic entropy tuning, which balances exploration/exploitation without manual temperature scheduling.

### Code location

`agents/SACAgent.py` — classes `GaussianPolicy`, `TwinQNetwork`, `SACAgent`.

### Architecture

```
State s_t (dim=13 or 16)
       │
       ▼
GaussianPolicy (actor)
  Linear(d_in, 128) → ReLU
  Linear(128,  128) → ReLU
  ├─ mu_head:     Linear(128, 1) → μ
  └─ log_std_head: Linear(128, 1) → log σ (clamped to [−20, 2])
       │
       ▼  reparameterisation + tanh squash
  action â ∈ [−1, 1]

TwinQNetwork (critic)
  Q1: Linear(d_in+1, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 1)
  Q2: same structure (independent weights)
  returns (Q₁(s,a), Q₂(s,a))
```

### Update equations

**Critic (clipped double-Q):**

```
y = r + γ(1−d) · [min(Q₁ᵀ, Q₂ᵀ)(s', â') − α log π(â'|s')]
L_critic = MSE(Q₁(s,a), y) + MSE(Q₂(s,a), y)
```

**Actor (SAC policy gradient):**

```
L_actor = 𝔼[α log π(â|s) − min(Q₁, Q₂)(s, â)]
```

**Entropy temperature (automatic tuning):**

```
L_α = −𝔼[log α · (log π(â|s) + H̄)]
H̄ = target_entropy_scale × action_dim = −0.5   (from sac.yaml)
```

**Target network soft update:**

```
θᵀ ← τ · θ + (1−τ) · θᵀ     τ = 0.005
```

### Key hyperparameters (from `configs/sac.yaml`)

| Symbol | Key | Value | Role |
|--------|-----|-------|------|
| γ | `gamma` | 0.99 | Discount factor |
| τ | `tau` | 0.005 | Target network EMA |
| η_lr | `lr` | 3e-4 | Learning rate (actor, critic, α) |
| d_h | `hidden_dim` | 128 | Hidden layer width |
| B | `batch_size` | 256 | Mini-batch size |
| N_warm | `warmup_steps` | 2 000 | Steps before first gradient update |
| H̄ | `target_entropy_scale` | −0.5 | Target entropy = −0.5 × action_dim |

### Code ↔ paper mapping

| Paper symbol | Code variable | Location |
|-------------|--------------|----------|
| π_θ | `SACAgent.actor` (GaussianPolicy) | `SACAgent.py:192` |
| Q_φ | `SACAgent.critic` (TwinQNetwork) | `SACAgent.py:193` |
| Q̄_φ | `SACAgent.critic_target` | `SACAgent.py:194` |
| α | `SACAgent.alpha` (auto-tuned) | `SACAgent.py:232` |
| D | `SACAgent.buffer` (ReplayBuffer) | `SACAgent.py:236` |
| τ | `SACAgent.tau` | `SACAgent.py:182` |
| γ | `SACAgent.gamma` | `SACAgent.py:181` |

---

## 2. Hierarchical Federated Averaging (HFedAvg)

### Role in the system

After each episode, selected agents upload their local model (or LoRA adapters) to their edge aggregator. Each edge aggregates locally and forwards a single update to the cloud server. The cloud applies the global aggregation strategy and broadcasts the result to all agents.

This two-level hierarchy (vehicle → edge → cloud) reduces cross-WAN traffic by a factor of `num_agents / num_edges` compared to flat cloud-only federation.

### Code location

- `training/FederatedServer.py` — cloud-level aggregation
- `training/EdgeAggregator.py` — edge-level intermediate aggregation

### Aggregation strategies

All strategies are implemented in `FederatedServer.aggregate()`. The strategy is set via `configs/sac.yaml → federated.aggregation`, overridden per-method in `configs/training.yaml`.

| Strategy | Server-side update | Key hyperparameter |
|----------|-------------------|--------------------|
| `fedavg` | w_i = n_i / Σn_j; w_global ← Σ w_i · w_i | — |
| `fedprox` | Same as FedAvg; proximal term μ/2‖w−w_g‖² added client-side | μ = 0.01 |
| `fedavgm` | velocity = β·v + (1−β)·Δ; w_global += η·velocity | β = 0.9, η = 1.0 |
| `fedadam` | Adam on pseudo-gradient Δ = w_agg − w_global | β₁=0.9, β₂=0.99, η=0.01 |
| `fedopt` | Alias for `fedavgm` (backwards compat.) | same |

**Note:** For `fedprox`, the proximal penalty is enforced client-side in `SACAgent._learn()` via `SACAgent.set_fedprox_global()`. The server-side aggregation is identical to FedAvg.

### HFedAvg round (pseudocode)

```
For each FL round:
    selected = SWIFTScheduler.select_clients(...)  # if SWIFT enabled
    For each edge e:
        local_updates = [agent[i].get_parameters() for i in selected ∩ edge_e.vehicle_ids]
        edge_params = EdgeAggregator.aggregate(local_updates)   # weighted FedAvg
    global_params = FederatedServer.aggregate(edge_updates)     # strategy-specific
    For each agent i:
        agent[i].set_parameters(global_params)
```

### Code ↔ paper mapping

| Paper symbol | Code | Location |
|-------------|------|----------|
| w_global | `FederatedServer.global_params` | `FederatedServer.py:67` |
| w_i | `agent.get_parameters()` → numpy dict | `SACAgent.py:271` |
| HFedAvg round | `FederatedServer.aggregate()` | `FederatedServer.py:85` |
| Edge aggregation | `EdgeAggregator.aggregate()` | `EdgeAggregator.py` |
| n_i (sample weight) | `update['n_samples']` | `FederatedServer.py:103` |

---

## 3. SWIFT Client Selection

### Role in the system

SWIFT (Scheduling With In-Flow Time) replaces the default "collect from all agents" with a principled selection that (1) excludes EVs about to depart, (2) prioritises agents not seen recently, and (3) ensures representation across driver archetypes.

This matters for EV charging because an EV with 10 minutes of parking left cannot meaningfully contribute to or benefit from an FL update.

### Code location

`training/SWIFTScheduler.py` — class `SWIFTScheduler`.

### Selection algorithm

```
Step 1 — Eligibility filter
    eligible = {i : t_dep(i) − t_current ≥ min_stay_hours}
    if eligible = ∅: eligible = all agents  (safety fallback)

Step 2 — Force-select (starvation prevention)
    force_selected = {i ∈ eligible : round − last_round[i] ≥ force_select_after}

Step 3 — Utility scoring (on eligible \ force_selected)
    staleness_norm(i) = min(round − last_round[i], staleness_cap) / staleness_cap
    soc_gap(i)        = max(0, soc_req(i) − soc(i))
    diversity(i)      = 1.0 if driver_type ∈ {flexible, night_charger} else 0.5
    U(i) = w_s · staleness_norm + w_g · soc_gap + w_d · diversity

Step 4 — Top-k selection
    k = ceil(|eligible| × fraction) − |force_selected|
    utility_selected = top-k(eligible \ force_selected, key=U)

Step 5 — Merge and update staleness
    selected = sorted(force_selected ∪ utility_selected)
    last_round[i] = current_round  for each i ∈ selected
```

### Configuration (from `configs/swift.yaml`)

| Key | Default | Symbol | Role |
|-----|---------|--------|------|
| `min_stay_hours` | 2 | τ_min | Eligibility threshold (hours) |
| `fraction` | 0.6 | f | Fraction of eligible agents selected |
| `utility_weights.staleness` | 0.4 | w_s | Staleness weight |
| `utility_weights.soc_gap` | 0.4 | w_g | SOC-gap weight |
| `utility_weights.diversity` | 0.2 | w_d | Diversity weight |
| `staleness_cap` | 15 | s_cap | Staleness normalisation ceiling |
| `force_select_after` | 20 | f_after | Rounds before forced selection |

### Code ↔ paper mapping

| Paper symbol | Code | Location |
|-------------|------|----------|
| τ_min | `config['min_stay_hours']` | `SWIFTScheduler.py:63` |
| U(i) | `scores[i]` | `SWIFTScheduler.py:101` |
| eligible set | `eligible` list | `SWIFTScheduler.py:70` |
| selected set | `selected` | `SWIFTScheduler.py:113` |
| last_round[i] | `self._last_round[i]` | `SWIFTScheduler.py:114` |

---

## 4. LoRA Model Compression

### Role in the system

Without LoRA, each FL round transmits the full actor + critic weight dictionaries (~MB per agent). With LoRA, only the low-rank adapter matrices A and B are transmitted. For `rank=4` and `hidden_dim=128`, this reduces the transmitted weight per wrapped layer from 128×128 = 16 384 parameters to 2×(4×128) = 1 024 — a **94 % reduction per layer**.

The base weights W₀ are frozen at initialization and never updated or transmitted. The effective weight during forward passes is:

```
W_eff = W₀ + B · A · (alpha / rank)
```

### Code location

`utils/lora.py` — classes `LoRALinear`, functions `apply_lora`, `get_lora_state_dict`, `load_lora_state_dict`.

### Initialization

```
A ∈ ℝ^{rank × d_in}   initialized Kaiming-uniform  (non-zero for gradient flow)
B ∈ ℝ^{d_out × rank}  initialized zeros             (LoRA starts as identity: Δ=0)
scaling = alpha / rank
```

### FL aggregation with LoRA

`SACAgent.get_parameters()` returns only LoRA weights when `use_lora=True`:

```python
params.update(get_lora_state_dict(self.actor,  prefix='actor.'))
params.update(get_lora_state_dict(self.critic, prefix='critic.'))
```

`SACAgent.set_parameters()` loads only LoRA weights:

```python
load_lora_state_dict(self.actor,  actor_params,  prefix='actor.',  device=self.device)
load_lora_state_dict(self.critic, critic_params, prefix='critic.', device=self.device)
```

The FederatedServer and EdgeAggregator are unaware of LoRA — they operate on whatever dict `get_parameters()` returns.

### Target modules (from `configs/lora.yaml`)

| Module | Belongs to | Wrapped layers |
|--------|-----------|----------------|
| `fc1`, `fc2` | GaussianPolicy (actor) | Hidden layers |
| `mu_head`, `log_std_head` | GaussianPolicy (actor) | Output heads |
| `q1`, `q2` | TwinQNetwork (critic) | Both Q-networks (via Sequential index) |
| `critic`, `actor_mu` | PPOAgent ActorCritic | PPO hidden + actor head |

### Configuration (from `configs/lora.yaml`)

| Key | Default | Symbol | Role |
|-----|---------|--------|------|
| `enabled` | `false` | — | Master switch (also `USE_LORA=true` env var) |
| `rank` | 4 | r | Low-rank dimension |
| `alpha` | 8 | α_L | Scaling factor; effective = α_L / r |
| `target_modules` | see file | — | Layer name substrings to wrap |

### Code ↔ paper mapping

| Paper symbol | Code | Location |
|-------------|------|----------|
| W₀ | `LoRALinear.weight` (frozen) | `lora.py:59` |
| A | `LoRALinear.lora_A` | `lora.py:68` |
| B | `LoRALinear.lora_B` | `lora.py:69` |
| α_L / r | `LoRALinear.scaling` | `lora.py:57` |
| W_eff | `LoRALinear.forward()` output | `lora.py:72` |
| LoRA-only FL params | `get_lora_state_dict()` | `lora.py:164` |
| rank r | `lora.yaml → rank` | `configs/lora.yaml:11` |
