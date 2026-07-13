# Glossary

Canonical expansions for all acronyms and terms used in the paper, code, and documentation. When an abbreviation appears anywhere in this project, this file is the single source of truth for its meaning.

---

## Algorithms & Methods

| Acronym | Expansion | Context |
|---------|-----------|---------|
| **DRL** | Deep Reinforcement Learning | General framework |
| **HFDRL** | Hierarchical Federated Deep Reinforcement Learning | This project's proposed method: SAC + HFedAvg + SWIFT + LoRA |
| **RL** | Reinforcement Learning | General framework |
| **MARL** | Multi-Agent Reinforcement Learning | Multiple EV agents learning simultaneously |
| **SAC** | Soft Actor-Critic | Off-policy continuous-action RL algorithm (Haarnoja et al., 2018) |
| **PPO** | Proximal Policy Optimisation | On-policy continuous-action RL algorithm (Schulman et al., 2017) |
| **MDP** | Markov Decision Process | Mathematical framework for sequential decision making |
| **MPC** | Model Predictive Control | Optimisation-based control using a lookahead window |
| **EDF** | Earliest Deadline First | Scheduling heuristic; here adapted for EV urgency |
| **TD** | Temporal Difference | RL learning signal using bootstrapped value estimates |

---

## Federated Learning

| Acronym | Expansion | Context |
|---------|-----------|---------|
| **FL** | Federated Learning | Distributed training without sharing raw data |
| **FedAvg** | Federated Averaging | McMahan et al. (2017); weighted average of client models |
| **HFedAvg** | Hierarchical Federated Averaging | Two-level aggregation: vehicle → edge → cloud |
| **FedProx** | Federated Proximal | FedAvg + proximal regularisation term μ/2‖w−w_g‖² (Li et al., 2020) |
| **FedAvgM** | Federated Averaging with Momentum | Server-side momentum on the pseudo-gradient |
| **FedAdam** | Federated Adam | Server-side Adam optimiser on the pseudo-gradient (Reddi et al., 2020) |
| **FedOpt** | Federated Optimisation | Alias for FedAvgM in this codebase (backwards compatibility) |
| **FHDP** | Federated Hybrid Distributed Parallelism | Intermediate edge aggregation pattern used by `EdgeAggregator` |
| **SWIFT** | Scheduling With In-Flow Time | Client selection policy for EV FL: eligibility filter + utility scoring |

---

## Model Compression

| Acronym | Expansion | Context |
|---------|-----------|---------|
| **LoRA** | Low-Rank Adaptation | Parameter-efficient fine-tuning (Hu et al., 2021); freezes W₀, trains low-rank A, B |

---

## Power Systems & EV

| Acronym | Expansion | Context |
|---------|-----------|---------|
| **EV** | Electric Vehicle | The charging agents in the simulation |
| **SOC** | State of Charge | Battery energy level, normalised ∈ [0, 1] |
| **CC-CV** | Constant Current – Constant Voltage | Standard EV charging profile; modelled as P_max = ū·(1−α·SOC) |
| **V2G** | Vehicle-to-Grid | Bi-directional power flow: EV discharging back into the grid |
| **IEEE 33-bus** | IEEE 33-bus radial distribution network | Standard benchmark grid topology (`case33bw` in pandapower) |
| **p.u.** | Per unit | Voltage normalisation: 1.0 p.u. = nominal voltage (12.66 kV here) |
| **MW** | Megawatt | Grid-level power unit (EV loads in kW are divided by 1000) |
| **kWh** | Kilowatt-hour | Energy unit for battery capacity and energy cost |
| **MEC** | Mobile Edge Computing | Low-latency edge servers close to end users |
| **LTE** | Long Term Evolution | 4G mobile standard; used as vehicle uplink model (10 Mbps, 20 ms) |

---

## Datasets & Organisations

| Acronym | Expansion | Context |
|---------|-----------|---------|
| **ISO-NE** | Independent System Operator – New England | Source of real hourly wholesale electricity price data |
| **NHTS** | National Household Travel Survey | US Bureau of Transportation Statistics survey; basis for driver archetype profiles |
| **ACN** | Adaptive Charging Network | Caltech EV charging dataset (Lee et al., 2019); alternative driver archetype source |

---

## Statistics & Evaluation

| Acronym | Expansion | Context |
|---------|-----------|---------|
| **CI95** | 95 % Confidence Interval | CI = mean ± 1.96 × std / √n; z = 1.96 from `constants.py` |
| **MSE** | Mean Squared Error | Critic loss function in SAC |
| **AAAI** | Association for the Advancement of Artificial Intelligence | Target venue for the paper; drives the 10-seed evaluation protocol |

---

## Project

| Acronym | Expansion | Context |
|---------|-----------|---------|
| **PFE** | Projet de Fin d'Études | End-of-studies project (equivalent to master's thesis) |

---

## Key Terms

| Term | Definition |
|------|-----------|
| **Grid congestion signal (λ)** | `clip(max_voltage_deviation / VOLTAGE_BAND, 0, 2)`. Broadcast to all agents as a shared coordination signal. See `GridEnv.step()`. |
| **Driver archetype** | Statistical model of a driver's daily routine: departure time, arrival time, soc_init, soc_req. Three archetypes: commuter, flexible, night_charger. |
| **Dwell time** | Remaining parking hours: `t_dep − current_step`. Minimum dwell (`min_stay_hours`) gates SWIFT eligibility. |
| **Staleness** | Number of FL rounds since an agent last participated. Used in SWIFT utility: stale agents are prioritised to prevent training starvation. |
| **Utility score (SWIFT)** | `U = w_s·staleness_norm + w_g·soc_gap + w_d·diversity`. Higher = more valuable to select. |
| **FL round** | One aggregation cycle: collect → edge aggregate → cloud aggregate → broadcast. Typically one round per episode. |
| **Convergence episode** | Smallest episode at which the rolling-mean reward reaches 90 % of its final value (window = 10, threshold from `constants.py`). |
| **Pseudo-gradient (FedAvgM / FedAdam)** | δ = w_aggregated − w_global. Treated as a gradient by the server-side optimiser. |
| **Proximal term (FedProx)** | μ/2·‖w_local − w_global‖². Added to actor and critic losses in `SACAgent._learn()` when strategy = 'fedprox'. |
| **LoRA scaling** | `alpha / rank`. Scales the LoRA delta contribution; initialized so LoRA starts as identity (B = 0). |
| **Warmup steps** | `warmup_steps = 2000` random steps before the first SAC gradient update. Ensures the replay buffer is diverse before learning begins. |
| **Soft update (τ)** | `θ_target ← τ·θ + (1−τ)·θ_target`. Stabilises SAC critic by slowly tracking the online network. |
| **Auto-entropy tuning** | SAC automatically adjusts temperature α to keep policy entropy near the target H̄ = −0.5·action_dim. No manual scheduling needed. |
| **Tanh squash** | Output of GaussianPolicy passes through tanh to bound actions to [−1, 1]. Log-prob corrected for the squash Jacobian. |
| **Run directory** | Timestamped output folder under `results/` for a single simulation run. Contains plots, metrics JSON, optional CSV. |
