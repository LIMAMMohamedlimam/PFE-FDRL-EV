"""
HeuristicAgents.py
==================
Rule-based EV charging baselines that implement the BaseAgent interface.
All agents return continuous actions in [-1, 1] and have no-op FL methods.

State vector layout (EVClientEnv.get_state — base 13-dim):
  state[0]    = SOC (0-1)
  state[1]    = t_sin  (cyclic hour encoding)
  state[2]    = t_cos  (cyclic hour encoding)
  state[3]    = t_remaining_norm = (t_dep - current_step) / 24.0
  state[4]    = grid signal (normalised)
  state[5]    = voltage deviation (normalised)
  state[6]    = EV aggregate load (normalised)
  state[7]    = EV load delta (normalised)
  state[8:13] = 5-hour price forecast (each stored as raw_price / 0.5)
"""

import numpy as np
from agents.BaseAgent import BaseAgent


# ---------------------------------------------------------------------------
# Abstract heuristic base
# ---------------------------------------------------------------------------

class HeuristicAgent(BaseAgent):
    """Shared FL stubs and interface bridge for all rule-based agents."""

    def __init__(self, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def get_parameters(self):
        return {}

    def set_parameters(self, parameters):
        pass

    def save_trained_model(self, *args, **kwargs):
        pass

    def load_trained_model(self, *args, **kwargs):
        pass

    def select_action(self, state) -> float:
        raise NotImplementedError

    def get_action(self, state, eval_mode: bool = False) -> float:
        return self.select_action(state)


# ---------------------------------------------------------------------------
# Concrete heuristics
# ---------------------------------------------------------------------------

class RandomAgent(HeuristicAgent):
    """Uniform random action in [-1, 1] at every step."""

    def select_action(self, state) -> float:
        return float(np.random.uniform(-1.0, 1.0))


class GreedyAgent(HeuristicAgent):
    """Charge at maximum power whenever SOC < soc_req, otherwise idle."""

    def __init__(self, soc_req: float = 0.9, **kwargs):
        self.soc_req = soc_req

    def select_action(self, state) -> float:
        soc = float(state[0])
        return 1.0 if soc < self.soc_req else 0.0


class EarliestDeadlineFirst(HeuristicAgent):
    """
    Charge proportional to urgency = (soc_req - soc) / time_remaining.
    Urgency is clipped to [-1, 1]. If time_remaining <= 0, action = +1.
    """

    def __init__(self, soc_req: float = 0.9, **kwargs):
        self.soc_req = soc_req

    def select_action(self, state) -> float:
        soc = float(state[0])
        # state[3] is normalised as (t_dep - step) / 24.0; convert to hours
        t_remaining = float(state[3]) * 24.0

        if t_remaining <= 0:
            return 1.0

        urgency = (self.soc_req - soc) / t_remaining
        return float(np.clip(urgency, -1.0, 1.0))


class PriceAwareAgent(HeuristicAgent):
    """
    Charge at max when price is cheap, idle when expensive, linear in between.

    Thresholds apply to the normalised price stored in state[8]
    (raw_price / 0.5, so 0.3 norm ≈ $0.15/kWh, 0.7 norm ≈ $0.35/kWh).
    """

    def __init__(
        self,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
        **kwargs,
    ):
        self.low = low_threshold
        self.high = high_threshold

    def select_action(self, state) -> float:
        price_norm = float(state[8])

        if price_norm < self.low:
            return 1.0
        elif price_norm > self.high:
            return 0.0
        else:
            return 1.0 - (price_norm - self.low) / (self.high - self.low)


class SimpleMPCAgent(HeuristicAgent):
    """
    1-step lookahead MPC over candidate actions {-1, 0, +1}.

    For each action, estimates next SOC and selects the action that minimises:
        cost_weight * price * |action| - soc_weight * soc_progress

    where soc_progress = max(0, next_soc - current_soc).

    Uses the current (first) price from the 5-hour forecast in state[8:13].
    """

    def __init__(
        self,
        soc_req: float = 0.9,
        cost_weight: float = 1.0,
        soc_weight: float = 1.0,
        capacity: float = 60.0,
        max_power: float = 7.0,
        eta: float = 0.95,
        dt: float = 1.0,
        **kwargs,
    ):
        self.soc_req = soc_req
        self.cost_weight = cost_weight
        self.soc_weight = soc_weight
        self.capacity = capacity
        self.max_power = max_power
        self.eta = eta
        self.dt = dt

    def select_action(self, state) -> float:
        soc = float(state[0])
        # state[8] = raw_price / 0.5  →  raw_price = state[8] * 0.5
        price = float(state[8]) * 0.5

        best_action = 0.0
        best_value = float('inf')

        for action in (-1.0, 0.0, 1.0):
            p_kw = action * self.max_power
            delta_soc = (self.eta * p_kw * self.dt) / self.capacity
            next_soc = float(np.clip(soc + delta_soc, 0.0, 1.0))

            soc_progress = max(0.0, next_soc - soc)
            value = self.cost_weight * price * abs(action) - self.soc_weight * soc_progress

            if value < best_value:
                best_value = value
                best_action = action

        return float(best_action)
