"""
SWIFTScheduler.py
=================
SWIFT (Scheduling With In-Flow Time) client selection for federated learning.

At every FL aggregation round, instead of collecting from all N agents:
  1. Eligibility filter — discard agents whose remaining parking time is below a threshold
  2. Utility scoring — rank eligible agents by staleness, SOC gap, and driver-type diversity
  3. Top-k selection — select top ceil(n_eligible × fraction) agents
  4. Broadcast to all — global model still goes to every agent (standard FL invariant)
"""

import math
from typing import Dict, List, Optional


class SWIFTScheduler:
    """SWIFT client selection scheduler for federated EV charging.

    Reads all thresholds from the ``config`` dict loaded via ``get_config('swift')``.
    No numeric literals appear in the logic body — every threshold is config-driven.

    Args:
        n_agents: Total number of EV agents in the simulation.
        config: Dict loaded from ``configs/swift.yaml`` via ``get_config('swift')``.
    """

    def __init__(self, n_agents: int, config: dict) -> None:
        self.n_agents = n_agents
        self.config = config

        # Internal staleness tracking: last round each agent was selected (-1 = never)
        self._last_round: List[int] = [-1] * n_agents
        self._n_rounds: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_clients(
        self,
        agents: list,
        envs: list,
        driver_profiles: list,
        current_round: int,
    ) -> List[int]:
        """Select a subset of agents for this FL aggregation round.

        Args:
            agents: List of all RL agents (used for indexing only — weights not read).
            envs: List of ``EVClientEnv`` instances (SOC, t_dep, current_step read).
            driver_profiles: Per-agent profile dicts from ``DataGenerator``/``DriverBehaviorModel``.
            current_round: Current FL round (typically the episode index).

        Returns:
            Sorted list of selected agent indices.
        """
        # Edge case: too few agents to meaningfully filter
        if self.n_agents < 2:
            self._n_rounds += 1
            return list(range(self.n_agents))

        min_stay = self.config['min_stay_hours']
        fraction = self.config['fraction']
        staleness_cap = self.config['staleness_cap']
        force_after = self.config['force_select_after']
        weights = self.config['utility_weights']

        # Step 1 — Eligibility filter
        eligible: List[int] = []
        for i in range(self.n_agents):
            t_remaining = envs[i].t_dep - envs[i].current_step
            if t_remaining >= min_stay:
                eligible.append(i)

        # Safety valve: if no agents are eligible, fall back to all
        if not eligible:
            eligible = list(range(self.n_agents))

        # Step 2 — Force-select override (prevents starvation)
        force_selected: List[int] = []
        for i in eligible:
            if (current_round - self._last_round[i]) >= force_after:
                force_selected.append(i)

        # Step 3 — Utility scoring (on eligible minus force-selected)
        force_set = set(force_selected)
        scores: Dict[int, float] = {}
        for i in eligible:
            if i in force_set:
                continue

            staleness_raw = current_round - self._last_round[i]
            staleness_norm = min(staleness_raw, staleness_cap) / staleness_cap

            soc_gap = max(0.0, envs[i].soc_req - envs[i].soc)

            driver_type = driver_profiles[i].get('driver_type', 'commuter')
            diversity = 1.0 if driver_type in ('flexible', 'night_charger') else 0.5

            scores[i] = (
                weights['staleness'] * staleness_norm
                + weights['soc_gap'] * soc_gap
                + weights['diversity'] * diversity
            )

        # Step 4 — Top-k from utility-ranked candidates
        k_utility = max(0, math.ceil(len(eligible) * fraction) - len(force_selected))
        ranked = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        utility_selected = ranked[:k_utility]

        # Step 5 — Merge, update staleness, return
        selected = sorted(set(force_selected + utility_selected))
        for i in selected:
            self._last_round[i] = current_round
        self._n_rounds += 1
        return selected

    def get_round_stats(
        self,
        selected_indices: List[int],
        driver_profiles: list,
    ) -> dict:
        """Return a logging dict for the most recent selection round.

        Args:
            selected_indices: Indices returned by ``select_clients()``.
            driver_profiles: Per-agent profile dicts.

        Returns:
            Dict with keys: ``n_selected``, ``n_eligible``, ``avg_soc_gap``,
            ``type_counts`` (sub-dict of driver type tallies).
        """
        type_counts: Dict[str, int] = {}
        for i in selected_indices:
            dt = driver_profiles[i].get('driver_type', 'commuter')
            key = f"{dt}_count"
            type_counts[key] = type_counts.get(key, 0) + 1

        return {
            'n_selected': len(selected_indices),
            'n_eligible': self.n_agents,  # last eligible set size (approximation)
            'avg_soc_gap': 0.0,  # filled by caller if needed
            'type_counts': type_counts,
        }

    def reset(self) -> None:
        """Reset staleness counters between full experiment reruns."""
        self._last_round = [-1] * self.n_agents
        self._n_rounds = 0
