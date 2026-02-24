import numpy as np


class EdgeAggregator:
    """
    Edge-level intermediate aggregator (FHDP — Federated Hybrid Distributed Parallelism).

    Each Edge node manages a subset of vehicles (V_m) and performs local aggregation
    before forwarding the result to the Cloud server.

    Aggregation formula:
        θ_edge_m = Σ (n_i / N_m) · θ_vehicle_i     for i ∈ V_m

    Usage:
        edge = EdgeAggregator(edge_id=0, vehicle_ids=[0, 1, 2])
        for vid in edge.vehicle_ids:
            edge.collect(vid, agent.get_parameters(), n_samples=24)
        edge_params, total_n = edge.aggregate()
    """

    def __init__(self, edge_id, vehicle_ids=None):
        """
        Args:
            edge_id: Unique identifier for this edge node
            vehicle_ids: List of vehicle indices connected to this edge
        """
        self.edge_id = edge_id
        self.vehicle_ids = list(vehicle_ids) if vehicle_ids else []

        # Collected updates: {vehicle_id: {'params': dict, 'n_samples': int}}
        self._collected = {}

    def assign_vehicles(self, vehicle_ids):
        """Update the set of connected vehicles."""
        self.vehicle_ids = list(vehicle_ids)
        self._collected = {}

    def collect(self, vehicle_id, params, n_samples):
        """
        Collect a model update from a single vehicle.

        Args:
            vehicle_id: Index of the reporting vehicle
            params: dict {key: np.ndarray} from agent.get_parameters()
            n_samples: Number of transitions the vehicle trained on
        """
        self._collected[vehicle_id] = {
            'params': {k: v.copy() if isinstance(v, np.ndarray) else v
                       for k, v in params.items()},
            'n_samples': n_samples,
        }

    def aggregate(self):
        """
        Perform weighted aggregation of collected vehicle updates.

        Handles sparse parameter dicts (e.g. Q-tables where different agents
        may have visited different states) by taking the union of all keys.

        Returns:
            (aggregated_params: dict, total_samples: int)
            Returns (None, 0) if no updates were collected.
        """
        if not self._collected:
            return None, 0

        total_samples = sum(v['n_samples'] for v in self._collected.values())
        if total_samples == 0:
            return None, 0

        # Collect the UNION of all parameter keys across all vehicles
        all_keys = set()
        for vehicle_update in self._collected.values():
            all_keys.update(vehicle_update['params'].keys())

        # Determine the shape/type for each key from whichever vehicle has it
        key_templates = {}
        for key in all_keys:
            for vehicle_update in self._collected.values():
                if key in vehicle_update['params']:
                    key_templates[key] = np.zeros_like(vehicle_update['params'][key])
                    break

        aggregated = {k: v.copy() for k, v in key_templates.items()}

        for vehicle_update in self._collected.values():
            weight = vehicle_update['n_samples'] / total_samples
            for key in all_keys:
                if key in vehicle_update['params']:
                    aggregated[key] += weight * vehicle_update['params'][key]
                # else: contributes zeros (already initialized)

        # Clear buffer for next round
        self._collected = {}

        return aggregated, total_samples

    def n_collected(self):
        """Number of vehicles that have reported in this round."""
        return len(self._collected)

    def n_vehicles(self):
        """Total number of vehicles assigned to this edge."""
        return len(self.vehicle_ids)
