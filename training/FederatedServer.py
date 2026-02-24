import numpy as np
import copy


class FederatedServer:
    """
    Cloud-level global aggregation server.

    Supports two strategies:
      - **FedAvg**: Weighted average of edge parameters.
      - **FedOpt**: Server-side momentum (FedAvgM / FedAdam-style) for faster convergence.

    Usage:
        server = FederatedServer(strategy='fedavg')
        server.initialize(global_params)   # set initial global model
        ...
        server.aggregate(edge_updates)     # [{params: dict, n_samples: int}, ...]
        global_params = server.broadcast()
    """

    def __init__(self, strategy='fedavg', server_lr=1.0, beta=0.9):
        """
        Args:
            strategy: 'fedavg' or 'fedopt'
            server_lr: Learning rate for FedOpt server-side update
            beta: Momentum coefficient for FedOpt
        """
        assert strategy in ('fedavg', 'fedopt'), f"Unknown strategy: {strategy}"
        self.strategy = strategy
        self.server_lr = server_lr
        self.beta = beta

        self.global_params = None       # dict {key: np.ndarray}
        self.momentum = None            # dict {key: np.ndarray}  (FedOpt only)
        self.round_number = 0

    def initialize(self, params):
        """Set the initial global model parameters."""
        self.global_params = {k: v.copy() for k, v in params.items()}
        if self.strategy == 'fedopt':
            self.momentum = {k: np.zeros_like(v) for k, v in params.items()}

    def aggregate(self, edge_updates):
        """
        Aggregate edge-level model updates into a new global model.

        Handles sparse parameter dicts (e.g. Q-tables) by using the union
        of all keys from edges and global model.

        Args:
            edge_updates: list of dicts, each with:
                - 'params': dict {key: np.ndarray}
                - 'n_samples': int (number of data points behind this edge)

        Returns:
            Updated global parameters (dict).
        """
        if not edge_updates:
            return self.global_params

        # --- Weighted average across edges ---
        total_samples = sum(u['n_samples'] for u in edge_updates)
        if total_samples == 0:
            return self.global_params

        # Collect union of all keys from edges AND global model
        all_keys = set(self.global_params.keys())
        for update in edge_updates:
            all_keys.update(update['params'].keys())

        # Determine shape/type template for each key
        key_templates = {}
        for key in all_keys:
            if key in self.global_params:
                key_templates[key] = np.zeros_like(self.global_params[key])
            else:
                # Key only exists in edge updates, find a template
                for update in edge_updates:
                    if key in update['params']:
                        key_templates[key] = np.zeros_like(update['params'][key])
                        break

        aggregated = {k: v.copy() for k, v in key_templates.items()}

        for update in edge_updates:
            weight = update['n_samples'] / total_samples
            for key in all_keys:
                if key in update['params']:
                    aggregated[key] += weight * update['params'][key]

        if self.strategy == 'fedavg':
            # Direct replacement
            self.global_params = aggregated

        elif self.strategy == 'fedopt':
            # Server-side momentum update
            # Ensure momentum has all keys
            for key in all_keys:
                if key not in self.momentum:
                    self.momentum[key] = np.zeros_like(key_templates[key])
                if key not in self.global_params:
                    self.global_params[key] = np.zeros_like(key_templates[key])

            for key in all_keys:
                delta = aggregated[key] - self.global_params[key]
                self.momentum[key] = self.beta * self.momentum[key] + (1 - self.beta) * delta
                self.global_params[key] = self.global_params[key] + self.server_lr * self.momentum[key]

        self.round_number += 1
        return self.global_params

    def broadcast(self):
        """Return a copy of the current global parameters for distribution."""
        if self.global_params is None:
            return None
        return {k: v.copy() for k, v in self.global_params.items()}

    def get_round(self):
        return self.round_number
