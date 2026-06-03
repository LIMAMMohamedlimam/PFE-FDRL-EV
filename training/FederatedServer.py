import numpy as np
import copy


# Strategies implemented at the server level.
# Note: 'fedprox' proximal penalty is applied client-side (SACAgent._learn);
#       at the server it aggregates identically to 'fedavg'.
_VALID_STRATEGIES = frozenset({'fedavg', 'fedopt', 'fedprox', 'fedavgm', 'fedadam'})


class FederatedServer:
    """
    Cloud-level global aggregation server.

    Supported strategies
    --------------------
    fedavg   : Weighted FedAvg — direct parameter replacement.
    fedprox  : Same aggregation as FedAvg; proximal term is enforced client-side
               via SACAgent.set_fedprox_global().
    fedopt   : Alias kept for backwards compatibility → momentum update (same as fedavgm).
    fedavgm  : FedAvg with server-side momentum.
               velocity = β·velocity + (1-β)·delta
               global  += server_lr · velocity
    fedadam  : Server-side Adam on the aggregated pseudo-gradient.
               m = β1·m + (1-β1)·delta
               v = β2·v + (1-β2)·delta²
               global += lr · m_hat / (√v_hat + ε)

    Usage
    -----
        server = FederatedServer(strategy='fedavg')
        server.initialize(global_params)   # set initial global model
        ...
        server.aggregate(edge_updates)     # [{params: dict, n_samples: int}, ...]
        global_params = server.broadcast()
    """

    def __init__(
        self,
        strategy: str = 'fedavg',
        server_lr: float = 1.0,
        beta: float = 0.9,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.99,
        adam_eps: float = 1e-3,
    ):
        """
        Args:
            strategy   : One of _VALID_STRATEGIES.
            server_lr  : Step size for FedAvgM / FedAdam server updates.
            beta       : Momentum coefficient (FedAvgM / fedopt).
            adam_beta1 : First-moment decay for FedAdam.
            adam_beta2 : Second-moment decay for FedAdam.
            adam_eps   : Numerical stability constant for FedAdam.
        """
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Valid: {sorted(_VALID_STRATEGIES)}"
            )
        self.strategy = strategy
        self.server_lr = server_lr
        self.beta = beta
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps

        self.global_params = None   # dict {key: np.ndarray}
        self.momentum = None        # FedAvgM / fedopt velocity
        self.adam_m = None          # FedAdam first moment
        self.adam_v = None          # FedAdam second moment
        self.adam_t = 0             # FedAdam step counter
        self.round_number = 0

    def initialize(self, params: dict):
        """Set the initial global model parameters."""
        self.global_params = {k: v.copy() for k, v in params.items()}

        if self.strategy in ('fedopt', 'fedavgm'):
            self.momentum = {k: np.zeros_like(v) for k, v in params.items()}
        elif self.strategy == 'fedadam':
            self.adam_m = {k: np.zeros_like(v) for k, v in params.items()}
            self.adam_v = {k: np.zeros_like(v) for k, v in params.items()}
            self.adam_t = 0

    def aggregate(self, edge_updates: list) -> dict:
        """
        Aggregate edge-level model updates into a new global model.

        Handles sparse parameter dicts (e.g. Q-tables) by using the union
        of all keys from edges and the global model.

        Args:
            edge_updates: list of dicts, each with:
                - 'params'   : dict {key: np.ndarray}
                - 'n_samples': int (data points behind this edge)

        Returns:
            Updated global parameters (dict).
        """
        if not edge_updates:
            return self.global_params

        total_samples = sum(u['n_samples'] for u in edge_updates)
        if total_samples == 0:
            return self.global_params

        # --- Weighted average across edges ---
        all_keys = set(self.global_params.keys())
        for update in edge_updates:
            all_keys.update(update['params'].keys())

        key_templates = {}
        for key in all_keys:
            if key in self.global_params:
                key_templates[key] = np.zeros_like(self.global_params[key])
            else:
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

        # --- Apply strategy-specific server update ---
        if self.strategy in ('fedavg', 'fedprox'):
            self.global_params = aggregated

        elif self.strategy in ('fedopt', 'fedavgm'):
            for key in all_keys:
                if key not in self.momentum:
                    self.momentum[key] = np.zeros_like(key_templates[key])
                if key not in self.global_params:
                    self.global_params[key] = np.zeros_like(key_templates[key])

            for key in all_keys:
                delta = aggregated[key] - self.global_params[key]
                self.momentum[key] = self.beta * self.momentum[key] + (1 - self.beta) * delta
                self.global_params[key] = (
                    self.global_params[key] + self.server_lr * self.momentum[key]
                )

        elif self.strategy == 'fedadam':
            self.adam_t += 1
            for key in all_keys:
                if key not in self.adam_m:
                    self.adam_m[key] = np.zeros_like(key_templates[key])
                if key not in self.adam_v:
                    self.adam_v[key] = np.zeros_like(key_templates[key])
                if key not in self.global_params:
                    self.global_params[key] = np.zeros_like(key_templates[key])

            bc1 = 1.0 - self.adam_beta1 ** self.adam_t
            bc2 = 1.0 - self.adam_beta2 ** self.adam_t

            for key in all_keys:
                delta = aggregated[key] - self.global_params[key]
                self.adam_m[key] = (
                    self.adam_beta1 * self.adam_m[key] + (1 - self.adam_beta1) * delta
                )
                self.adam_v[key] = (
                    self.adam_beta2 * self.adam_v[key] + (1 - self.adam_beta2) * delta ** 2
                )
                m_hat = self.adam_m[key] / bc1
                v_hat = self.adam_v[key] / bc2
                self.global_params[key] = (
                    self.global_params[key]
                    + self.server_lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
                )

        self.round_number += 1
        return self.global_params

    def broadcast(self) -> dict:
        """Return a copy of the current global parameters for distribution."""
        if self.global_params is None:
            return None
        return {k: v.copy() for k, v in self.global_params.items()}

    def collect_selected(self, agents: list, selected_indices: list) -> list:
        """Collect local parameters from a subset of agents.

        Drop-in replacement for the full-collection path when SWIFT is active.

        Args:
            agents:           Full list of agents.
            selected_indices: Indices of agents selected by SWIFTScheduler.

        Returns:
            list of dicts: [{'params': dict, 'n_samples': int}, ...]
        """
        edge_updates = []
        for i in selected_indices:
            edge_updates.append({
                'params': agents[i].get_parameters(),
                'n_samples': 1,
            })
        return edge_updates

    def get_round(self) -> int:
        return self.round_number
