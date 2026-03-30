"""
network_simulator.py — Core network simulation components
==========================================================
Provides node abstractions (Agent, Edge, Cloud) and a NetworkSimulator
that tracks communication costs using the formula:

    time = (size_MB * 8 / bandwidth_Mbps) + (latency_ms / 1000)

All measurements are non-intrusive — they observe parameter dictionaries
returned by existing agent/aggregation code without modifying them.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────
# Node hierarchy
# ─────────────────────────────────────────────────────────────────────

class Node:
    """Base class representing a network node in the FL topology."""

    def __init__(self, node_id: str, node_type: str,
                 bandwidth_mbps: float, latency_ms: float):
        self.node_id = node_id
        self.node_type = node_type          # 'agent' | 'edge' | 'cloud'
        self.bandwidth_mbps = bandwidth_mbps
        self.latency_ms = latency_ms

    def __repr__(self):
        return (f"{self.node_type}({self.node_id}, "
                f"bw={self.bandwidth_mbps}Mbps, lat={self.latency_ms}ms)")


class AgentNode(Node):
    """Vehicle / RL agent node — typically on a wireless uplink."""

    def __init__(self, node_id: str, bandwidth_mbps: float = 10.0,
                 latency_ms: float = 20.0):
        super().__init__(node_id, 'agent', bandwidth_mbps, latency_ms)


class EdgeNode(Node):
    """Edge server — aggregates a cluster of agents."""

    def __init__(self, node_id: str, bandwidth_mbps: float = 100.0,
                 latency_ms: float = 5.0):
        super().__init__(node_id, 'edge', bandwidth_mbps, latency_ms)


class CloudNode(Node):
    """Cloud server — performs global aggregation."""

    def __init__(self, node_id: str = 'cloud',
                 bandwidth_mbps: float = 1000.0, latency_ms: float = 50.0):
        super().__init__(node_id, 'cloud', bandwidth_mbps, latency_ms)


# ─────────────────────────────────────────────────────────────────────
# Communication log entry
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CommunicationLog:
    """Record of a single network transfer."""
    round_num: int
    src_id: str
    src_type: str
    dst_id: str
    dst_type: str
    size_bytes: int
    transfer_time_s: float
    direction: str              # 'upload' or 'download'


# ─────────────────────────────────────────────────────────────────────
# Utility: measure parameter dict size
# ─────────────────────────────────────────────────────────────────────

def measure_params_size(params: dict) -> int:
    """
    Compute the total byte size of a parameter dictionary.

    Works with both numpy arrays (from get_parameters()) and
    torch tensors (fallback via .nelement() * .element_size()).

    Args:
        params: dict of {key: np.ndarray or torch.Tensor}

    Returns:
        Total size in bytes.
    """
    total = 0
    for v in params.values():
        if isinstance(v, np.ndarray):
            total += v.nbytes
        elif hasattr(v, 'nelement') and hasattr(v, 'element_size'):
            # torch.Tensor
            total += v.nelement() * v.element_size()
        else:
            # Fallback: estimate from sys.getsizeof
            import sys
            total += sys.getsizeof(v)
    return total


# ─────────────────────────────────────────────────────────────────────
# Network Simulator
# ─────────────────────────────────────────────────────────────────────

class NetworkSimulator:
    """
    Central tracker for FL communication costs.

    Computes transfer times using:
        time = (size_MB * 8 / bandwidth_Mbps) + (latency_ms / 1000)

    Maintains per-round and total statistics. The simulator is purely
    observational — it does not alter any data that flows through it.
    """

    def __init__(self):
        self.logs: List[CommunicationLog] = []
        self.current_round: int = 0
        self._round_logs: List[CommunicationLog] = []

    def set_round(self, round_num: int):
        """Set the current FL round number for logging."""
        self.current_round = round_num

    def compute_transfer_time(self, size_bytes: int,
                              sender: Node, receiver: Node) -> float:
        """
        Calculate transfer time in seconds between two nodes.

        Formula: time = (size_MB * 8 / bandwidth_Mbps) + (latency_ms / 1000)

        The bandwidth used is the minimum of sender and receiver
        (bottleneck link). Latency is the sum of both endpoints
        (round-trip approximation for one-way: max of the two).
        """
        size_mb = size_bytes / (1024 * 1024)
        # Bottleneck bandwidth = min of the two endpoints
        effective_bw = min(sender.bandwidth_mbps, receiver.bandwidth_mbps)
        # Latency = max of the two (dominant propagation delay)
        effective_latency_ms = max(sender.latency_ms, receiver.latency_ms)

        if effective_bw <= 0:
            raise ValueError(f"Invalid bandwidth: {effective_bw} Mbps")

        transmission_time = (size_mb * 8) / effective_bw
        propagation_time = effective_latency_ms / 1000.0
        return transmission_time + propagation_time

    def log_transfer(self, src: Node, dst: Node,
                     params: dict, direction: str = 'upload') -> CommunicationLog:
        """
        Log a parameter transfer between two nodes.

        Measures the parameter dictionary size, computes transfer time,
        and records the communication event.

        Args:
            src: Sending node.
            dst: Receiving node.
            params: Parameter dictionary being transmitted.
            direction: 'upload' or 'download'.

        Returns:
            The CommunicationLog entry.
        """
        size = measure_params_size(params)
        time_s = self.compute_transfer_time(size, src, dst)

        entry = CommunicationLog(
            round_num=self.current_round,
            src_id=src.node_id,
            src_type=src.node_type,
            dst_id=dst.node_id,
            dst_type=dst.node_type,
            size_bytes=size,
            transfer_time_s=time_s,
            direction=direction,
        )
        self.logs.append(entry)
        self._round_logs.append(entry)
        return entry

    def end_round(self):
        """Finalize the current round and reset per-round accumulators."""
        self._round_logs = []

    # ── Reporting ──

    def get_round_summary(self, round_num: Optional[int] = None) -> dict:
        """Get metrics for a specific round (or current)."""
        rn = round_num if round_num is not None else self.current_round
        round_logs = [l for l in self.logs if l.round_num == rn]
        return self._summarize(round_logs, label=f"Round {rn}")

    def get_total_summary(self) -> dict:
        """Get aggregate metrics across all rounds."""
        summary = self._summarize(self.logs, label="Total")
        n_rounds = len(set(l.round_num for l in self.logs)) if self.logs else 1
        summary['n_rounds'] = n_rounds
        summary['avg_time_per_round'] = summary['total_time_s'] / max(1, n_rounds)
        summary['avg_bytes_per_round'] = summary['total_bytes'] / max(1, n_rounds)
        return summary

    def get_per_node_type_summary(self) -> dict:
        """Break down communication by node type (agent, edge, cloud)."""
        result = {}
        for node_type in ('agent', 'edge', 'cloud'):
            # Bytes sent BY this node type
            sent = [l for l in self.logs if l.src_type == node_type]
            received = [l for l in self.logs if l.dst_type == node_type]
            result[node_type] = {
                'bytes_sent': sum(l.size_bytes for l in sent),
                'bytes_received': sum(l.size_bytes for l in received),
                'transfers_sent': len(sent),
                'transfers_received': len(received),
                'time_sending': sum(l.transfer_time_s for l in sent),
                'time_receiving': sum(l.transfer_time_s for l in received),
            }
        return result

    def _summarize(self, logs: list, label: str = '') -> dict:
        if not logs:
            return {
                'label': label,
                'total_bytes': 0, 'total_mb': 0.0,
                'total_time_s': 0.0, 'n_transfers': 0,
                'upload_bytes': 0, 'download_bytes': 0,
            }
        return {
            'label': label,
            'total_bytes': sum(l.size_bytes for l in logs),
            'total_mb': sum(l.size_bytes for l in logs) / (1024 * 1024),
            'total_time_s': sum(l.transfer_time_s for l in logs),
            'n_transfers': len(logs),
            'upload_bytes': sum(l.size_bytes for l in logs if l.direction == 'upload'),
            'download_bytes': sum(l.size_bytes for l in logs if l.direction == 'download'),
        }

    def reset(self):
        """Clear all logs and reset state."""
        self.logs.clear()
        self._round_logs.clear()
        self.current_round = 0
