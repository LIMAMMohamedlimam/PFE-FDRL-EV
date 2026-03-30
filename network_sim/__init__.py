"""
network_sim — Network Simulation Layer for FL Communication Overhead
====================================================================
Non-intrusive simulation layer that measures and compares communication
costs between Cloud-only and Hierarchical FL architectures.

This package wraps existing training/aggregation logic without modifying it.
"""

from network_sim.network_simulator import (
    Node, AgentNode, EdgeNode, CloudNode,
    NetworkSimulator, measure_params_size,
)
from network_sim.simulation_runner import (
    run_network_simulation, run_comparison, SimulationResult,
)
from network_sim.config import DEFAULT_CONFIG, DEFAULT_CLOUD_ONLY_CONFIG
