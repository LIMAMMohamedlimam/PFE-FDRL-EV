"""
config.py — Default configuration for network simulation
=========================================================
"""

# ── Hierarchical mode (agents → edges → cloud) ──
DEFAULT_CONFIG = {
    # Simulation structure
    'n_agents': 20,
    'n_edges': 3,
    'n_rounds': 5,              # Number of FL aggregation rounds
    'episodes_per_round': 20,    # Training episodes between each FL round
    'simulation_hours': 24,

    # RL settings (reuses existing agents, not modified)
    'policy': 'ppo',            # 'sac' | 'ppo' | 'qlearning'
    'aggregation': 'fedavg',    # 'fedavg' | 'fedopt'
    'use_lora': False,
    'dev_mode': False,           # Use training_dev.yaml for speed

    # Network parameters — Agent (vehicle) uplink
    'agent_bandwidth_mbps': 10,     # Typical LTE uplink
    'agent_latency_ms': 20,        # LTE one-way latency

    # Network parameters — Edge server
    'edge_bandwidth_mbps': 100,     # Edge ↔ Cloud link
    'edge_latency_ms': 5,          # Low-latency edge

    # Network parameters — Cloud server
    'cloud_bandwidth_mbps': 1000,   # Cloud backbone
    'cloud_latency_ms': 50,        # Cross-region latency
}


# ── Cloud-only mode (agents → cloud directly) ──
DEFAULT_CLOUD_ONLY_CONFIG = {
    **DEFAULT_CONFIG,
    'n_edges': 0,       # No edge servers → direct agent-to-cloud
}
