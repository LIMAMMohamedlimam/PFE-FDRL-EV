"""
config.py — Default configuration for network simulation
=========================================================
"""

# ── Hierarchical mode (agents → edges → cloud) ──
DEFAULT_CONFIG = {
    # Simulation structure
    'n_agents': 20,
    'n_edges': 2,
    'n_rounds': 6,              # Number of FL aggregation rounds
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


# ── LoRA Network Study — bandwidth scenarios ──
# Used by training/LoRANetworkStudy.py (Phase B analytical comm overhead).
# Each entry defines one agent uplink condition.  The authoritative list lives
# in configs/lora_network_study.yaml; this dict is a fallback if the YAML
# cannot be loaded.
BANDWIDTH_SCENARIOS = [
    {'name': '1_Mbps',   'agent_bw': 1,           'agent_lat': 100, 'label': '1 Mbps'},
    {'name': '5_Mbps',   'agent_bw': 5,            'agent_lat': 50,  'label': '5 Mbps'},
    {'name': '10_Mbps',  'agent_bw': 10,           'agent_lat': 20,  'label': '10 Mbps'},
    {'name': '100_Mbps', 'agent_bw': 100,          'agent_lat': 5,   'label': '100 Mbps'},
    {'name': 'variable', 'agent_bw': 'variable',
     'bw_range': [1, 5, 10],                        'agent_lat': 50,  'label': 'Variable'},
]
