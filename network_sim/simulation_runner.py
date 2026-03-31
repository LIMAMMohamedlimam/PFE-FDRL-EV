"""
simulation_runner.py — Run FL communication overhead simulations
================================================================
Provides two modes:
    Mode A — Cloud-only:  agents → cloud directly
    Mode B — Hierarchical: agents → edges → cloud

Uses existing training loops and aggregation logic unmodified.
Communication is measured through the wrapper layer (wrappers.py).

Example usage:
    from network_sim import run_comparison, DEFAULT_CONFIG
    results = run_comparison(DEFAULT_CONFIG)
"""

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from network_sim.network_simulator import (
    NetworkSimulator, AgentNode, EdgeNode, CloudNode, measure_params_size,
)
from network_sim.wrappers import InstrumentedAgent, InstrumentedEdge, InstrumentedServer
from network_sim.config import DEFAULT_CONFIG, DEFAULT_CLOUD_ONLY_CONFIG

# Existing project modules (NOT modified, only imported)
from env.EVClientEnv import EVClientEnv
from env.GridEnv import GridEnv
from utils.DataLoader import DataGenerator
from agents.SACAgent import SACAgent
from agents.PPOAgent import PPOAgent
from training.FederatedServer import FederatedServer
from training.EdgeAggregator import EdgeAggregator
from utils.config_loader import get_config


# ─────────────────────────────────────────────────────────────────────
# SimulationResult
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Structured output from a network simulation run."""
    mode: str                       # 'cloud_only' or 'hierarchical'
    total_bytes: int = 0
    total_mb: float = 0.0
    total_time_s: float = 0.0
    avg_time_per_round: float = 0.0
    avg_bytes_per_round: float = 0.0
    n_rounds: int = 0
    n_agents: int = 0
    n_edges: int = 0
    upload_bytes: int = 0
    download_bytes: int = 0
    per_node_type: Dict = field(default_factory=dict)
    round_summaries: List[Dict] = field(default_factory=list)
    model_size_bytes: int = 0       # Size of one model

    def to_dict(self) -> dict:
        return {
            'mode': self.mode,
            'total_bytes': self.total_bytes,
            'total_mb': self.total_mb,
            'total_time_s': self.total_time_s,
            'avg_time_per_round': self.avg_time_per_round,
            'avg_bytes_per_round': self.avg_bytes_per_round,
            'n_rounds': self.n_rounds,
            'n_agents': self.n_agents,
            'n_edges': self.n_edges,
            'upload_bytes': self.upload_bytes,
            'download_bytes': self.download_bytes,
            'model_size_bytes': self.model_size_bytes,
        }


# ─────────────────────────────────────────────────────────────────────
# Helpers — reuse the same environment/agent creation patterns
# ─────────────────────────────────────────────────────────────────────

def _create_envs(n_agents: int, dev_mode: bool = True):
    """Create EV environments and profiles (same logic as ComparisonPipeline)."""
    cfg = get_config('training_dev' if dev_mode else 'training')
    ev_cap = cfg.get('ev_capacity', 60)
    ev_max = cfg.get('ev_max_power', 7.4)

    profiles = DataGenerator.get_nhts_profile(n_agents)
    envs = []
    for i in range(n_agents):
        env_cfg = {
            'capacity': ev_cap,
            'max_power': ev_max,
            'initial_soc': profiles[i]['soc_init'],
            'soc_req': profiles[i]['soc_req'],
            't_dep': profiles[i]['duration'],
            'dt': 1.0,
        }
        envs.append(EVClientEnv(env_cfg))
    return envs, profiles, cfg


def _create_agents(policy: str, n_agents: int, input_dim: int,
                   use_lora: bool = False):
    """Create RL agents (same logic as ComparisonPipeline._make_agents)."""
    agents = []
    for _ in range(n_agents):
        if policy == 'sac':
            agents.append(SACAgent(input_dim=input_dim, action_dim=1,
                                   use_lora=use_lora))
        elif policy == 'ppo':
            agents.append(PPOAgent(input_dim=input_dim, action_dim=1,
                                   use_lora=use_lora))
        else:
            raise ValueError(f"Unsupported policy for network sim: {policy}. "
                             f"Use 'sac' or 'ppo'.")
    return agents


def _run_training_episode(agents, envs, profiles, grid, sim_hours: int):
    """
    Run one training episode using existing agent logic.
    Simplified version of ComparisonPipeline's inner loop — matches the
    real step/get_state signatures so communication interception is valid.
    """
    total_reward = 0.0
    grid.reset()

    n_agents = len(agents)
    active = [True] * n_agents
    lambda_prev = 0.0
    volt_prev = 0.0

    for i, env in enumerate(envs):
        if hasattr(env, 'driver_behavior_enabled') and env.driver_behavior_enabled:
            env.reset(initial_soc=profiles[i]['soc_init'])
        else:
            env.soc = profiles[i]['soc_init']
            env.current_step = 0

    for hour in range(sim_hours):
        price = DataGenerator.get_iso_ne_price(hour, mode='train')
        price_forecast = [
            DataGenerator.get_iso_ne_price((hour + h) % 24, mode='train')
            for h in range(5)
        ]
        base_load_mw = np.random.normal(3.5, 0.2)

        # 1) Agent actions
        states = {}
        actions = {}
        grid_injections = {}
        for i in range(n_agents):
            if not active[i]:
                continue
            state = envs[i].get_state(lambda_prev, volt_prev, price_forecast)
            states[i] = state
            raw = agents[i].get_action(state, eval_mode=False)
            # Scale action to power (kW)
            p_kw = float(raw) * envs[i]._get_max_power(envs[i].soc)
            actions[i] = (raw, p_kw)
            bus = i % 32 + 1  # simple bus mapping
            grid_injections[bus] = grid_injections.get(bus, 0.0) + (p_kw / 1000.0)

        # 2) Grid physics step
        lambda_grid, grid_info = grid.step(grid_injections, base_load_mw)
        lambda_prev = float(lambda_grid)
        volt_prev = float(grid_info.get('max_voltage', 1.0) - 1.0)

        # 3) Agent learning
        for i in range(n_agents):
            if not active[i] or i not in actions:
                continue
            raw, p_kw = actions[i]
            r_t, done, _, energy_cost = envs[i].step(
                p_kw, lambda_grid, volt_prev, price
            )
            next_state = envs[i].get_state(lambda_grid, volt_prev, price_forecast)
            try:
                agents[i].update(states[i], raw, r_t, next_state, done=done)
            except TypeError:
                agents[i].update(states[i], raw, r_t, next_state)
            total_reward += r_t
            if done:
                active[i] = False

    return total_reward


# ─────────────────────────────────────────────────────────────────────
# Main simulation function
# ─────────────────────────────────────────────────────────────────────

def run_network_simulation(config: dict = None) -> SimulationResult:
    """
    Run an FL network simulation measuring communication overhead.

    HOW TO PLUG INTO EXISTING TRAINING LOOP:
    -----------------------------------------
    1. Real agents are created normally (SACAgent / PPOAgent).
    2. Each agent is wrapped in InstrumentedAgent(real_agent, node, sim).
    3. Real EdgeAggregators are wrapped in InstrumentedEdge.
    4. Real FederatedServer is wrapped in InstrumentedServer.
    5. The training loop runs normally — wrappers intercept parameter
       transfers to log communication costs.
    6. After all rounds, metrics are extracted from the NetworkSimulator.

    Args:
        config: Dict with simulation parameters. See config.py for defaults.

    Returns:
        SimulationResult with all communication metrics.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    n_agents = cfg['n_agents']
    n_edges = cfg['n_edges']
    n_rounds = cfg['n_rounds']
    episodes_per_round = cfg.get('episodes_per_round', 1)
    policy = cfg['policy']
    aggregation = cfg['aggregation']
    use_lora = cfg.get('use_lora', False)
    dev_mode = cfg.get('dev_mode', True)

    is_cloud_only = (n_edges == 0)
    mode_name = 'cloud_only' if is_cloud_only else 'hierarchical'

    print(f"\n{'='*60}")
    print(f"  Network Simulation: {mode_name.upper()}")
    print(f"  Policy={policy} | Aggregation={aggregation} | LoRA={use_lora}")
    print(f"  Agents={n_agents} | Edges={n_edges} | Rounds={n_rounds}")
    print(f"{'='*60}")

    # ── Create network nodes ──
    simulator = NetworkSimulator()

    agent_nodes = [
        AgentNode(f"agent_{i}",
                  bandwidth_mbps=cfg['agent_bandwidth_mbps'],
                  latency_ms=cfg['agent_latency_ms'])
        for i in range(n_agents)
    ]

    cloud_node = CloudNode(
        bandwidth_mbps=cfg['cloud_bandwidth_mbps'],
        latency_ms=cfg['cloud_latency_ms'],
    )

    if is_cloud_only:
        edge_nodes = []
    else:
        edge_nodes = [
            EdgeNode(f"edge_{i}",
                     bandwidth_mbps=cfg['edge_bandwidth_mbps'],
                     latency_ms=cfg['edge_latency_ms'])
            for i in range(n_edges)
        ]

    # ── Create real RL components ──
    envs, profiles, train_cfg = _create_envs(n_agents, dev_mode)
    sim_hours = int(train_cfg.get('simulation_hours',
                                  cfg.get('simulation_hours', 24)))
    grid = GridEnv()

    dummy_state = envs[0].get_state(0.0, 0.0, [0.0] * 5)
    input_dim = len(dummy_state)
    real_agents = _create_agents(policy, n_agents, input_dim, use_lora)

    # ── Wrap agents with instrumentation ──
    agents = []
    for i, real_agent in enumerate(real_agents):
        wrapped = InstrumentedAgent(real_agent, agent_nodes[i], simulator)
        if is_cloud_only:
            # Cloud-only: agents upload directly to cloud
            wrapped.set_upload_target(cloud_node)
        else:
            # Hierarchical: each agent uploads to its assigned edge
            edge_idx = min(i * n_edges // n_agents, n_edges - 1)
            wrapped.set_upload_target(edge_nodes[edge_idx])
        agents.append(wrapped)

    # ── Create and wrap FL infrastructure ──
    real_server = FederatedServer(strategy=aggregation)
    # Seed global model (not counted as communication)
    real_server.initialize(real_agents[0].get_parameters())

    if is_cloud_only:
        # Cloud-only mode: broadcast back to all agent nodes
        server = InstrumentedServer(
            real_server, cloud_node,
            broadcast_targets=agent_nodes,
            simulator=simulator,
        )
        edges = []
    else:
        # Hierarchical mode: broadcast to edge nodes (edges relay to agents)
        server = InstrumentedServer(
            real_server, cloud_node,
            broadcast_targets=edge_nodes,
            simulator=simulator,
        )
        # Create real edges and wrap them
        vehicles_per_edge = np.array_split(range(n_agents), n_edges)
        edges = []
        for eid, vids in enumerate(vehicles_per_edge):
            real_edge = EdgeAggregator(edge_id=eid, vehicle_ids=list(vids))
            wrapped_edge = InstrumentedEdge(
                real_edge, edge_nodes[eid], cloud_node, simulator,
            )
            edges.append(wrapped_edge)

    # Measure model size once
    model_params = real_agents[0].get_parameters()
    model_size = measure_params_size(model_params)
    print(f"  Model size: {model_size:,} bytes ({model_size/1024:.1f} KB)")

    # ── Training + FL rounds ──
    for round_num in range(n_rounds):
        simulator.set_round(round_num)

        # --- Local training ---
        for ep in range(episodes_per_round):
            _run_training_episode(agents, envs, profiles, grid, sim_hours)

        # --- FL aggregation ---
        if is_cloud_only:
            # Mode A: agents → cloud directly
            # 1) Agents upload to cloud (logged by InstrumentedAgent.get_parameters)
            agent_updates = []
            for i, agent in enumerate(agents):
                params = agent.get_parameters()   # logs agent→cloud upload
                agent_updates.append({
                    'params': params,
                    'n_samples': sim_hours,
                })

            # 2) Cloud aggregates (logged by InstrumentedServer)
            global_params = server.aggregate(agent_updates)

            # 3) Broadcast back (download logged by InstrumentedAgent.set_parameters)
            for agent in agents:
                agent.set_parameters(global_params, log_transfer=False)

        else:
            # Mode B: agents → edges → cloud
            # 1) Agents → Edges (logged by InstrumentedAgent.get_parameters)
            for edge in edges:
                for vid in edge.vehicle_ids:
                    params = agents[vid].get_parameters()  # logs agent→edge
                    edge.collect(vid, params, sim_hours)

            # 2) Edges aggregate → Cloud (logged by InstrumentedEdge.aggregate)
            edge_updates = []
            for edge in edges:
                params, n = edge.aggregate()  # logs edge→cloud
                if params is not None:
                    edge_updates.append({'params': params, 'n_samples': n})

            # 3) Cloud aggregates + broadcasts (logged by InstrumentedServer)
            if edge_updates:
                global_params = server.aggregate(edge_updates)

                # 4) Download: edge relay to agents
                #    (logged by InstrumentedAgent.set_parameters)
                for agent in agents:
                    agent.set_parameters(global_params)

        round_summary = simulator.get_round_summary(round_num)
        print(f"  Round {round_num+1}/{n_rounds}: "
              f"{round_summary['total_mb']:.2f} MB, "
              f"{round_summary['total_time_s']:.4f}s")
        simulator.end_round()

    # ── Build result ──
    total = simulator.get_total_summary()
    per_node = simulator.get_per_node_type_summary()

    result = SimulationResult(
        mode=mode_name,
        total_bytes=total['total_bytes'],
        total_mb=total['total_mb'],
        total_time_s=total['total_time_s'],
        avg_time_per_round=total.get('avg_time_per_round', 0),
        avg_bytes_per_round=total.get('avg_bytes_per_round', 0),
        n_rounds=n_rounds,
        n_agents=n_agents,
        n_edges=n_edges,
        upload_bytes=total['upload_bytes'],
        download_bytes=total['download_bytes'],
        per_node_type=per_node,
        model_size_bytes=model_size,
    )

    print(f"\n  {'─'*50}")
    print(f"  TOTAL: {result.total_mb:.2f} MB | {result.total_time_s:.4f}s")
    print(f"  Upload: {result.upload_bytes / (1024*1024):.2f} MB | "
          f"Download: {result.download_bytes / (1024*1024):.2f} MB")
    return result


# ─────────────────────────────────────────────────────────────────────
# Comparison runner
# ─────────────────────────────────────────────────────────────────────

def run_comparison(config: dict = None,
                   save_csv: str = None) -> Dict[str, SimulationResult]:
    """
    Run both Cloud-only and Hierarchical modes, print comparison.

    Args:
        config: Base config dict (hierarchical settings).
                Cloud-only is derived by setting n_edges=0.
        save_csv: Optional path to save results as CSV.

    Returns:
        Dict with keys 'cloud_only' and 'hierarchical'.
    """
    # base_cfg = {**DEFAULT_CONFIG, **(config or {})}
    base_cfg = DEFAULT_CONFIG

    print("base_cfg",  base_cfg)

    # ── Mode A: Cloud-only ──
    cloud_cfg = {**base_cfg, 'n_edges': 0}
    result_cloud = run_network_simulation(cloud_cfg)

    # ── Mode B: Hierarchical ──
    hier_cfg = {**base_cfg}
    if hier_cfg['n_edges'] == 0:
        hier_cfg['n_edges'] = 3  # Default 3 edges if not set
    result_hier = run_network_simulation(hier_cfg)

    # ── Print comparison table ──
    print(f"\n{'═'*70}")
    print(f"  COMPARISON: Cloud-Only vs Hierarchical")
    print(f"{'═'*70}")
    print(f"{'Metric':<35} {'Cloud-Only':>15} {'Hierarchical':>15}")
    print(f"{'─'*70}")

    rows = [
        ('Agents', f"{result_cloud.n_agents}", f"{result_hier.n_agents}"),
        ('Edges', f"{result_cloud.n_edges}", f"{result_hier.n_edges}"),
        ('Rounds', f"{result_cloud.n_rounds}", f"{result_hier.n_rounds}"),
        ('Model size (KB)', f"{result_cloud.model_size_bytes/1024:.1f}",
                            f"{result_hier.model_size_bytes/1024:.1f}"),
        ('', '', ''),
        ('Total data (MB)', f"{result_cloud.total_mb:.2f}",
                            f"{result_hier.total_mb:.2f}"),
        ('Upload (MB)', f"{result_cloud.upload_bytes/(1024**2):.2f}",
                        f"{result_hier.upload_bytes/(1024**2):.2f}"),
        ('Download (MB)', f"{result_cloud.download_bytes/(1024**2):.2f}",
                          f"{result_hier.download_bytes/(1024**2):.2f}"),
        ('', '', ''),
        ('Total time (s)', f"{result_cloud.total_time_s:.4f}",
                           f"{result_hier.total_time_s:.4f}"),
        ('Avg time/round (s)', f"{result_cloud.avg_time_per_round:.4f}",
                               f"{result_hier.avg_time_per_round:.4f}"),
    ]

    for label, cv, hv in rows:
        if label == '':
            print(f"{'─'*70}")
        else:
            print(f"  {label:<33} {cv:>15} {hv:>15}")

    # ── Savings ──
    if result_cloud.total_time_s > 0:
        time_savings = (1 - result_hier.total_time_s /
                        result_cloud.total_time_s) * 100
        print(f"\n  ⏱  Hierarchical time savings: {time_savings:.1f}%")
    if result_cloud.total_bytes > 0:
        data_savings = (1 - result_hier.total_bytes /
                        result_cloud.total_bytes) * 100
        print(f"  📦 Hierarchical data savings:  {data_savings:.1f}%")

    print(f"{'═'*70}\n")

    # ── Optional CSV export ──
    results = {'cloud_only': result_cloud, 'hierarchical': result_hier}
    if save_csv:
        _save_csv(results, save_csv)

    return results


def _save_csv(results: dict, path: str):
    """Save comparison results to CSV."""
    import csv
    fieldnames = list(results['cloud_only'].to_dict().keys())

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results.values():
            writer.writerow(result.to_dict())

    print(f"  Results saved to: {path}")


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='FL Network Communication Overhead Simulation')
    parser.add_argument('--agents', type=int, default=10, help='Number of agents')
    parser.add_argument('--edges', type=int, default=3, help='Number of edge servers')
    parser.add_argument('--rounds', type=int, default=5, help='Number of FL rounds')
    parser.add_argument('--policy', default='sac', choices=['sac', 'ppo'])
    parser.add_argument('--aggregation', default='fedavg',
                        choices=['fedavg', 'fedopt'])
    parser.add_argument('--lora', action='store_true', help='Enable LoRA')
    parser.add_argument('--csv', type=str, default=None, help='Save CSV path')
    parser.add_argument('--mode', default='compare',
                        choices=['compare', 'cloud', 'hierarchical'])
    args = parser.parse_args()

    cfg = {
        'n_agents': args.agents,
        'n_edges': args.edges,
        'n_rounds': args.rounds,
        'policy': args.policy,
        'aggregation': args.aggregation,
        'use_lora': args.lora,
    }

    if args.mode == 'compare':
        run_comparison(cfg, save_csv=args.csv)
    elif args.mode == 'cloud':
        cfg['n_edges'] = 0
        run_network_simulation(cfg)
    elif args.mode == 'hierarchical':
        run_network_simulation(cfg)
