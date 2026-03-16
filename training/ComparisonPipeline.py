"""
ComparisonPipeline.py
=====================
Unified runner that trains every combination of
  {Q-Learning, PPO, SAC} × {Standalone, FedAvg, FedOpt}
and produces comparative plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')           # headless – no display needed
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Project modules
from env.EVClientEnv import EVClientEnv
from env.GridEnv import GridEnv
from utils.EvalMetrics import EvalMetrics
from utils.DataLoader import DataGenerator
from agents.QLearningAgent import QLearningAgent
from agents.PPOAgent import PPOAgent
from agents.SACAgent import SACAgent
from training.FederatedServer import FederatedServer
from training.EdgeAggregator import EdgeAggregator
from utils.device_utils import device_info
from utils.config_loader import get_config


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _make_envs_and_profiles(n_agents, ev_capacity, ev_max_power):
    """Create EV environments and matching driver profiles."""
    profiles = DataGenerator.get_nhts_profile(n_agents)
    envs = []
    for i in range(n_agents):
        cfg = {
            'capacity': ev_capacity,
            'max_power': ev_max_power,
            'initial_soc': profiles[i]['soc_init'],
            'soc_req': profiles[i]['soc_req'],
            't_dep': profiles[i]['duration'],
            'dt': 1.0,
        }
        envs.append(EVClientEnv(cfg))
    return envs, profiles


def _make_agents(policy, n_agents, input_dim, cfg):
    """Instantiate N agents of the given policy type."""
    agents = []
    for _ in range(n_agents):
        if policy == 'qlearning':
            agents.append(QLearningAgent(
                action_space_size=3,
                state_bins=None,
                epsilon=cfg.get('epsilon_init', 1.0),
                learning_rate=cfg.get('learning_rate', 0.1),
                gamma=cfg.get('gamma', 0.99),
            ))
        elif policy == 'ppo':
            agents.append(PPOAgent(
                input_dim=input_dim,
                action_dim=1,
                lr=cfg.get('learning_rate', 1e-4),
                update_timestep=cfg.get('update_timestep', 240),
                K_epochs=cfg.get('k_epochs', 10),
            ))
        elif policy == 'sac':
            agents.append(SACAgent(
                input_dim=input_dim,
                action_dim=1,
                lr=cfg.get('learning_rate', 3e-4),
                batch_size=cfg.get('batch_size', 256),
                warmup_steps=cfg.get('warmup_steps', 500),
            ))
        else:
            raise ValueError(f"Unknown policy: {policy}")
    return agents


def _action_to_power(agent, action, env, policy):
    """Convert raw agent output to physical power (kW)."""
    p_max = env._get_max_power(env.soc)
    if policy == 'qlearning':
        # Discrete: 0=discharge, 1=idle, 2=charge
        if action == 0:
            return -p_max
        elif action == 1:
            return 0.0
        else:
            return p_max
    else:
        # Continuous [-1, 1] → kW
        return action * p_max


# ────────────────────────────────────────────────────────────────────────────
# Core training function (unified for all policies and aggregation modes)
# ────────────────────────────────────────────────────────────────────────────

def run_single_experiment(
    policy='ppo',
    aggregation='none',       # 'none' | 'fedavg' | 'fedopt'
    verbose=True,
    progress_enabled=True,
    dev_mode=False,
    **extra_cfg,
):
    """
    Train and evaluate a single policy+aggregation combination using YAML configs.

    Returns:
        metrics: EvalMetrics object with all logged data
    """
    # Load configurations
    if dev_mode:
        train_cfg = get_config('training_dev')
    else:
        train_cfg = get_config('training')
    env_cfg = get_config('env')
    
    n_episodes = train_cfg.get('num_episodes', 300)
    n_test_episodes = train_cfg.get('num_test_episodes', 10)
    n_agents = train_cfg.get('num_agents', 10)
    sim_hours = train_cfg.get('simulation_hours', 24)
    grid_type = train_cfg.get('grid_type', 'case33bw')
    
    ev_capacity = env_cfg.get('battery_capacity', 60.0)
    ev_max_power = env_cfg.get('max_power', 11.0)
    
    # We use 2 edge servers for FHDP currently
    n_edges = 2
    fl_rounds_per_episode = 1

    combo_name = f"{policy}_{aggregation}"
    run_name = f"{combo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Base configuration passed down (SAC loads its own via sac.yaml now)
    cfg = {
        'gamma': 0.99,
        'epsilon_init': 1.0,
        'epsilon_decay': 0.95,
        'epsilon_min': 0.05,
        'learning_rate': {'qlearning': 0.1, 'ppo': 1e-4, 'sac': 3e-4}[policy],
        'update_timestep': 240,
        'k_epochs': 10,
        'batch_size': 256,
        'warmup_steps': 200,
        **extra_cfg,
    }

    sim_config = {
        'type': combo_name,
        'policy': policy,
        'aggregation': aggregation,
        'n_episodes': n_episodes,
        'n_agents': n_agents,
        'n_edges': n_edges,
        'simulation_hours': sim_hours,
        'n_test_episodes': n_test_episodes,
        **cfg,
    }

    metrics = EvalMetrics(run_name=run_name, config=sim_config)
    grid = GridEnv(network_type=grid_type)
    envs, profiles = _make_envs_and_profiles(n_agents, ev_capacity, ev_max_power)
    agent_bus_map = {i: (i % 30) + 2 for i in range(n_agents)}

    # Input dimension probe
    dummy_state = envs[0].get_state(0.0, 0.0, [0.0] * 5)
    input_dim = len(dummy_state)
    agents = _make_agents(policy, n_agents, input_dim, cfg)

    # --- FL infrastructure (if applicable) ---
    server = None
    edges = []
    if aggregation != 'none':
        server = FederatedServer(strategy=aggregation)
        server.initialize(agents[0].get_parameters())  # seed global model

        # Distribute vehicles across edges
        vehicles_per_edge = np.array_split(range(n_agents), n_edges)
        for eid, vids in enumerate(vehicles_per_edge):
            edges.append(EdgeAggregator(edge_id=eid, vehicle_ids=list(vids)))

    # ══════════════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════════════
    desc = f"Train {combo_name}"
    
    # We disable tqdm if either verbose is False or if progress_enabled is False
    show_tqdm = verbose and progress_enabled
    for episode in tqdm(range(n_episodes), desc=desc, disable=not show_tqdm):
        total_reward = 0.0
        total_cost = 0.0
        grid.reset()

        active = [True] * n_agents
        lambda_prev = 0.0
        volt_prev = 0.0
        prev_ev_total_mw = 0.0

        for i, env in enumerate(envs):
            env.soc = profiles[i]['soc_init']
            env.current_step = 0

        for hour in range(sim_hours):
            price = DataGenerator.get_iso_ne_price(hour, mode='train')
            price_forecast = [DataGenerator.get_iso_ne_price((hour + h) % 24, mode='train') for h in range(5)]
            base_load_mw = np.random.normal(3.5, 0.2)

            # -- aggregate target (for stability penalty) --
            p_target_kw = 0.0
            n_active = 0
            for i, env in enumerate(envs):
                if not active[i]:
                    continue
                n_active += 1
                soc_gap = max(0.0, env.soc_req - env.soc)
                e_gap = (soc_gap * env.capacity) / env.eta
                t_left = max(1, env.t_dep - env.current_step)
                p_target_kw += min(e_gap / t_left, env._get_max_power(env.soc))
            p_target_mw = p_target_kw / 1000.0

            current_states = {}
            actions = {}
            grid_injections = {}

            # 1) Agent actions
            for i, agent in enumerate(agents):
                if not active[i]:
                    continue
                s_t = envs[i].get_state(lambda_prev, volt_prev, price_forecast)
                current_states[i] = s_t
                raw = agent.get_action(s_t, eval_mode=False)
                p_kw = _action_to_power(agent, raw, envs[i], policy)
                actions[i] = (raw, p_kw)
                bus = agent_bus_map[i]
                grid_injections[bus] = grid_injections.get(bus, 0.0) + (p_kw / 1000.0)

            # 2) Grid physics
            lambda_grid, grid_info = grid.step(grid_injections, base_load_mw)
            ev_total_mw = float(sum(grid_injections.values()))
            delta_ev_mw = ev_total_mw - prev_ev_total_mw

            # Stability penalty (continuous policies)
            # Extracted to reward function usually, but keeping global tracking here
            SCALE = 0.10
            r_ramp = -2.0 * (delta_ev_mw / SCALE) ** 2
            r_track = -1.0 * ((ev_total_mw - p_target_mw) / SCALE) ** 2
            shared_penalty = (r_ramp + r_track) / max(1, n_active)

            prev_ev_total_mw = ev_total_mw
            lambda_prev = float(lambda_grid)
            volt_prev = float(grid_info['max_voltage'] - 1.0)
            metrics.log_step(base_load_mw + sum(grid_injections.values()))

            # 3) Agent learning
            for i, agent in enumerate(agents):
                if not active[i]:
                    continue
                raw, p_kw = actions[i]
                r_t, done, _, energy_cost = envs[i].step(p_kw, lambda_grid, grid_info['max_voltage'] - 1.0, price)
                total_cost += energy_cost
                r_t += shared_penalty
                s_next = envs[i].get_state(lambda_grid, grid_info['max_voltage'] - 1.0, price_forecast)

                try:
                    agent.update(current_states[i], raw, r_t, s_next, done=done)
                except TypeError:
                    agent.update(current_states[i], raw, r_t, s_next)

                total_reward += r_t
                if done:
                    active[i] = False

        # -- episode-end satisfaction --
        sats = []
        for i, env in enumerate(envs):
            req = profiles[i]['soc_req']
            sats.append(min(1.0, env.soc / req) if req > 0 else 1.0)
        metrics.log_satisfaction(sats)
        metrics.log_episode(total_reward, mode='train')
        metrics.log_cost(total_cost)

        # Epsilon decay (Q-Learning)
        if policy == 'qlearning':
            for agent in agents:
                agent.epsilon = max(cfg['epsilon_min'], agent.epsilon * cfg['epsilon_decay'])

        # ── Federated aggregation round ──
        if aggregation != 'none' and (episode + 1) % fl_rounds_per_episode == 0:
            # 1) Vehicles → Edges
            for edge in edges:
                for vid in edge.vehicle_ids:
                    n_samples = sim_hours  # each vehicle contributed sim_hours transitions
                    edge.collect(vid, agents[vid].get_parameters(), n_samples)

            # 2) Edges → Cloud
            edge_updates = []
            for edge in edges:
                params, n = edge.aggregate()
                if params is not None:
                    edge_updates.append({'params': params, 'n_samples': n})

            # 3) Cloud aggregation
            if edge_updates:
                global_params = server.aggregate(edge_updates)

                # 4) Broadcast back to all vehicles
                for agent in agents:
                    agent.set_parameters(global_params)

        if verbose and (episode + 1) % 10 == 0:
            avg_r = np.mean(metrics.episode_rewards[-10:])
            print(f"  [{combo_name}] Ep {episode+1} | AvgR: {avg_r:.2f} | Cost: ${total_cost:.2f}")

    # ══════════════════════════════════════════════════════════════════
    # TESTING
    # ══════════════════════════════════════════════════════════════════
    for test_ep in range(n_test_episodes):
        total_test_reward = 0.0
        grid.reset()
        for env in envs:
            env.soc = 0.2
            env.current_step = 0

        active = [True] * n_agents
        lambda_prev = 0.0
        volt_prev = 0.0
        prev_ev_total_mw = 0.0

        for hour in range(sim_hours):
            price = DataGenerator.get_iso_ne_price(hour, mode='test')
            pf = [DataGenerator.get_iso_ne_price((hour + h) % 24, mode='test') for h in range(5)]
            base_load = np.random.normal(3.8, 0.3)

            grid_inj = {}
            actions = {}

            for i, agent in enumerate(agents):
                if not active[i]:
                    continue
                s_t = envs[i].get_state(lambda_prev, volt_prev, pf)
                raw = agent.get_action(s_t, eval_mode=True)
                p_kw = _action_to_power(agent, raw, envs[i], policy)
                actions[i] = (raw, p_kw)
                bus = agent_bus_map[i]
                grid_inj[bus] = grid_inj.get(bus, 0.0) + (p_kw / 1000.0)

            lambda_grid, grid_info = grid.step(grid_inj, base_load)
            ev_total_mw = float(sum(grid_inj.values()))

            for i, agent in enumerate(agents):
                if not active[i]:
                    continue
                _, p_kw = actions[i]
                r_t, done, _, _ = envs[i].step(p_kw, lambda_grid, grid_info['max_voltage'] - 1.0, price)
                total_test_reward += r_t
                if done:
                    active[i] = False

            prev_ev_total_mw = ev_total_mw
            lambda_prev = float(lambda_grid)
            volt_prev = float(grid_info['max_voltage'] - 1.0)

        metrics.log_episode(total_test_reward, mode='test')

    return metrics


# ────────────────────────────────────────────────────────────────────────────
# Full comparison across all combinations
# ────────────────────────────────────────────────────────────────────────────

def _run_combo(args):
    """
    Top-level (picklable) worker for ProcessPoolExecutor.
    Each worker process instantiates its own agents, envs, and (if available)
    its own CUDA context — so multiple combos run truly in parallel.
    """
    p, a, kwargs, progress_enabled, dev_mode = args
    combo_name = f"{p}_{a}"
    m = run_single_experiment(
        policy=p,
        aggregation=a,
        verbose=False,   # suppress per-worker tqdm noise in parallel mode
        progress_enabled=progress_enabled,
        dev_mode=dev_mode,
        **kwargs,
    )
    m.plot_metrics()    # saves PNG + CSV inside the worker
    return combo_name, m


def run_comparison(
    policies=('qlearning', 'ppo', 'sac'),
    aggregations=('none', 'fedavg', 'fedopt'),
    max_workers=None,   # None → auto (min(combos, cpu_count))
    dev_mode=False,
    **kwargs,
):
    """
    Run all (policy × aggregation) combinations and produce comparison plots.

    Experiments are dispatched to a ProcessPoolExecutor so multiple combos
    run in parallel. Each worker process gets its own GPU context (if CUDA is
    available), saturating available compute.

    Args:
        policies:        Tuple of policy names to test
        aggregations:    Tuple of aggregation strategies to test
        max_workers:     Max parallel workers (default: min(n_combos, cpu_count))
        dev_mode:        Whether to run in development mode
        **kwargs:        Extra config forwarded to run_single_experiment

    Returns:
        results: dict {combo_name: EvalMetrics}
    """
    results = {}
    combos = [(p, a) for p in policies for a in aggregations]
    n_combos = len(combos)
    cpu_count = mp.cpu_count()
    workers = min(n_combos, cpu_count) if max_workers is None else max_workers
    if dev_mode:
        train_cfg = get_config('training_dev')
    else:
        train_cfg = get_config('training')
    n_episodes = train_cfg.get('num_episodes', 300)
    n_test_episodes = train_cfg.get('num_test_episodes', 10)
    
    progress_cfg = train_cfg.get('progress', {})
    progress_enabled = progress_cfg.get('enabled', True)

    print(f"\n{'='*60}")
    print(f"  COMPARISON PIPELINE: {n_combos} combinations")
    print(f"  Episodes: {n_episodes} train + {n_test_episodes} test each")
    print(f"  Compute device : {device_info()}")
    print(f"  Parallel workers: {workers} / {cpu_count} CPU cores")
    print(f"{'='*60}\n")

    # Build argument list for workers
    work_items = [
        (p, a, kwargs, progress_enabled, dev_mode)
        for p, a in combos
    ]

    if workers == 1:
        # Single-process path: cleaner tracebacks during debugging
        for item in work_items:
            combo_name, m = _run_combo(item)
            print(f"  ✓ Done: {combo_name}")
            results[combo_name] = m
    else:
        # Multi-process path: true parallelism
        # Use spawn context to prevent CUDA poison fork errors
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            future_map = {pool.submit(_run_combo, item): item[0] + '_' + item[1]
                          for item in work_items}
            
            # Use Option B: wrap as_completed with tqdm in the main process
            with tqdm(total=n_combos, desc="Overall Progress", disable=not progress_enabled) as pbar:
                for future in as_completed(future_map):
                    label = future_map[future]
                    try:
                        combo_name, m = future.result()
                        # Use tqdm.write instead of print to not break the bar
                        if progress_enabled:
                            tqdm.write(f"  ✓ Done: {combo_name}")
                        else:
                            print(f"  ✓ Done: {combo_name}")
                        results[combo_name] = m
                    except Exception as exc:
                        if progress_enabled:
                            tqdm.write(f"  ✗ FAILED: {label} — {exc}")
                        else:
                            print(f"  ✗ FAILED: {label} — {exc}")
                    finally:
                        pbar.update(1)

    # ── Combined comparison plot ──
    _plot_comparison(results, n_episodes)

    return results


def _plot_comparison(results, n_episodes):
    """Generate side-by-side comparison figures."""
    if not results:
        return

    os.makedirs('results', exist_ok=True)

    # Colour map for policies
    policy_colors = {
        'qlearning': '#e74c3c',
        'ppo': '#3498db',
        'sac': '#2ecc71',
    }
    agg_styles = {
        'none': '-',
        'fedavg': '--',
        'fedopt': ':',
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('FDRL Comparison: All Policies × Aggregation Strategies', fontsize=16, fontweight='bold')

    # 1 — Reward Convergence
    ax = axes[0, 0]
    for name, m in results.items():
        parts = name.split('_')
        pol, agg = parts[0], parts[1]
        color = policy_colors.get(pol, 'gray')
        ls = agg_styles.get(agg, '-')
        # Smooth with moving average
        rewards = np.array(m.episode_rewards)
        if len(rewards) > 10:
            smoothed = np.convolve(rewards, np.ones(10) / 10, mode='valid')
            ax.plot(smoothed, color=color, linestyle=ls, label=name, alpha=0.8)
        else:
            ax.plot(rewards, color=color, linestyle=ls, label=name, alpha=0.8)
    ax.set_title('Reward Convergence (MA-10)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 2 — Energy Cost
    ax = axes[0, 1]
    for name, m in results.items():
        parts = name.split('_')
        pol, agg = parts[0], parts[1]
        color = policy_colors.get(pol, 'gray')
        ls = agg_styles.get(agg, '-')
        if m.episode_costs:
            costs = np.array(m.episode_costs)
            if len(costs) > 10:
                smoothed = np.convolve(costs, np.ones(10) / 10, mode='valid')
                ax.plot(smoothed, color=color, linestyle=ls, label=name, alpha=0.8)
            else:
                ax.plot(costs, color=color, linestyle=ls, label=name, alpha=0.8)
    ax.set_title('Energy Cost (MA-10)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cost ($)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 3 — Satisfaction
    ax = axes[0, 2]
    for name, m in results.items():
        parts = name.split('_')
        pol, agg = parts[0], parts[1]
        color = policy_colors.get(pol, 'gray')
        ls = agg_styles.get(agg, '-')
        if m.satisfaction_history:
            ax.plot(m.satisfaction_history, color=color, linestyle=ls, label=name, alpha=0.8)
    ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.set_title('Client Satisfaction')
    ax.set_xlabel('Episode')
    ax.set_ylabel('SOC Ratio')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 4 — Grid Stability (bar chart)
    ax = axes[1, 0]
    names = list(results.keys())
    stabilities = [results[n].compute_stability_metric() for n in names]
    colors_bar = [policy_colors.get(n.split('_')[0], 'gray') for n in names]
    bars = ax.barh(range(len(names)), stabilities, color=colors_bar, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('σ_g (MW)')
    ax.set_title('Grid Stability (lower = better)')
    ax.grid(True, axis='x', alpha=0.3)

    # 5 — Test Reward (box plot)
    ax = axes[1, 1]
    test_data = []
    test_labels = []
    for name, m in results.items():
        if m.test_rewards:
            test_data.append(m.test_rewards)
            test_labels.append(name)
    if test_data:
        bp = ax.boxplot(test_data, labels=test_labels, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            pol = test_labels[i].split('_')[0]
            patch.set_facecolor(policy_colors.get(pol, 'gray'))
            patch.set_alpha(0.6)
        ax.set_title('Test Reward Distribution')
        ax.set_ylabel('Total Reward')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # 6 — Summary table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = []
    for name, m in results.items():
        tr = np.mean(m.episode_rewards[-5:]) if len(m.episode_rewards) >= 5 else 0
        te = np.mean(m.test_rewards) if m.test_rewards else 0
        sg = m.compute_stability_metric()
        sat = np.mean(m.satisfaction_history[-5:]) if len(m.satisfaction_history) >= 5 else 0
        table_data.append([name, f"{tr:.1f}", f"{te:.1f}", f"{sg:.4f}", f"{sat:.2%}"])
    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=['Combo', 'Train R', 'Test R', 'σ_g', 'Sat%'],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)
    ax.set_title('Summary', fontsize=12, pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = f"results/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n-> Comparison plot saved to {save_path}")
    plt.close(fig)
