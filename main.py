import logging
import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as nw
from tqdm import tqdm
from datetime import datetime
import argparse
import sys
import questionary

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s | %(levelname)s | %(message)s',
)
from utils.config_loader import get_config

# Ensure these imports are available from your project files
from env.EVClientEnv import EVClientEnv
from env.GridEnv import GridEnv
from utils.EvalMetrics import EvalMetrics
from utils.DataLoader import DataGenerator
from agents.QLearningAgent import QLearningAgent
from agents.PPOAgent import PPOAgent
from agents.SACAgent import SACAgent
from training.FederatedServer import FederatedServer
from training.EdgeAggregator import EdgeAggregator
from training.ComparisonPipeline import (
    run_single_experiment,
    run_comparison,
    run_methods_from_config,
)
from training.MultiSeedRunner import run_multiseed_pipeline
from training.DwellTimeStudy import run_dwell_time_study
from training.LoRANetworkStudy import run_lora_network_study
from training.StressTestStudy import run_stress_test_study, run_single_stress_run


def run_Q_learning_simulation(dev_mode=False):
    print("--- 1. Initialization of Federated EV Charging Simulation (Q-Learning) ---")
    if dev_mode:
        train_cfg = get_config('training_dev')
    else:
        train_cfg = get_config('training')
    # --- CONFIGURATION DICTIONARY ---
    env_cfg = get_config('env')
    simulation_config = {
        "type": "Q-Learning",
        "n_episodes": train_cfg.get('num_episodes', 300),
        "n_agents": train_cfg.get('num_agents', 10),
        "simulation_hours": train_cfg.get('simulation_hours', 24),
        "epsilon_init": 1.0,
        "epsilon_decay": 0.95,
        "epsilon_min": 0.05,
        "learning_rate": 0.1, 
        "gamma": 0.99,         
        "grid_type": train_cfg.get('grid_type', 'case33bw'),
        "ev_capacity": env_cfg.get('battery_capacity', 60.0),
        "ev_max_power": env_cfg.get('max_power', 11.0),
        "n_test_episodes": train_cfg.get('num_test_episodes', 10)
    }

    # Initialize Run Name
    run_name = input("Enter a name for this simulation run (for saving results): ")
    run_name = run_name.strip() + f"_QLearn_{datetime.now().strftime('%Y%m%d_%H%M%S')}" \
        if run_name else f"run_QLearn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize Metrics with Config
    metrics = EvalMetrics(run_name=run_name, config=simulation_config)
    
    # Initialize Core Systems
    grid = GridEnv(network_type=simulation_config['grid_type'])
    driver_profiles = DataGenerator.get_nhts_profile(simulation_config['n_agents'])
    agent_bus_map = {i: (i % 30) + 2 for i in range(simulation_config['n_agents'])}

    # Create Agents and Environments
    agents, envs = [], []
    for i in range(simulation_config['n_agents']):
        config = {
            'capacity': simulation_config['ev_capacity'],
            'max_power': simulation_config['ev_max_power'],
            'initial_soc': driver_profiles[i]['soc_init'],
            'soc_req': driver_profiles[i]['soc_req'],
            't_dep': driver_profiles[i]['duration'],
            'dt': 1.0
        }
        envs.append(EVClientEnv(config))
        agents.append(QLearningAgent(
            action_space_size=3, 
            state_bins=None, 
            epsilon=simulation_config['epsilon_init'],
            learning_rate=simulation_config['learning_rate'],
            gamma=simulation_config['gamma']
        ))

    # --- 2. TRAINING LOOP ---
    print(f"--- 2. Starting Training ({simulation_config['n_episodes']} Episodes) ---")

    for episode in tqdm(range(simulation_config['n_episodes']), desc="Training Episodes"):
        total_episode_reward = 0.0
        total_episode_cost = 0.0

        # Reset systems
        grid.reset()
        for i, env in enumerate(envs):
            env.soc = driver_profiles[i]['soc_init']
            env.current_step = 0

        # Reset signals
        active = [True] * simulation_config['n_agents']
        lambda_prev = 0.0
        volt_prev = 0.0
        prev_ev_total_mw = 0.0
        delta_ev_prev = 0.0

        for hour in range(simulation_config['simulation_hours']):
            # Global context
            price = DataGenerator.get_iso_ne_price(hour, mode='train')
            price_forecast = [DataGenerator.get_iso_ne_price((hour + h) % 24, mode='train') for h in range(5)]
            base_load_mw = np.random.normal(3.5, 0.2)

            # Agent decisions
            current_states = {}
            actions = {}
            grid_injections_mw = {}

            for i, agent in enumerate(agents):
                if not active[i]:
                    continue

                # IMPORTANT: realistic state inputs (previous step signals)
                s_t = envs[i].get_state(
                    grid_signal=lambda_prev,
                    voltage_dev=volt_prev,
                    price_forecast=price_forecast,
                    ev_total_mw=prev_ev_total_mw,
                    delta_ev_mw=delta_ev_prev
                )
                current_states[i] = s_t

                # Epsilon-greedy discrete action
                a_idx = agent.get_action(s_t, eval_mode=False)

                # Map action -> physical power (kW)
                # 0=idle, 1=half-charge, 2=full-charge (no V2G discharge)
                p_max = envs[i]._get_max_power(envs[i].soc)
                if a_idx == 0:
                    p_kw = 0.0
                elif a_idx == 1:
                    p_kw = p_max * 0.5
                else:
                    p_kw = p_max

                actions[i] = (a_idx, p_kw)

                # Prepare grid injections (MW)
                bus = agent_bus_map[i]
                grid_injections_mw[bus] = grid_injections_mw.get(bus, 0.0) + (p_kw / 1000.0)

            # Grid physics
            lambda_grid, grid_info = grid.step(grid_injections_mw, base_load_mw)

            # Compute aggregate EV load + ramp (for stability + next states)
            ev_total_mw = float(sum(grid_injections_mw.values()))
            delta_ev_mw = ev_total_mw - prev_ev_total_mw

            # Log stability using controllable signal (EV only)
            metrics.log_step(base_load_mw + ev_total_mw)

            # Learning update (only active EVs)
            for i, agent in enumerate(agents):
                if not active[i]:
                    continue

                a_idx, p_kw = actions[i]
                # print(f"price={price:.2f}, p_kw={p_kw:.2f}, lambda={lambda_grid:.4f}, volt_dev={grid_info['max_voltage'] - 1.0:.4f}")
                r_t, done, _, energy_cost = envs[i].step(
                    action_power=p_kw,
                    grid_signal=lambda_grid,
                    voltage_dev=grid_info['max_voltage'] - 1.0,
                    price_current=price
                )
                total_episode_cost += energy_cost

                s_next = envs[i].get_state(
                    grid_signal=lambda_grid,
                    voltage_dev=grid_info['max_voltage'] - 1.0,
                    price_forecast=price_forecast,
                    ev_total_mw=ev_total_mw,
                    delta_ev_mw=delta_ev_mw
                )

                agent.update(current_states[i], a_idx, r_t, s_next)
                total_episode_reward += r_t

                if done:
                    active[i] = False  # EV disconnected

            # Update previous-step broadcast signals for next hour
            lambda_prev = float(lambda_grid)
            volt_prev = float(grid_info['max_voltage'] - 1.0)
            prev_ev_total_mw = ev_total_mw
            delta_ev_prev = delta_ev_mw

        # End of episode: satisfaction metrics
        episode_satisfactions = []
        for i, env in enumerate(envs):
            req = driver_profiles[i]['soc_req']
            final = env.soc
            ratio = min(1.0, final / req) if req > 0 else 1.0
            episode_satisfactions.append(ratio)

        metrics.log_satisfaction(episode_satisfactions)
        metrics.log_episode(total_episode_reward, mode='train')
        metrics.log_cost(total_episode_cost)

        # Decay epsilon
        for agent in agents:
            agent.epsilon = max(simulation_config['epsilon_min'], agent.epsilon * simulation_config['epsilon_decay'])

        if (episode + 1) % 1000 == 0:
            print(f"  Ep {episode+1} | Reward: {total_episode_reward:.2f} | Cost: ${total_episode_cost:.2f} | Eps: {agents[0].epsilon:.2f}")

    # --- 3. TESTING PHASE ---
    print("--- 3. Starting Evaluation (Generalization Phase) ---")

    N_TEST_EPISODES = simulation_config['n_test_episodes']
    all_test_costs = []

    for test_ep in range(N_TEST_EPISODES):
        total_test_reward = 0.0
        total_test_cost = 0.0

        grid.reset()
        for env in envs:
            env.soc = 0.2
            env.current_step = 0

        active = [True] * simulation_config['n_agents']
        lambda_prev = 0.0
        volt_prev = 0.0
        prev_ev_total_mw = 0.0
        delta_ev_prev = 0.0

        for hour in range(simulation_config['simulation_hours']):
            price_test = DataGenerator.get_iso_ne_price(hour, mode='test')
            price_forecast_test = [DataGenerator.get_iso_ne_price((hour + h) % 24, mode='test') for h in range(5)]
            base_load_test = np.random.normal(3.8, 0.3)

            grid_injections_test = {}
            actions = {}

            # 1) decide actions only for active EVs
            for i, agent in enumerate(agents):
                if not active[i]:
                    continue

                s_t = envs[i].get_state(
                    grid_signal=lambda_prev,
                    voltage_dev=volt_prev,
                    price_forecast=price_forecast_test,
                    ev_total_mw=prev_ev_total_mw,
                    delta_ev_mw=delta_ev_prev
                )

                # eval_mode=True => no exploration
                a_idx = agent.get_action(s_t, eval_mode=True)

                p_max = envs[i]._get_max_power(envs[i].soc)
                # 0=idle, 1=half-charge, 2=full-charge (no V2G discharge)
                p_kw = (0.0 if a_idx == 0 else (p_max * 0.5 if a_idx == 1 else p_max))

                # cost
                total_test_cost += p_kw * 1.0 * price_test

                actions[i] = (a_idx, p_kw)

                bus = agent_bus_map[i]
                grid_injections_test[bus] = grid_injections_test.get(bus, 0.0) + (p_kw / 1000.0)

            # 2) grid step
            lambda_grid, grid_info = grid.step(grid_injections_test, base_load_test)

            ev_total_mw = float(sum(grid_injections_test.values()))
            delta_ev_mw = ev_total_mw - prev_ev_total_mw

            # 3) env step only for active EVs; deactivate on done
            for i, agent in enumerate(agents):
                if not active[i]:
                    continue

                _, p_kw = actions[i]

                r_t, done, _, _ = envs[i].step(
                    action_power=p_kw,
                    grid_signal=lambda_grid,
                    voltage_dev=grid_info['max_voltage'] - 1.0,
                    price_current=price_test
                )

                total_test_reward += r_t

                if done:
                    active[i] = False

            # 4) update prev signals
            lambda_prev = float(lambda_grid)
            volt_prev = float(grid_info['max_voltage'] - 1.0)
            prev_ev_total_mw = ev_total_mw
            delta_ev_prev = delta_ev_mw

        # Log this test episode reward (so boxplot has a distribution)
        metrics.log_episode(total_test_reward, mode='test')
        all_test_costs.append(total_test_cost)

        print(f"  TestEp {test_ep+1}/{N_TEST_EPISODES} | Reward: {total_test_reward:.2f} | Cost: ${total_test_cost:.2f}")

    print(f"\nTest Phase Avg Cost: ${np.mean(all_test_costs):.2f}  (over {N_TEST_EPISODES} episodes)")

    # --- 4. RESULTS & VISUALIZATION ---
    print("\n--- 4. Final Metrics ---")
    sigma_g = metrics.compute_stability_metric()
    print(f"-> Grid Stability (sigma_g): {sigma_g:.4f} MW")

    train_perf = np.mean(metrics.episode_rewards[-5:]) if len(metrics.episode_rewards) >= 5 else np.mean(metrics.episode_rewards)
    test_perf = np.mean(metrics.test_rewards[-N_TEST_EPISODES:]) if len(metrics.test_rewards) >= N_TEST_EPISODES else metrics.test_rewards[-1]
    print(f"-> Train Performance (Last 5 avg): {train_perf:.2f}")
    print(f"-> Test Performance (Avg over {N_TEST_EPISODES}): {test_perf:.2f}")

    metrics.plot_metrics()


# ─── collapsed from run_PPO_policy_simulation / run_SAC_simulation
#     / run_SAC_lora_simulation / run_PPO_lora_simulation ────────────────────
_CONTINUOUS_AGENT_DEFAULTS = {
    'sac': {'reward_weights': {'ramp': 2.0, 'track': 1.0, 'scale_mw': 0.10}},
    'ppo': {'reward_weights': {'ramp': 2.0, 'track': 1.0, 'scale_mw': 0.10}},
}


def _run_continuous_agent_simulation(policy: str, use_lora: bool = False, dev_mode: bool = False):
    """Standalone continuous-action simulation (SAC or PPO, with optional LoRA)."""
    policy    = policy.lower()
    train_cfg = get_config('training_dev' if dev_mode else 'training')
    env_cfg   = get_config('env')

    lora_tag = '+LoRA' if use_lora else ''
    run_type = f"{policy.upper()}{lora_tag}"
    simulation_config = {
        'type':             f'{run_type}-Continuous',
        'n_episodes':       train_cfg.get('num_episodes', 300),
        'n_agents':         train_cfg.get('num_agents', 10),
        'simulation_hours': train_cfg.get('simulation_hours', 24),
        'grid_type':        train_cfg.get('grid_type', 'case33bw'),
        'ev_capacity':      env_cfg.get('battery_capacity', 60.0),
        'ev_max_power':     env_cfg.get('max_power', 11.0),
        'n_test_episodes':  train_cfg.get('num_test_episodes', 10),
        'use_lora':         use_lora,
        **_CONTINUOUS_AGENT_DEFAULTS[policy],
    }

    print(f"--- 1. Initialization: {run_type} EV Charging ---")

    run_name = f"{run_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics  = EvalMetrics(run_name=run_name, config=simulation_config)
    grid     = GridEnv(network_type=simulation_config['grid_type'])
    driver_profiles = DataGenerator.get_nhts_profile(simulation_config['n_agents'])
    agent_bus_map   = {i: (i % 30) + 2 for i in range(simulation_config['n_agents'])}

    dummy_env = EVClientEnv({'capacity': 60.0, 'max_power': 11.0, 'initial_soc': 0.5,
                             'soc_req': 0.8, 't_dep': 10, 'dt': 1.0})
    input_dim = len(dummy_env.get_state(0.0, 0.0, [0.1] * 5))

    AgentClass = SACAgent if policy == 'sac' else PPOAgent
    n_ag  = simulation_config['n_agents']
    n_ep  = simulation_config['n_episodes']
    sim_h = simulation_config['simulation_hours']
    rw    = simulation_config['reward_weights']

    agents, envs = [], []
    for i in range(n_ag):
        envs.append(EVClientEnv({
            'capacity':    simulation_config['ev_capacity'],
            'max_power':   simulation_config['ev_max_power'],
            'initial_soc': driver_profiles[i]['soc_init'],
            'soc_req':     driver_profiles[i]['soc_req'],
            't_dep':       driver_profiles[i]['duration'],
            'dt':          1.0,
        }))
        agents.append(AgentClass(input_dim=input_dim, action_dim=1, use_lora=use_lora))

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"--- 2. Training ({n_ep} episodes) ---")
    for episode in tqdm(range(n_ep), desc=f"Training {run_type}"):
        total_episode_reward = total_episode_cost = 0.0
        grid.reset()
        active = [True] * n_ag
        lambda_prev = volt_prev = prev_ev_total_mw = 0.0

        for i, env in enumerate(envs):
            env.soc = driver_profiles[i]['soc_init']
            env.current_step = 0

        for hour in range(sim_h):
            price          = DataGenerator.get_iso_ne_price(hour, mode='train')
            price_forecast = [DataGenerator.get_iso_ne_price((hour + h) % 24, mode='train') for h in range(5)]
            base_load_mw   = np.random.normal(3.5, 0.2)

            p_target_kw, n_active = 0.0, 0
            for i, env in enumerate(envs):
                if not active[i]: continue
                n_active += 1
                soc_gap = max(0.0, env.soc_req - env.soc)
                e_gap   = (soc_gap * env.capacity) / env.eta
                t_left  = max(1, env.t_dep - env.current_step)
                p_target_kw += min(e_gap / t_left, env._get_max_power(env.soc))
            p_target_mw = p_target_kw / 1000.0

            current_states, actions, grid_injections_mw = {}, {}, {}
            for i, agent in enumerate(agents):
                if not active[i]: continue
                s_t = envs[i].get_state(lambda_prev, volt_prev, price_forecast)
                current_states[i] = s_t
                raw_action = agent.get_action(s_t, eval_mode=False)
                p_kw = raw_action * envs[i]._get_max_power(envs[i].soc)
                actions[i] = (raw_action, p_kw)
                bus = agent_bus_map[i]
                grid_injections_mw[bus] = grid_injections_mw.get(bus, 0.0) + (p_kw / 1000.0)

            lambda_grid, grid_info = grid.step(grid_injections_mw, base_load_mw)
            ev_total_mw  = float(sum(grid_injections_mw.values()))
            delta_ev_mw  = ev_total_mw - prev_ev_total_mw
            r_ramp       = -rw['ramp']  * (delta_ev_mw  / rw['scale_mw']) ** 2
            r_track      = -rw['track'] * ((ev_total_mw - p_target_mw) / rw['scale_mw']) ** 2
            shared_penalty = (r_ramp + r_track) / max(1, n_active)
            prev_ev_total_mw = ev_total_mw
            lambda_prev      = float(lambda_grid)
            volt_prev        = float(grid_info['max_voltage'] - 1.0)
            metrics.log_step(base_load_mw + ev_total_mw)

            for i, agent in enumerate(agents):
                if not active[i]: continue
                raw_action, p_kw = actions[i]
                r_t, done, _, energy_cost = envs[i].step(
                    p_kw, lambda_grid, grid_info['max_voltage'] - 1.0, price)
                total_episode_cost  += energy_cost
                r_t += shared_penalty
                s_next = envs[i].get_state(lambda_grid, grid_info['max_voltage'] - 1.0, price_forecast)
                agent.update(current_states[i], raw_action, r_t, s_next, done=done)
                total_episode_reward += r_t
                if done:
                    active[i] = False

        sats = [min(1.0, envs[i].soc / driver_profiles[i]['soc_req'])
                if driver_profiles[i]['soc_req'] > 0 else 1.0
                for i in range(n_ag)]
        metrics.log_satisfaction(sats)
        metrics.log_episode(total_episode_reward, mode='train')
        metrics.log_cost(total_episode_cost)

    # ── Test phase ────────────────────────────────────────────────────────────
    print(f"--- 3. Evaluation ({run_type}) ---")
    N_TEST = simulation_config['n_test_episodes']
    for test_ep in range(N_TEST):
        total_test_reward = 0.0
        grid.reset()
        for env in envs:
            env.soc = 0.2
            env.current_step = 0
        active = [True] * n_ag
        lambda_prev = volt_prev = prev_ev_total_mw = 0.0

        for hour in range(sim_h):
            price_test = DataGenerator.get_iso_ne_price(hour, mode='test')
            pf_test    = [DataGenerator.get_iso_ne_price((hour + h) % 24, mode='test') for h in range(5)]
            base_load_test = np.random.normal(3.8, 0.3)
            grid_inj, actions = {}, {}
            for i, agent in enumerate(agents):
                if not active[i]: continue
                s_t = envs[i].get_state(lambda_prev, volt_prev, pf_test)
                raw = agent.get_action(s_t, eval_mode=True)
                p_kw = raw * envs[i]._get_max_power(envs[i].soc)
                actions[i] = (raw, p_kw)
                bus = agent_bus_map[i]
                grid_inj[bus] = grid_inj.get(bus, 0.0) + (p_kw / 1000.0)
            l_g, g_i = grid.step(grid_inj, base_load_test)
            for i, agent in enumerate(agents):
                if not active[i]: continue
                _, p_kw = actions[i]
                r_t, done, _, _ = envs[i].step(p_kw, l_g, g_i['max_voltage'] - 1.0, price_test)
                total_test_reward += r_t
                if done: active[i] = False
            prev_ev_total_mw = float(sum(grid_inj.values()))
            lambda_prev = float(l_g)
            volt_prev   = float(g_i['max_voltage'] - 1.0)
        metrics.log_episode(total_test_reward, mode='test')
        print(f"  TestEp {test_ep+1}/{N_TEST} | Reward: {total_test_reward:.2f}")

    print(f"\nGrid Stability (sigma_g): {metrics.compute_stability_metric():.4f} MW")
    metrics.plot_metrics()


def _run_PPO_simulation(dev_mode=False):
    _run_continuous_agent_simulation('ppo', use_lora=False, dev_mode=dev_mode)


def _run_SAC_simulation(dev_mode=False):
    _run_continuous_agent_simulation('sac', use_lora=False, dev_mode=dev_mode)


def _run_SAC_lora_simulation(dev_mode=False):
    _run_continuous_agent_simulation('sac', use_lora=True, dev_mode=dev_mode)


def _run_PPO_lora_simulation(dev_mode=False):
    _run_continuous_agent_simulation('ppo', use_lora=True, dev_mode=dev_mode)


def run_federated_lora_simulation(dev_mode=False):
    """Federated training with LoRA enabled — interactive policy/aggregation selection."""
    print("[LoRA mode: ENABLED for Federated Training]")
    policy_choice = questionary.select(
        "Select RL policy (LoRA-enabled):",
        choices=[
            questionary.Choice("PPO + LoRA", value='ppo'),
            questionary.Choice("SAC + LoRA", value='sac'),
        ]
    ).ask()
    if policy_choice is None:
        sys.exit(0)

    agg_choice = questionary.select(
        "Select aggregation strategy:",
        choices=[
            questionary.Choice("FedAvg", value='fedavg'),
            questionary.Choice("FedOpt (server momentum)", value='fedopt'),
        ]
    ).ask()
    if agg_choice is None:
        sys.exit(0)

    print(f"\n>>> Running Federated + LoRA: {policy_choice} + {agg_choice}")
    m = run_single_experiment(
        policy=policy_choice,
        aggregation=agg_choice,
        verbose=True,
        dev_mode=dev_mode,
        use_lora=True,  # ← LoRA enabled for federated training
    )
    m.plot_metrics()


def run_federated_simulation(dev_mode=False):
    """Federated training with interactive policy/aggregation selection."""
    policy_choice = questionary.select(
        "Select RL policy:",
        choices=[
            questionary.Choice("PPO", value='ppo'),
            questionary.Choice("SAC", value='sac'),
            questionary.Choice("Q-Learning", value='qlearning'),
        ]
    ).ask()
    if policy_choice is None:
        sys.exit(0)

    agg_choice = questionary.select(
        "Select aggregation strategy:",
        choices=[
            questionary.Choice("FedAvg", value='fedavg'),
            questionary.Choice("FedOpt (server momentum)", value='fedopt'),
        ]
    ).ask()
    if agg_choice is None:
        sys.exit(0)

    print(f"\n>>> Running Federated: {policy_choice} + {agg_choice}")
    m = run_single_experiment(
        policy=policy_choice,
        aggregation=agg_choice,
        verbose=True,
        dev_mode=dev_mode,
    )
    m.plot_metrics()


def run_federated_swift_simulation(dev_mode=False, use_lora=False):
    """Federated training with SWIFT client selection."""
    from training.SWIFTScheduler import SWIFTScheduler
    from utils.config_loader import get_config as _get_config

    lora_label = ' + LoRA' if use_lora else ''
    print(f"[SWIFT mode: ENABLED{lora_label} for Federated Training]")

    policy_choice = questionary.select(
        "Select RL policy:",
        choices=[
            questionary.Choice("PPO", value='ppo'),
            questionary.Choice("SAC", value='sac'),
            questionary.Choice("Q-Learning", value='qlearning'),
        ]
    ).ask()
    if policy_choice is None:
        sys.exit(0)

    agg_choice = questionary.select(
        "Select aggregation strategy:",
        choices=[
            questionary.Choice("FedAvg", value='fedavg'),
            questionary.Choice("FedOpt (server momentum)", value='fedopt'),
        ]
    ).ask()
    if agg_choice is None:
        sys.exit(0)

    # Load and display SWIFT config for user awareness
    swift_cfg = _get_config('swift')
    print(f"\n>>> SWIFT config: fraction={swift_cfg['fraction']}, "
          f"min_stay_hours={swift_cfg['min_stay_hours']}, "
          f"force_select_after={swift_cfg['force_select_after']}")
    print(f">>> LoRA: {'enabled' if use_lora else 'disabled'}")
    print(f">>> Running Federated + SWIFT: {policy_choice} + {agg_choice}\n")

    m = run_single_experiment(
        policy=policy_choice,
        aggregation=agg_choice,
        verbose=True,
        dev_mode=dev_mode,
        use_lora=use_lora,
        use_swift=True,
    )
    m.plot_metrics()


# def main():
#     parser = argparse.ArgumentParser(description="FDRL EV Charging Simulation")
#     parser.add_argument(
#         'mode',
#         type=int,
#         nargs='?',
#         help="1=Training, 2=Development"
#     )

#     args = parser.parse_args()

#     if args.mode is None:
#         choice = questionary.select(
#             "Select simulation mode:",
#             choices=[
#                 questionary.Choice("PPO Policy Simulation", value=1),
#                 questionary.Choice("Q-Learning Simulation", value=2),
#                 questionary.Choice("SAC Policy Simulation", value=3),
#                 questionary.Choice("Federated Training (select policy + aggregation)", value=4),
#                 questionary.Choice("Full Comparison Pipeline (all combos)", value=5),
#             ],
#             use_arrow_keys=True
#         ).ask()

#         if choice is None:
#             sys.exit(0)
#         args.mode = choice

#     if args.mode == 1:
#         run_PPO_policy_simulation()
#     elif args.mode == 2:
#         run_Q_learning_simulation()
#     elif args.mode == 3:
#         run_SAC_simulation()
#     elif args.mode == 4:
#         run_federated_simulation()
#     elif args.mode == 5:
#         run_comparison()
#     else:
#         print(f"Error: {args.mode} is not a valid option. Use 1-5.")
#         sys.exit(1)


import argparse
import sys
import questionary


# ─────────────────────────────────────────────────────────────────────────────
# New baseline runners (Category 1 / 2 / 3)
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_suite(dev_mode=False):
    """Run ALL baselines listed in training.yaml → methods section."""
    print("\n>>> Baseline Suite: reading methods from configs/training.yaml ...")
    results = run_methods_from_config(dev_mode=dev_mode, verbose=True)
    print(f"\n>>> Completed {len(results)} baselines.")


def run_heuristic_simulation(dev_mode=False):
    """Interactive heuristic baseline selection and evaluation."""
    choice = questionary.select(
        "Select heuristic baseline:",
        choices=[
            questionary.Choice("Random (uniform random action)", value='random'),
            questionary.Choice("Greedy (charge when SOC < req)", value='greedy'),
            questionary.Choice("EDF — Earliest Deadline First", value='edf'),
            questionary.Choice("Price-Aware (threshold-based)", value='price_aware'),
            questionary.Choice("Simple MPC (1-step lookahead)", value='simple_mpc'),
        ]
    ).ask()
    if choice is None:
        sys.exit(0)

    print(f"\n>>> Running heuristic baseline: {choice}")
    m = run_single_experiment(
        policy=choice,
        aggregation='none',
        verbose=True,
        dev_mode=dev_mode,
    )
    m.plot_metrics()


def run_federated_variant_simulation(dev_mode=False):
    """Interactive federated aggregation variant selection (FedProx / FedAvgM / FedAdam)."""
    agg_choice = questionary.select(
        "Select federated aggregation variant:",
        choices=[
            questionary.Choice("FedProx (proximal regularisation, μ=0.01)", value='fedprox'),
            questionary.Choice("FedAvgM (server-side momentum, β=0.9)", value='fedavgm'),
            questionary.Choice("FedAdam (server-side Adam, lr=0.01)", value='fedadam'),
        ]
    ).ask()
    if agg_choice is None:
        sys.exit(0)

    lora_choice = questionary.confirm("Enable LoRA compression?", default=False).ask()
    swift_choice = questionary.confirm("Enable SWIFT client selection?", default=False).ask()

    print(f"\n>>> Running SAC + {agg_choice}"
          f"{'  +LoRA' if lora_choice else ''}"
          f"{'  +SWIFT' if swift_choice else ''}")

    m = run_single_experiment(
        policy='sac',
        aggregation=agg_choice,
        verbose=True,
        dev_mode=dev_mode,
        use_lora=lora_choice,
        use_swift=swift_choice,
        mu_fedprox=0.01,
        beta_momentum=0.9,
        adam_lr=0.01,
    )
    m.plot_metrics()


def run_sac_local_simulation(dev_mode=False):
    """SAC without any federated aggregation — each agent trains independently."""
    print("\n>>> Running SAC Local-Only (no federation) ...")
    m = run_single_experiment(
        policy='sac',
        aggregation='none',
        verbose=True,
        dev_mode=dev_mode,
    )
    m.plot_metrics()


def run_sac_centralized_simulation(dev_mode=False):
    """SAC Centralized Oracle — all agents share one network and replay buffer."""
    print("\n>>> Running SAC Centralized Oracle (shared network) ...")
    m = run_single_experiment(
        policy='sac',
        aggregation='none',
        centralized=True,
        verbose=True,
        dev_mode=dev_mode,
    )
    m.plot_metrics()


def run_dwell_study(dev_mode=False):
    """SWIFT dwell-time analysis — FedAvg vs SWIFT vs HFDRL across 1h/2h/4h/6h/8h windows."""
    from training.DwellTimeStudy import run_single_triple
    print("\n>>> SWIFT Dwell-Time Study")

    run_mode = questionary.select(
        "Run mode:",
        choices=[
            questionary.Choice("Full study  (all methods × dwells × seeds)", value='full'),
            questionary.Choice("Single run  (pick one method, dwell, seed)", value='single'),
        ]
    ).ask()
    if run_mode is None:
        sys.exit(0)

    if run_mode == 'single':
        method_choice = questionary.select(
            "Method:",
            choices=[
                questionary.Choice("FedAvg-SAC  (baseline federated)", value='FedAvg-SAC'),
                questionary.Choice("SWIFT-SAC   (smart selection)",     value='SWIFT-SAC'),
                questionary.Choice("HFDRL       (SWIFT + LoRA)",        value='HFDRL'),
            ]
        ).ask()
        if method_choice is None:
            sys.exit(0)

        dwell_choice = questionary.select(
            "Dwell-time scenario:",
            choices=[
                questionary.Choice("1h  (very short window)", value=1),
                questionary.Choice("2h",                      value=2),
                questionary.Choice("4h",                      value=4),
                questionary.Choice("6h",                      value=6),
                questionary.Choice("8h  (long window)",       value=8),
            ]
        ).ask()
        if dwell_choice is None:
            sys.exit(0)

        seed_choice = questionary.select(
            "Seed:",
            choices=[questionary.Choice(str(s), value=s)
                     for s in [0, 1, 2, 3, 4, 42, 123, 456, 789, 999]]
        ).ask()
        if seed_choice is None:
            sys.exit(0)

        print(f"\n>>> method={method_choice}  dwell={dwell_choice}h  seed={seed_choice}"
              f"  mode={'dev' if dev_mode else 'full'}")
        run_single_triple(
            method_name=method_choice,
            dwell_hours=dwell_choice,
            seed=seed_choice,
            dev_mode=dev_mode,
        )
        return

    # ── Full study ────────────────────────────────────────────────────────────
    from utils.config_loader import get_config as _gc
    cfg = _gc('dwell_time_study')

    n_seeds_choice = questionary.select(
        "Number of seeds:",
        choices=[
            questionary.Choice("5  seeds  (minimum)", value=5),
            questionary.Choice("10 seeds  (AAAI preferred)", value=10),
        ]
    ).ask()
    if n_seeds_choice is None:
        sys.exit(0)

    seed_sets = {5: [0, 1, 2, 3, 4], 10: [0, 1, 2, 3, 4, 42, 123, 456, 789, 999]}
    seeds = seed_sets[n_seeds_choice]

    dwell_choice = questionary.select(
        "Dwell-time scenarios to run:",
        choices=[
            questionary.Choice("All five  (1h, 2h, 4h, 6h, 8h)", value=[1, 2, 4, 6, 8]),
            questionary.Choice("Short only (1h, 2h)",              value=[1, 2]),
            questionary.Choice("Quick test (2h, 6h)",              value=[2, 6]),
            questionary.Choice("1h only",                          value=[1]),
            questionary.Choice("2h only",                          value=[2]),
            questionary.Choice("4h only",                          value=[4]),
            questionary.Choice("6h only",                          value=[6]),
            questionary.Choice("8h only",                          value=[8]),
        ]
    ).ask()
    if dwell_choice is None:
        sys.exit(0)

    method_filter = questionary.checkbox(
        "Limit to specific methods? (space to toggle, enter to confirm all):",
        choices=[
            questionary.Choice("FedAvg-SAC", value='FedAvg-SAC', checked=True),
            questionary.Choice("SWIFT-SAC",  value='SWIFT-SAC',  checked=True),
            questionary.Choice("HFDRL",      value='HFDRL',      checked=True),
        ]
    ).ask()
    if not method_filter:
        sys.exit(0)

    print(f"\n>>> Seeds: {seeds}  |  Dwell: {dwell_choice}h  |  Methods: {method_filter}"
          f"  |  Mode: {'dev' if dev_mode else 'full'}")
    run_dwell_time_study(seeds=seeds, dwell_hours_list=dwell_choice,
                         method_filter=method_filter, dev_mode=dev_mode)


def run_lora_network_study_menu(dev_mode=False):
    """LoRA Network Constraints Study — HFDRL vs HFDRL+LoRA across BW scenarios."""
    from training.LoRANetworkStudy import run_single_pair, METHOD_BY_NAME
    print("\n>>> LoRA Network Constraints Study")

    run_mode = questionary.select(
        "Run mode:",
        choices=[
            questionary.Choice("Full study  (both methods × seeds)", value='full'),
            questionary.Choice("Group run   (select method subset)", value='group'),
            questionary.Choice("Single pair (pick one method + seed)", value='single'),
        ]
    ).ask()
    if run_mode is None:
        sys.exit(0)

    if run_mode == 'single':
        method_choice = questionary.select(
            "Method:",
            choices=[
                questionary.Choice("HFDRL (no LoRA)  — full model FL",    value='HFDRL (no LoRA)'),
                questionary.Choice("HFDRL + LoRA     — LoRA-compressed FL", value='HFDRL + LoRA'),
            ]
        ).ask()
        if method_choice is None:
            sys.exit(0)

        seed_choice = questionary.select(
            "Seed:",
            choices=[questionary.Choice(str(s), value=s)
                     for s in [0, 1, 2, 3, 4, 42, 123, 456, 789, 999]]
        ).ask()
        if seed_choice is None:
            sys.exit(0)

        print(f"\n>>> method={method_choice}  seed={seed_choice}"
              f"  mode={'dev' if dev_mode else 'full'}")
        run_single_pair(method_name=method_choice, seed=seed_choice, dev_mode=dev_mode)
        return

    if run_mode == 'group':
        group_choice = questionary.select(
            "Method group:",
            choices=[
                questionary.Choice("Both methods (full)", value='full'),
                questionary.Choice("HFDRL + LoRA only",   value='lora_only'),
                questionary.Choice("HFDRL (no LoRA) only", value='nolora_only'),
            ]
        ).ask()
        if group_choice is None:
            sys.exit(0)
    else:
        group_choice = None   # full study

    n_seeds_choice = questionary.select(
        "Number of seeds:",
        choices=[
            questionary.Choice("5  seeds  (minimum)", value=5),
            questionary.Choice("10 seeds  (AAAI preferred)", value=10),
        ]
    ).ask()
    if n_seeds_choice is None:
        sys.exit(0)

    seed_sets = {5: [0, 1, 2, 3, 4], 10: [0, 1, 2, 3, 4, 42, 123, 456, 789, 999]}
    seeds = seed_sets[n_seeds_choice]

    print(f"\n>>> Seeds: {seeds}  |  Group: {group_choice or 'full'}"
          f"  |  Mode: {'dev' if dev_mode else 'full'}")
    run_lora_network_study(seeds=seeds, method_filter=group_choice, dev_mode=dev_mode)


def run_multiseed_evaluation(dev_mode=False):
    """Multi-seed statistical evaluation pipeline for AAAI publication."""
    print("\n>>> Multi-Seed Statistical Evaluation")

    n_seeds_choice = questionary.select(
        "Number of seeds:",
        choices=[
            questionary.Choice("5  seeds  (minimum for significance)", value=5),
            questionary.Choice("10 seeds  (preferred for AAAI)", value=10),
        ]
    ).ask()
    if n_seeds_choice is None:
        sys.exit(0)

    seed_sets = {
        5:  [0, 1, 2, 3, 4],
        10: [0, 1, 2, 3, 4, 42, 123, 456, 789, 999],
    }
    seeds = seed_sets[n_seeds_choice]

    method_filter = None
    filter_choice = questionary.confirm(
        "Run only the HFDRL method + key baselines? (faster; full suite otherwise)",
        default=False,
    ).ask()
    if filter_choice:
        method_filter = [
            'SAC HFedAvg SWIFT LoRA',
            'SAC Local-Only',
            'SAC Centralized Oracle',
            'Random',
            'Greedy',
            'Price-Aware',
        ]

    print(f"\n>>> Seeds: {seeds}")
    print(f">>> Methods: {'filtered' if method_filter else 'all'}")
    run_multiseed_pipeline(seeds=seeds, dev_mode=dev_mode, method_filter=method_filter)


def run_stress_test_menu(dev_mode=False):
    """Robustness stress-test study — Forecast Error & Non-IID Data (AAAI)."""
    print("\n>>> Robustness Stress Test Study")

    sub_choice = questionary.select(
        "Sub-study to run:",
        choices=[
            questionary.Choice("Both  (Forecast Error + Non-IID)",  value='both'),
            questionary.Choice("Forecast Error only",               value='forecast_error'),
            questionary.Choice("Non-IID Data only",                 value='non_iid'),
            questionary.Choice("Single run  (pick one combination)", value='single'),
        ]
    ).ask()
    if sub_choice is None:
        sys.exit(0)

    # ── Single-run path ────────────────────────────────────────────────────────
    if sub_choice == 'single':
        sub_single = questionary.select(
            "Sub-study for single run:",
            choices=[
                questionary.Choice("Forecast Error", value='forecast_error'),
                questionary.Choice("Non-IID Data",   value='non_iid'),
            ]
        ).ask()
        if sub_single is None:
            sys.exit(0)

        if sub_single == 'forecast_error':
            scenario_choice = questionary.select(
                "Noise level σ ($/kWh):",
                choices=[
                    questionary.Choice("0.00  (no noise — baseline)",          value=0.0),
                    questionary.Choice("0.02  (~10% relative noise — mild)",    value=0.02),
                    questionary.Choice("0.05  (~25% relative noise — moderate)", value=0.05),
                    questionary.Choice("0.10  (~50% relative noise — severe)",  value=0.10),
                    questionary.Choice("0.20  (~100% relative noise — extreme)", value=0.20),
                ]
            ).ask()
        else:
            scenario_choice = questionary.select(
                "Dirichlet α (data heterogeneity):",
                choices=[
                    questionary.Choice("1000  (≈ IID — homogeneous)",           value=1000.0),
                    questionary.Choice("10    (mild heterogeneity)",             value=10.0),
                    questionary.Choice("1.0   (moderate heterogeneity)",         value=1.0),
                    questionary.Choice("0.5   (high heterogeneity)",             value=0.5),
                    questionary.Choice("0.1   (extreme heterogeneity)",          value=0.1),
                ]
            ).ask()
        if scenario_choice is None:
            sys.exit(0)

        method_choice = questionary.select(
            "Method:",
            choices=[
                questionary.Choice("FedAvg-SAC  (baseline federated)", value='FedAvg-SAC'),
                questionary.Choice("SWIFT-SAC   (smart selection)",     value='SWIFT-SAC'),
                questionary.Choice("HFDRL       (SWIFT + LoRA)",        value='HFDRL'),
            ]
        ).ask()
        if method_choice is None:
            sys.exit(0)

        archetype_choice = 'nhts'
        if sub_single == 'non_iid':
            archetype_choice = questionary.select(
                "Driver archetype table:",
                choices=[
                    questionary.Choice("NHTS 2017  (default, Bureau of Transportation Statistics)",
                                       value='nhts'),
                    questionary.Choice("ACN-Data   (Lee et al., 2019, Caltech)",
                                       value='acn'),
                ]
            ).ask()
            if archetype_choice is None:
                sys.exit(0)

        seed_choice = questionary.select(
            "Seed:",
            choices=[questionary.Choice(str(s), value=s) for s in [0, 1, 2, 42, 123]]
        ).ask()
        if seed_choice is None:
            sys.exit(0)

        print(f"\n>>> sub={sub_single}  scenario={scenario_choice}  "
              f"method={method_choice}  seed={seed_choice}  "
              f"archetype={archetype_choice}  mode={'dev' if dev_mode else 'full'}")
        run_single_stress_run(
            sub_study=sub_single,
            scenario_value=scenario_choice,
            method_name=method_choice,
            seed=seed_choice,
            dev_mode=dev_mode,
            archetype_set=archetype_choice,
        )
        return

    # ── Full / sub-study path ─────────────────────────────────────────────────
    archetype_choice = 'nhts'
    if sub_choice in ('non_iid', 'both'):
        archetype_choice = questionary.select(
            "Driver archetype table for Non-IID sub-study:",
            choices=[
                questionary.Choice("NHTS 2017  (default, Bureau of Transportation Statistics)",
                                   value='nhts'),
                questionary.Choice("ACN-Data   (Lee et al., 2019, Caltech)",
                                   value='acn'),
            ]
        ).ask()
        if archetype_choice is None:
            sys.exit(0)

    n_seeds_choice = questionary.select(
        "Number of seeds:",
        choices=[
            questionary.Choice("5  seeds  (recommended for ablations)", value=5),
            questionary.Choice("10 seeds  (full AAAI rigor)",           value=10),
        ]
    ).ask()
    if n_seeds_choice is None:
        sys.exit(0)

    seed_sets = {5: [0, 1, 2, 42, 123], 10: [0, 1, 2, 3, 4, 42, 123, 456, 789, 999]}
    seeds = seed_sets[n_seeds_choice]

    method_filter = questionary.checkbox(
        "Limit to specific methods? (space to toggle, enter to confirm all):",
        choices=[
            questionary.Choice("FedAvg-SAC", value='FedAvg-SAC', checked=True),
            questionary.Choice("SWIFT-SAC",  value='SWIFT-SAC',  checked=True),
            questionary.Choice("HFDRL",      value='HFDRL',      checked=True),
        ]
    ).ask()
    if not method_filter:
        sys.exit(0)

    print(f"\n>>> Sub-study: {sub_choice}  |  Seeds: {seeds}  |  Methods: {method_filter}"
          f"  |  Archetypes: {archetype_choice}  |  Mode: {'dev' if dev_mode else 'full'}")
    run_stress_test_study(
        sub_study=sub_choice,
        seeds=seeds,
        method_filter=method_filter,
        dev_mode=dev_mode,
        archetype_set=archetype_choice,
    )


def _build_menu_choices(dev: bool) -> list:
    d = " (dev)" if dev else ""
    return [
        questionary.Choice(f"PPO Policy Training{d}",                                      value=1),
        questionary.Choice(f"Q-Learning Training{d}",                                      value=2),
        questionary.Choice(f"SAC Policy Training{d}",                                      value=3),
        questionary.Choice(f"Federated Training — policy + aggregation{d}",                value=4),
        questionary.Choice(f"Full Comparison Pipeline — all combos{d}",                    value=5),
        questionary.Choice(f"SAC + LoRA Training{d}",                                      value=6),
        questionary.Choice(f"PPO + LoRA Training{d}",                                      value=7),
        questionary.Choice(f"Federated Training + LoRA{d}",                                value=8),
        questionary.Choice(f"Federated + SWIFT scheduling{d}",                             value=9),
        questionary.Choice(f"Federated + SWIFT + LoRA{d}",                                 value=10),
        questionary.Choice("── Baselines ──────────────────────────",                      value=-1),
        questionary.Choice(f"Baseline Suite — all methods from config{d}",                 value=11),
        questionary.Choice(f"Heuristic Baseline — Random/Greedy/EDF/Price/MPC{d}",         value=12),
        questionary.Choice(f"Federated Variant — FedProx/FedAvgM/FedAdam{d}",              value=13),
        questionary.Choice(f"SAC Local-Only — no federation{d}",                           value=14),
        questionary.Choice(f"SAC Centralized Oracle{d}",                                   value=15),
        questionary.Choice("── Statistics ─────────────────────────",                      value=-2),
        questionary.Choice(f"Multi-Seed Statistical Evaluation (AAAI){d}",                 value=16),
        questionary.Choice(f"SWIFT Dwell-Time Study (AAAI){d}",                            value=17),
        questionary.Choice(f"LoRA Network Constraints Study (AAAI){d}",                    value=18),
        questionary.Choice(f"Robustness Stress Tests — Forecast Error + Non-IID (AAAI){d}", value=19),
    ]


def _dispatch(simulation, dev_mode: bool, args) -> None:
    """Execute the selected simulation in the appropriate mode."""
    if dev_mode:
        DataGenerator.dev_mode = True
    d = dev_mode
    if simulation == 1:
        _run_PPO_simulation(d)
    elif simulation == 2:
        run_Q_learning_simulation(d)
    elif simulation == 3:
        _run_SAC_simulation(d)
    elif simulation == 4:
        run_federated_simulation(d)
    elif simulation == 5:
        run_comparison(dev_mode=d)
    elif simulation == 6:
        _run_SAC_lora_simulation(d)
    elif simulation == 7:
        _run_PPO_lora_simulation(d)
    elif simulation == 8:
        run_federated_lora_simulation(d)
    elif simulation == 9:
        run_federated_swift_simulation(d)
    elif simulation == 10:
        run_federated_swift_simulation(dev_mode=d, use_lora=True)
    elif simulation == 11:
        run_baseline_suite(d)
    elif simulation == 12:
        run_heuristic_simulation(d)
    elif simulation == 13:
        run_federated_variant_simulation(d)
    elif simulation == 14:
        run_sac_local_simulation(d)
    elif simulation == 15:
        run_sac_centralized_simulation(d)
    elif simulation == 16:
        run_multiseed_evaluation(d)
    elif simulation == 17:
        run_dwell_study(d)
    elif simulation == 18:
        run_lora_network_study_menu(d)
    elif simulation in (19, 'stressTest'):
        _stress_args = (args.sub_study, args.scenario, args.method, args.seed)
        if all(a is not None for a in _stress_args):
            run_single_stress_run(
                sub_study=args.sub_study,
                scenario_value=args.scenario,
                method_name=args.method,
                seed=args.seed,
                dev_mode=d,
                archetype_set=args.archetype,
            )
        else:
            run_stress_test_menu(dev_mode=d)
    elif simulation in (-1, -2):
        print("Please select a valid simulation (not the separator).")
        sys.exit(1)
    else:
        mode_str = "Development" if dev_mode else "Training"
        print(f"Error: Simulation {simulation} is not valid for {mode_str} mode.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="FDRL EV Charging Simulation")
    parser.add_argument('mode', type=int, nargs='?', help="1=Training, 2=Development")
    parser.add_argument('--simulation', type=str, nargs='?',
                        help="Simulation type: integer (1-19) or named alias e.g. 'stressTest'")
    parser.add_argument('--sub-study', dest='sub_study',
                        choices=['forecast_error', 'non_iid'],
                        help="Stress-test sub-study type")
    parser.add_argument('--scenario',  type=float,
                        help="Scenario value: σ for forecast_error or α for non_iid")
    parser.add_argument('--method',    choices=['FedAvg-SAC', 'SWIFT-SAC', 'HFDRL'],
                        help="Method to evaluate")
    parser.add_argument('--seed',      type=int, help="Random seed for the stress-test run")
    parser.add_argument('--archetype', choices=['nhts', 'acn'], default='nhts',
                        help="Driver archetype table for non_iid (default: nhts)")

    args = parser.parse_args()

    if args.simulation is not None:
        try:
            args.simulation = int(args.simulation)
        except ValueError:
            pass  # keep as string alias, e.g. 'stressTest'

    # ── Step 1: mode selection ────────────────────────────────────────────────
    if args.mode is None:
        mode_choice = questionary.select(
            "Select operation mode:",
            choices=[
                questionary.Choice("Training Mode",    value=1),
                questionary.Choice("Development Mode", value=2),
            ],
            use_arrow_keys=True,
        ).ask()
        if mode_choice is None:
            sys.exit(0)
        args.mode = mode_choice

    if args.mode not in [1, 2]:
        print(f"Error: Mode {args.mode} is invalid. Use 1 (Training) or 2 (Development).")
        sys.exit(1)

    dev_mode  = (args.mode == 2)
    mode_name = "Development" if dev_mode else "Training"
    print(f"\n✓ Mode selected: {mode_name}\n")

    # ── Step 2: simulation selection ──────────────────────────────────────────
    if args.simulation is None:
        prompt = "Select development simulation:" if dev_mode else "Select training simulation:"
        sim_choice = questionary.select(
            prompt, choices=_build_menu_choices(dev_mode), use_arrow_keys=True,
        ).ask()
        if sim_choice is None:
            sys.exit(0)
        args.simulation = sim_choice

    # ── Step 3: dispatch ──────────────────────────────────────────────────────
    _dispatch(args.simulation, dev_mode, args)

    print("\n✓ Simulation completed successfully!")


if __name__ == "__main__":
    main()
