import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as nw
from tqdm import tqdm
from datetime import datetime
import argparse
import sys
import questionary
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
from training.ComparisonPipeline import run_single_experiment, run_comparison

def run_Q_learning_simulation():
    print("--- 1. Initialization of Federated EV Charging Simulation (Q-Learning) ---")

    # --- CONFIGURATION DICTIONARY ---
    train_cfg = get_config('training')
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
                p_max = envs[i]._get_max_power(envs[i].soc)
                if a_idx == 0:
                    p_kw = -p_max
                elif a_idx == 1:
                    p_kw = 0.0
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
            metrics.log_step(ev_total_mw)

            # Learning update (only active EVs)
            for i, agent in enumerate(agents):
                if not active[i]:
                    continue

                a_idx, p_kw = actions[i]

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

        if (episode + 1) % 10 == 0:
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
                p_kw = (-p_max if a_idx == 0 else (0.0 if a_idx == 1 else p_max))

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

def run_PPO_policy_simulation():
    print("--- 1. Initialization of Continuous PPO EV Charging ---")
    
    # --- CONFIGURATION DICTIONARY ---
    simulation_config = {
        "type": "PPO-Continuous",
        "n_episodes": 300,
        "n_agents": 10,
        "simulation_hours": 24,
        "learning_rate": 1e-4,
        "update_timestep": 240,
        "k_epochs": 10,
        "grid_type": "case33bw",
        "ev_capacity": 60.0,
        "ev_max_power": 11.0,
        "n_test_episodes": 10,
        "reward_weights": {"ramp": 2.0, "track": 1.0, "scale_mw": 0.10}
    }
    
    # Initialize Run Name
    run_name = f"Continuous_PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize Metrics with Config
    metrics = EvalMetrics(run_name=run_name, config=simulation_config)
    
    # Initialize Core Systems
    grid = GridEnv(network_type=simulation_config['grid_type']) 
    driver_profiles = DataGenerator.get_nhts_profile(simulation_config['n_agents'])
    agent_bus_map = {i: (i % 30) + 2 for i in range(simulation_config['n_agents'])}
    
    # Check input dimensions
    dummy_env = EVClientEnv({'capacity': 60.0, 'max_power': 11.0, 'initial_soc': 0.5, 'soc_req': 0.8, 't_dep': 10, 'dt': 1.0})
    dummy_state = dummy_env.get_state(0.0, 0.0, [0.1]*5) 
    input_dim = len(dummy_state)

    # Create Agents and Envs
    agents = []
    envs = []
    
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
        
        agents.append(PPOAgent(
            input_dim=input_dim,
            action_dim=1,
            lr=simulation_config['learning_rate'],
            update_timestep=simulation_config['update_timestep'],   
            K_epochs=simulation_config['k_epochs']
        ))

    # --- 2. TRAINING LOOP ---
    print(f"--- 2. Starting Training ({simulation_config['n_episodes']} Episodes) ---")
    
    for episode in tqdm(range(simulation_config['n_episodes']), desc="Training"):
        total_episode_reward = 0
        total_episode_cost = 0.0  
        
        grid.reset()
        
        # Reset Environments & Variables
        active = [True] * simulation_config['n_agents']
        lambda_prev = 0.0
        volt_prev = 0.0
        prev_ev_total_mw = 0.0

        for i, env in enumerate(envs):
            env.soc = driver_profiles[i]['soc_init']
            env.current_step = 0

        for hour in range(simulation_config['simulation_hours']):
            
            price = DataGenerator.get_iso_ne_price(hour, mode='train')
            price_forecast = [DataGenerator.get_iso_ne_price((hour+h)%24, mode='train') for h in range(5)]
            base_load_mw = np.random.normal(3.5, 0.2) 

            # ----- Build an aggregate EV target load (Optional Logic) -----
            p_target_kw = 0.0
            n_active = 0
            for i, env in enumerate(envs):
                if not active[i]: continue
                n_active += 1
                soc_gap = max(0.0, env.soc_req - env.soc)
                e_gap_kwh = (soc_gap * env.capacity) / env.eta
                t_left = max(1, env.t_dep - env.current_step)
                p_req_kw = min(e_gap_kwh / t_left, env._get_max_power(env.soc))
                p_target_kw += p_req_kw
            p_target_mw = p_target_kw / 1000.0
            
            current_states = {}
            actions = {} 
            grid_injections_mw = {}
            delta_ev_mw = 0.0 
            
            # --- 1. Agent Actions ---
            for i, agent in enumerate(agents):
                if not active[i]: continue

                s_t = envs[i].get_state(
                    grid_signal=lambda_prev,
                    voltage_dev=volt_prev,
                    price_forecast=price_forecast
                )
                current_states[i] = s_t
                
                # Get Continuous Action [-1, 1]
                raw_action = agent.get_action(s_t, eval_mode=False)
                
                # Scaling
                p_max_phys = envs[i]._get_max_power(envs[i].soc) 
                p_kw = raw_action * p_max_phys
               
                actions[i] = (raw_action, p_kw)
                
                bus = agent_bus_map[i]
                grid_injections_mw[bus] = grid_injections_mw.get(bus, 0.0) + (p_kw / 1000.0)

            # --- 2. Grid Physics ---
            lambda_grid, grid_info = grid.step(grid_injections_mw, base_load_mw)
            
            ev_total_mw = float(sum(grid_injections_mw.values()))
            delta_ev_mw = ev_total_mw - prev_ev_total_mw

            # Stability penalties
            SCALE_MW = simulation_config['reward_weights']['scale_mw']
            w_ramp = simulation_config['reward_weights']['ramp']
            w_track = simulation_config['reward_weights']['track']
            
            r_ramp = -w_ramp * (delta_ev_mw / SCALE_MW) ** 2
            r_track = -w_track * ((ev_total_mw - p_target_mw) / SCALE_MW) ** 2
            
            shared_stability_penalty = 0.0
            if n_active > 0:
                shared_stability_penalty = (r_ramp + r_track) / n_active

            # Update memory
            prev_ev_total_mw = ev_total_mw
            lambda_prev = float(lambda_grid)
            volt_prev = float(grid_info['max_voltage'] - 1.0)
            
            metrics.log_step(base_load_mw + sum(grid_injections_mw.values()))
            
            # --- 3. Agent Update ---
            for i, agent in enumerate(agents):
                if not active[i]: continue

                raw_action, p_kw = actions[i]

                r_t, done, new_soc, energy_cost = envs[i].step(
                    action_power=p_kw,
                    grid_signal=lambda_grid,
                    voltage_dev=grid_info['max_voltage'] - 1.0,
                    price_current=price
                )
                total_episode_cost += energy_cost
                
                # Add shared penalty to individual reward
                r_t += shared_stability_penalty

                s_next = envs[i].get_state(lambda_grid, grid_info['max_voltage'] - 1.0, price_forecast)

                agent.update(current_states[i], raw_action, r_t, s_next, done=done)

                total_episode_reward += r_t

                if done:
                    active[i] = False

        # --- End of Episode Logging ---
        episode_satisfactions = []
        for i, env in enumerate(envs):
            req = driver_profiles[i]['soc_req']
            ratio = min(1.0, env.soc / req) if req > 0 else 1.0
            episode_satisfactions.append(ratio)
        
        metrics.log_satisfaction(episode_satisfactions)
        metrics.log_episode(total_episode_reward, mode='train')
        metrics.log_cost(total_episode_cost)
        
        if (episode+1) % 10 == 0:
            print(f"  Ep {episode+1} | Reward: {total_episode_reward:.2f} | Cost: ${total_episode_cost:.2f}")

    # --- 3. TESTING PHASE ---
    print("--- 3. Evaluation (Continuous Actions) ---")

    N_TEST_EPISODES = simulation_config['n_test_episodes']
    all_test_costs = []

    for test_ep in range(N_TEST_EPISODES):
        total_test_reward = 0.0
        total_test_cost = 0.0

        grid.reset()
        for env in envs:
            env.soc = 0.2
            env.current_step = 0

        lambda_prev = 0.0
        volt_prev = 0.0
        prev_ev_total_mw = 0.0
        active = [True] * simulation_config['n_agents']

        for hour in range(simulation_config['simulation_hours']):
            price_test = DataGenerator.get_iso_ne_price(hour, mode='test')
            price_forecast_test = [DataGenerator.get_iso_ne_price((hour + h) % 24, mode='test') for h in range(5)]
            base_load_test = np.random.normal(3.8, 0.3)

            grid_injections_test = {}
            actions = [None] * simulation_config['n_agents']
            current_states = [None] * simulation_config['n_agents']

            # 1) Action selection
            for i, agent in enumerate(agents):
                if not active[i]: continue

                s_t = envs[i].get_state(
                    grid_signal=lambda_prev,
                    voltage_dev=volt_prev,
                    price_forecast=price_forecast_test
                )
                current_states[i] = s_t

                # Deterministic Action
                raw_action = agent.get_action(s_t, eval_mode=True)

                p_max_phys = envs[i]._get_max_power(envs[i].soc)
                p_kw = raw_action * p_max_phys
                
                # Cost calc
                total_test_cost += p_kw * 1.0 * price_test

                actions[i] = (raw_action, p_kw)

                bus = agent_bus_map[i]
                grid_injections_test[bus] = grid_injections_test.get(bus, 0.0) + (p_kw / 1000.0)

            # 2) Grid step
            lambda_grid, grid_info = grid.step(grid_injections_test, base_load_test)

            ev_total_mw = float(sum(grid_injections_test.values()))
            delta_ev_mw = ev_total_mw - prev_ev_total_mw

            # 3) Environment step
            for i, agent in enumerate(agents):
                if not active[i]: continue

                raw_action, p_kw = actions[i]

                r_t, done, _, _ = envs[i].step(
                    action_power=p_kw,
                    grid_signal=lambda_grid,
                    voltage_dev=grid_info['max_voltage'] - 1.0,
                    price_current=price_test
                )

                total_test_reward += r_t

                if done:
                    active[i] = False

            # 4) Update signals
            prev_ev_total_mw = ev_total_mw
            lambda_prev = float(lambda_grid)
            volt_prev = float(grid_info['max_voltage'] - 1.0)

        metrics.log_episode(total_test_reward, mode='test')
        all_test_costs.append(total_test_cost)

        print(f"  TestEp {test_ep+1}/{N_TEST_EPISODES} | Reward: {total_test_reward:.2f} | Cost: ${total_test_cost:.2f}")

    print(f"\nTest Phase Avg Cost: ${np.mean(all_test_costs):.2f}  (over {N_TEST_EPISODES} episodes)")

    # Final Results
    print(f"\nGrid Stability (sigma_g): {metrics.compute_stability_metric():.4f} MW")
    metrics.plot_metrics()

def run_SAC_simulation():
    """Standalone SAC simulation — same structure as PPO."""
    print("--- 1. Initialization of SAC EV Charging ---")

    simulation_config = {
        "type": "SAC-Continuous",
        "n_episodes": 300,
        "n_agents": 10,
        "simulation_hours": 24,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "warmup_steps": 500,
        "grid_type": "case33bw",
        "ev_capacity": 60.0,
        "ev_max_power": 11.0,
        "n_test_episodes": 10,
        "reward_weights": {"ramp": 2.0, "track": 1.0, "scale_mw": 0.10},
    }

    run_name = f"SAC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics = EvalMetrics(run_name=run_name, config=simulation_config)
    grid = GridEnv(network_type=simulation_config['grid_type'])
    driver_profiles = DataGenerator.get_nhts_profile(simulation_config['n_agents'])
    agent_bus_map = {i: (i % 30) + 2 for i in range(simulation_config['n_agents'])}

    dummy_env = EVClientEnv({'capacity': 60.0, 'max_power': 11.0, 'initial_soc': 0.5,
                             'soc_req': 0.8, 't_dep': 10, 'dt': 1.0})
    dummy_state = dummy_env.get_state(0.0, 0.0, [0.1] * 5)
    input_dim = len(dummy_state)

    agents, envs = [], []
    for i in range(simulation_config['n_agents']):
        config = {
            'capacity': simulation_config['ev_capacity'],
            'max_power': simulation_config['ev_max_power'],
            'initial_soc': driver_profiles[i]['soc_init'],
            'soc_req': driver_profiles[i]['soc_req'],
            't_dep': driver_profiles[i]['duration'],
            'dt': 1.0,
        }
        envs.append(EVClientEnv(config))
        agents.append(SACAgent(
            input_dim=input_dim, action_dim=1,
            lr=simulation_config['learning_rate'],
            batch_size=simulation_config['batch_size'],
            warmup_steps=simulation_config['warmup_steps'],
        ))

    # --- TRAINING ---
    print(f"--- 2. Starting Training ({simulation_config['n_episodes']} Episodes) ---")
    for episode in tqdm(range(simulation_config['n_episodes']), desc="Training SAC"):
        total_episode_reward = 0.0
        total_episode_cost = 0.0
        grid.reset()

        active = [True] * simulation_config['n_agents']
        lambda_prev, volt_prev, prev_ev_total_mw = 0.0, 0.0, 0.0

        for i, env in enumerate(envs):
            env.soc = driver_profiles[i]['soc_init']
            env.current_step = 0

        for hour in range(simulation_config['simulation_hours']):
            price = DataGenerator.get_iso_ne_price(hour, mode='train')
            price_forecast = [DataGenerator.get_iso_ne_price((hour + h) % 24, mode='train') for h in range(5)]
            base_load_mw = np.random.normal(3.5, 0.2)

            p_target_kw, n_active = 0.0, 0
            for i, env in enumerate(envs):
                if not active[i]: continue
                n_active += 1
                soc_gap = max(0.0, env.soc_req - env.soc)
                e_gap = (soc_gap * env.capacity) / env.eta
                t_left = max(1, env.t_dep - env.current_step)
                p_target_kw += min(e_gap / t_left, env._get_max_power(env.soc))
            p_target_mw = p_target_kw / 1000.0

            current_states, actions, grid_injections_mw = {}, {}, {}

            for i, agent in enumerate(agents):
                if not active[i]: continue
                s_t = envs[i].get_state(lambda_prev, volt_prev, price_forecast)
                current_states[i] = s_t
                raw_action = agent.get_action(s_t, eval_mode=False)
                p_max_phys = envs[i]._get_max_power(envs[i].soc)
                p_kw = raw_action * p_max_phys
                actions[i] = (raw_action, p_kw)
                bus = agent_bus_map[i]
                grid_injections_mw[bus] = grid_injections_mw.get(bus, 0.0) + (p_kw / 1000.0)

            lambda_grid, grid_info = grid.step(grid_injections_mw, base_load_mw)
            ev_total_mw = float(sum(grid_injections_mw.values()))
            delta_ev_mw = ev_total_mw - prev_ev_total_mw

            SCALE_MW = simulation_config['reward_weights']['scale_mw']
            r_ramp = -simulation_config['reward_weights']['ramp'] * (delta_ev_mw / SCALE_MW) ** 2
            r_track = -simulation_config['reward_weights']['track'] * ((ev_total_mw - p_target_mw) / SCALE_MW) ** 2
            shared_penalty = (r_ramp + r_track) / max(1, n_active)

            prev_ev_total_mw = ev_total_mw
            lambda_prev = float(lambda_grid)
            volt_prev = float(grid_info['max_voltage'] - 1.0)
            metrics.log_step(base_load_mw + sum(grid_injections_mw.values()))

            for i, agent in enumerate(agents):
                if not active[i]: continue
                raw_action, p_kw = actions[i]
                r_t, done, _, energy_cost = envs[i].step(p_kw, lambda_grid, grid_info['max_voltage'] - 1.0, price)
                total_episode_cost += energy_cost
                r_t += shared_penalty
                s_next = envs[i].get_state(lambda_grid, grid_info['max_voltage'] - 1.0, price_forecast)
                agent.update(current_states[i], raw_action, r_t, s_next, done=done)
                total_episode_reward += r_t
                if done:
                    active[i] = False

        sats = [min(1.0, envs[i].soc / driver_profiles[i]['soc_req'])
                if driver_profiles[i]['soc_req'] > 0 else 1.0
                for i in range(simulation_config['n_agents'])]
        metrics.log_satisfaction(sats)
        metrics.log_episode(total_episode_reward, mode='train')
        metrics.log_cost(total_episode_cost)

        if (episode + 1) % 10 == 0:
            print(f"  Ep {episode+1} | Reward: {total_episode_reward:.2f} | Cost: ${total_episode_cost:.2f}")

    # --- TESTING ---
    print("--- 3. Evaluation (SAC Continuous) ---")
    N_TEST = simulation_config['n_test_episodes']
    for test_ep in range(N_TEST):
        total_test_reward = 0.0
        grid.reset()
        for env in envs:
            env.soc = 0.2; env.current_step = 0
        active = [True] * simulation_config['n_agents']
        lambda_prev, volt_prev, prev_ev_total_mw = 0.0, 0.0, 0.0

        for hour in range(simulation_config['simulation_hours']):
            price_test = DataGenerator.get_iso_ne_price(hour, mode='test')
            pf_test = [DataGenerator.get_iso_ne_price((hour + h) % 24, mode='test') for h in range(5)]
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
            ev_mw = float(sum(grid_inj.values()))
            for i, agent in enumerate(agents):
                if not active[i]: continue
                _, p_kw = actions[i]
                r_t, done, _, _ = envs[i].step(p_kw, l_g, g_i['max_voltage'] - 1.0, price_test)
                total_test_reward += r_t
                if done: active[i] = False
            prev_ev_total_mw = ev_mw
            lambda_prev = float(l_g)
            volt_prev = float(g_i['max_voltage'] - 1.0)
        metrics.log_episode(total_test_reward, mode='test')
        print(f"  TestEp {test_ep+1}/{N_TEST} | Reward: {total_test_reward:.2f}")

    print(f"\nGrid Stability (sigma_g): {metrics.compute_stability_metric():.4f} MW")
    metrics.plot_metrics()


def run_federated_simulation():
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
        n_episodes=300,
        n_test_episodes=10,
        n_agents=10,
        verbose=True,
    )
    m.plot_metrics()


def main():
    parser = argparse.ArgumentParser(description="FDRL EV Charging Simulation")
    parser.add_argument(
        'mode',
        type=int,
        nargs='?',
        help="1=PPO, 2=Q-Learning, 3=SAC, 4=Federated, 5=Comparison"
    )

    args = parser.parse_args()

    if args.mode is None:
        choice = questionary.select(
            "Select simulation mode:",
            choices=[
                questionary.Choice("PPO Policy Simulation", value=1),
                questionary.Choice("Q-Learning Simulation", value=2),
                questionary.Choice("SAC Policy Simulation", value=3),
                questionary.Choice("Federated Training (select policy + aggregation)", value=4),
                questionary.Choice("Full Comparison Pipeline (all combos)", value=5),
            ],
            use_arrow_keys=True
        ).ask()

        if choice is None:
            sys.exit(0)
        args.mode = choice

    if args.mode == 1:
        run_PPO_policy_simulation()
    elif args.mode == 2:
        run_Q_learning_simulation()
    elif args.mode == 3:
        run_SAC_simulation()
    elif args.mode == 4:
        run_federated_simulation()
    elif args.mode == 5:
        run_comparison()
    else:
        print(f"Error: {args.mode} is not a valid option. Use 1-5.")
        sys.exit(1)


if __name__ == "__main__":
    main()