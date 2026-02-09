import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as nw
from EVClientEnv import EVClientEnv
from GridEnv import GridEnv
from EvalMetrics import EvalMetrics
from DataLoader import DataGenerator
from QLearningAgent import QLearningAgent
from tqdm import tqdm
from datetime import datetime
# --- RECAP: Ensure these classes are defined as per previous steps ---
# 1. EVClientEnv (Physics & Rewards)
# 2. GridEnv (Pandapower OPF/PF)
# 3. QLearningAgent (RL Algo)
# 4. EvalMetrics (Logger)
# 5. DataGenerator (ISO NE / NHTS Data)

def run_Q_learning_simulation():
    # --- 1. CONFIGURATION & INITIALIZATION ---
    print("--- 1. Initialization of Federated EV Charging Simulation ---")
    
    # Simulation Parameters
    N_EPISODES = 100       # Total training episodes
    N_AGENTS = 20          # Number of EVs [cite: 8]
    SIMULATION_HOURS = 24 # Daily cycle
    
    # Initialize Core Systems
    run_name = input("Enter a name for this simulation run (for saving results): ")
    run_name = run_name.strip() + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" + f"_{N_EPISODES}" + f"_{N_AGENTS}" + f"{SIMULATION_HOURS}" \
                if run_name else f"run_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{N_EPISODES}_{N_AGENTS}_{SIMULATION_HOURS}"
    metrics = EvalMetrics(run_name = run_name)
    grid = GridEnv(network_type='case33bw') # IEEE 33-bus distribution network
    
    # Generate Driver Profiles (NHTS 2017)
    # Different profiles for each agent: Arrival time, Departure time, SOC req
    # TODO : use real NHTS or  data for more realism, here we simulate it
    driver_profiles = DataGenerator.get_nhts_profile(N_AGENTS)
    
    # Map Agents to Grid Buses
    # We attach EVs to load buses (e.g., bus 2 to 6) to impact the grid
    agent_bus_map = {i: (i % 30) + 2 for i in range(N_AGENTS)}
    
    # Create Agents and Environments
    agents = []
    envs = []
    
    for i in range(N_AGENTS):
        # [cite_start]EV Physics Config [cite: 13-17]
        config = {
            'capacity': 60.0,    # kWh (e.g., Tesla Model 3)
            'max_power': 11.0,   # kW (Level 2 Charger) [cite: 22]
            'initial_soc': driver_profiles[i]['soc_init'],
            'soc_req': driver_profiles[i]['soc_req'],
            't_dep': driver_profiles[i]['duration'], # Time steps until departure
            'dt': 1.0            # 1 Hour steps
        }
        envs.append(EVClientEnv(config))
        
        # Q-Learning Agent
        # Actions: 0=Discharge (-11kW), 1=Idle (0kW), 2=Charge (+11kW)
        agents.append(QLearningAgent(action_space_size=3, state_bins=None , epsilon=1.0))

    # --- 2. TRAINING LOOP (CONVERGENCE & STABILITY) ---
    print(f"--- 2. Starting Training ({N_EPISODES} Episodes) ---")
    
    for episode in tqdm(range(N_EPISODES), desc="Training Episodes"):
        # Reset Episode Metrics
        total_episode_reward = 0
        episode_grid_loads = []
        
        # Reset Environments (Vehicles arrive with low SOC)
        grid.reset()
        for i, env in enumerate(envs):
            env.soc = driver_profiles[i]['soc_init']
            env.current_step = 0

        # --- Daily Cycle (00:00 to 23:00) ---
        for hour in range(SIMULATION_HOURS):
            
            # A. Global Observation (Context)
            # [cite_start]Price signal (ISO NE) [cite: 85]
            price = DataGenerator.get_iso_ne_price(hour, mode='train')
            price_forecast = [DataGenerator.get_iso_ne_price((hour+h)%24, mode='train') for h in range(5)]
            
            # Base Load (Non-EV demand)
            base_load_mw = np.random.normal(3.5, 0.2) 
            
            # B. Agent Actions (Decentralized)
            current_states = {}
            actions = {}
            grid_injections_mw = {}
            
            for i, agent in enumerate(agents):
                # [cite_start]Observe State: [SOC, Time, Price, Grid_Signal, Voltage] [cite: 85]
                # Note: We use previous step's grid signal (initially 0)
                s_t = envs[i].get_state(grid_signal=0.0, voltage_dev=0.0, price_forecast=price_forecast)
                current_states[i] = s_t
                
                # Choose Action (Epsilon-Greedy)
                a_idx = agent.get_action(s_t, eval_mode=False)
                
                # Convert Action Index -> Physical Power (kW)
                # Simple mapping: 0 -> -Pmax, 1 -> 0, 2 -> +Pmax
                p_max = envs[i]._get_max_power(envs[i].soc) # [cite: 22]
                if a_idx == 0: p_kw = -p_max   # V2G
                elif a_idx == 1: p_kw = 0.0
                else: p_kw = p_max             # G2V
                
                actions[i] = (a_idx, p_kw)
                
                # Prepare Grid Injection (kW -> MW)
                bus = agent_bus_map[i]
                grid_injections_mw[bus] = grid_injections_mw.get(bus, 0.0) + (p_kw / 1000.0)

            # C. Grid Physics (Centralized Aggregation)
            # [cite_start]Solve Power Flow / OPF to check constraints [cite: 29-30]
            lambda_grid, grid_info = grid.step(grid_injections_mw, base_load_mw)
            
            # Log Grid Stability Metric (Total Load)
            current_total_load = base_load_mw + sum(grid_injections_mw.values())
            metrics.log_step(current_total_load) # For sigma_g calculation
            
            # D. Agent Update (Learning)
            for i, agent in enumerate(agents):
                a_idx, p_kw = actions[i]
                
                # Execute Action in EV Physics Model
                # [cite_start]Reward includes: Energy Cost, Grid Penalty (lambda), Satisfaction [cite: 86-91]
                r_t, done, new_soc = envs[i].step(
                    action_power=p_kw,
                    grid_signal=lambda_grid,       # Feedback from Grid
                    voltage_dev=grid_info['max_voltage'] - 1.0, 
                    price_current=price
                )
                
                # Observe Next State
                s_next = envs[i].get_state(lambda_grid, grid_info['max_voltage']-1.0, price_forecast)
                
                # [cite_start]Update Q-Table [cite: 94]
                agent.update(current_states[i], a_idx, r_t, s_next)
                
                total_episode_reward += r_t

        # Fin de l'épisode : Calcul de la satisfaction
        episode_satisfactions = []
        
        for i, env in enumerate(envs):
            # Le SOC requis pour cet agent (défini dans le profil NHTS au début)
            req = driver_profiles[i]['soc_req']
            
            # Le SOC final atteint par l'agent
            final = env.soc
            
            # Calcul du ratio (plafonné à 1.0 car on ne peut pas être "plus que satisfait")
            # Si req est 0 (rare), on met 1.0
            ratio = min(1.0, final / req) if req > 0 else 1.0
            episode_satisfactions.append(ratio)
        
        # End of Episode Processing
        metrics.log_satisfaction(episode_satisfactions)
        metrics.log_episode(total_episode_reward, mode='train')
        
        # Decay Epsilon (Reduce exploration over time)
        for agent in agents:
            agent.epsilon = max(0.05, agent.epsilon * 0.95)
            
        if (episode+1) % 10 == 0:
            print(f"  Episode {episode+1}/{N_EPISODES} | Reward: {total_episode_reward:.2f} | Epsilon: {agents[0].epsilon:.2f}")

    # --- 3. TESTING PHASE (GENERALIZATION) ---
    print("--- 3. Starting Evaluation (Generalization Phase) ---")
    
    # Reset for Test Episode
    total_test_reward = 0
    grid.reset()
    for env in envs: env.soc = 0.2; env.current_step = 0
    
    for hour in range(SIMULATION_HOURS):
        # Use TEST Data (Unseen prices/profiles)
        price_test = DataGenerator.get_iso_ne_price(hour, mode='test')
        price_forecast_test = [DataGenerator.get_iso_ne_price((hour+h)%24, mode='test') for h in range(5)]
        base_load_test = np.random.normal(3.8, 0.3) # Slightly different load profile
        
        grid_injections_test = {}
        
        # Agents act WITHOUT Exploration (Greedy)
        for i, agent in enumerate(agents):
            s_t = envs[i].get_state(0.0, 0.0, price_forecast_test)
            
            # eval_mode=True forces epsilon=0
            a_idx = agent.get_action(s_t, eval_mode=True)
            
            p_max = envs[i]._get_max_power(envs[i].soc)
            p_kw = -p_max if a_idx == 0 else (0.0 if a_idx == 1 else p_max)
            
            bus = agent_bus_map[i]
            grid_injections_test[bus] = grid_injections_test.get(bus, 0.0) + (p_kw / 1000.0)
            
            # Step Physics (No learning update needed during inference)
            r_t, _, _ = envs[i].step(p_kw, 0.0, 0.0, price_test)
            total_test_reward += r_t
            
        # Grid Step (Just to log valid states if needed)
        grid.step(grid_injections_test, base_load_test)

    metrics.log_episode(total_test_reward, mode='test')

    # --- 4. RESULTS & VISUALIZATION ---
    print("\n--- 4. Final Metrics ---")
    
    # 1. Grid Stability (Standard Deviation of Power Changes)
    sigma_g = metrics.compute_stability_metric()
    print(f"-> Grid Stability (sigma_g): {sigma_g:.4f} MW (Lower is better)")
    
    # 2. Generalization Gap
    train_perf = np.mean(metrics.episode_rewards[-5:])
    test_perf = metrics.test_rewards[-1]
    print(f"-> Train Performance (Last 5 avg): {train_perf:.2f}")
    print(f"-> Test Performance (Unseen Data): {test_perf:.2f}")
    
    # Plotting
    metrics.plot_metrics()




from PPOAgent import PPOAgent  # <--- CHANGED

import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as nw
from EVClientEnv import EVClientEnv
from GridEnv import GridEnv
from EvalMetrics import EvalMetrics
from DataLoader import DataGenerator
from PPOAgent import PPOAgent 
from tqdm import tqdm
from datetime import datetime

def run_POO_policy_simulation():
    # --- 1. CONFIGURATION ---
    print("--- 1. Initialization of Continuous PPO EV Charging ---")
    
    # Simulation Parameters
    N_EPISODES = 150       # Increased slightly for continuous convergence
    N_AGENTS = 20          
    SIMULATION_HOURS = 24 
    
    run_name = f"Continuous_PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics = EvalMetrics(run_name = run_name)
    grid = GridEnv(network_type='case33bw') 
    
    driver_profiles = DataGenerator.get_nhts_profile(N_AGENTS)
    agent_bus_map = {i: (i % 30) + 2 for i in range(N_AGENTS)}
    
    # --- DYNAMIC STATE CHECK ---
    dummy_env = EVClientEnv({'capacity': 60.0, 'max_power': 11.0, 'initial_soc': 0.5, 'soc_req': 0.8, 't_dep': 10, 'dt': 1.0})
    dummy_state = dummy_env.get_state(0.0, 0.0, [0.1]*5)
    input_dim = len(dummy_state)

    agents = []
    envs = []
    
    for i in range(N_AGENTS):
        config = {
            'capacity': 60.0,    
            'max_power': 11.0,   
            'initial_soc': driver_profiles[i]['soc_init'],
            'soc_req': driver_profiles[i]['soc_req'],
            't_dep': driver_profiles[i]['duration'], 
            'dt': 1.0            
        }
        envs.append(EVClientEnv(config))
        
        # Action dim is 1 (Scalar power value)
        agents.append(PPOAgent(input_dim=input_dim, action_dim=1, lr=1e-4)) # Lower LR for continuous stability

    # --- 2. TRAINING LOOP ---
    print(f"--- 2. Starting Training ({N_EPISODES} Episodes) ---")
    
    for episode in tqdm(range(N_EPISODES), desc="Training"):
        total_episode_reward = 0
        grid.reset()
        for i, env in enumerate(envs):
            env.soc = driver_profiles[i]['soc_init']
            env.current_step = 0

        for hour in range(SIMULATION_HOURS):
            
            price = DataGenerator.get_iso_ne_price(hour, mode='train')
            price_forecast = [DataGenerator.get_iso_ne_price((hour+h)%24, mode='train') for h in range(5)]
            base_load_mw = np.random.normal(3.5, 0.2) 
            
            current_states = {}
            actions = {} # Stores raw [-1, 1] action
            grid_injections_mw = {}
            
            for i, agent in enumerate(agents):
                s_t = envs[i].get_state(grid_signal=0.0, voltage_dev=0.0, price_forecast=price_forecast)
                current_states[i] = s_t
                
                # Get Continuous Action in [-1, 1]
                raw_action = agent.get_action(s_t, eval_mode=False)
                
                # --- SCALING LOGIC ---
                # Get physical max power allowed by current SOC (e.g., can't discharge if empty)
                p_max_phys = envs[i]._get_max_power(envs[i].soc) 
                
                # Map [-1, 1] -> [-p_max_phys, +p_max_phys]
                p_kw = raw_action * p_max_phys
                
                actions[i] = (raw_action, p_kw)
                
                bus = agent_bus_map[i]
                grid_injections_mw[bus] = grid_injections_mw.get(bus, 0.0) + (p_kw / 1000.0)

            lambda_grid, grid_info = grid.step(grid_injections_mw, base_load_mw)
            metrics.log_step(base_load_mw + sum(grid_injections_mw.values()))
            
            for i, agent in enumerate(agents):
                raw_action, p_kw = actions[i]
                
                r_t, done, new_soc = envs[i].step(
                    action_power=p_kw,
                    grid_signal=lambda_grid,       
                    voltage_dev=grid_info['max_voltage'] - 1.0, 
                    price_current=price
                )
                
                s_next = envs[i].get_state(lambda_grid, grid_info['max_voltage']-1.0, price_forecast)
                
                # Store RAW action [-1, 1] in PPO buffer, not the scaled kW
                agent.update(current_states[i], raw_action, r_t, s_next, done=done)
                
                total_episode_reward += r_t

        # Metrics & Logs
        episode_satisfactions = []
        for i, env in enumerate(envs):
            req = driver_profiles[i]['soc_req']
            ratio = min(1.0, env.soc / req) if req > 0 else 1.0
            episode_satisfactions.append(ratio)
        
        metrics.log_satisfaction(episode_satisfactions)
        metrics.log_episode(total_episode_reward, mode='train')
        
        if (episode+1) % 10 == 0:
            print(f"  Ep {episode+1} | Reward: {total_episode_reward:.2f}")

    # --- 3. TESTING PHASE ---
    print("--- 3. Evaluation (Continuous Actions) ---")
    
    total_test_reward = 0
    grid.reset()
    for env in envs: env.soc = 0.2; env.current_step = 0
    
    for hour in range(SIMULATION_HOURS):
        price_test = DataGenerator.get_iso_ne_price(hour, mode='test')
        price_forecast_test = [DataGenerator.get_iso_ne_price((hour+h)%24, mode='test') for h in range(5)]
        base_load_test = np.random.normal(3.8, 0.3)
        
        grid_injections_test = {}
        
        for i, agent in enumerate(agents):
            s_t = envs[i].get_state(0.0, 0.0, price_forecast_test)
            
            # eval_mode=True -> Deterministic Mean Action
            raw_action = agent.get_action(s_t, eval_mode=True)
            
            p_max_phys = envs[i]._get_max_power(envs[i].soc)
            p_kw = raw_action * p_max_phys
            
            bus = agent_bus_map[i]
            grid_injections_test[bus] = grid_injections_test.get(bus, 0.0) + (p_kw / 1000.0)
            
            r_t, _, _ = envs[i].step(p_kw, 0.0, 0.0, price_test)
            total_test_reward += r_t
            
        grid.step(grid_injections_test, base_load_test)

    metrics.log_episode(total_test_reward, mode='test')
    
    # Final Results
    print(f"\nGrid Stability (sigma_g): {metrics.compute_stability_metric():.4f} MW")
    metrics.plot_metrics()





# Execute
if __name__ == "__main__":
    # run_Q_learning_simulation()
    run_POO_policy_simulation()