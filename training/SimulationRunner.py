import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as nw
from tqdm import tqdm
from datetime import datetime

# Import your local modules
from env.EVClientEnv import EVClientEnv
from env.GridEnv import GridEnv
from utils.EvalMetrics import EvalMetrics
from utils.DataLoader import DataGenerator

# Import Agents
from agents.QLearningAgent import QLearningAgent
# from PPOAgent import PPOAgent  # Uncomment when you have the file

class SimulationRunner:
    """
    A modular runner for EV Charging simulations.
    Compatible with any agent inheriting from BaseAgent (QLearning, PPO, etc).
    """
    def __init__(self, agent_class, agent_config, run_config):
        self.agent_class = agent_class
        self.agent_config = agent_config
        self.cfg = run_config
        
        # --- Initialization ---
        self.run_name = f"{self.cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics = EvalMetrics(run_name=self.run_name)
        
        # Grid Environment (IEEE 33-bus)
        self.grid = GridEnv(network_type='case33bw')
        
        # Data & Profiles
        self.driver_profiles = DataGenerator.get_nhts_profile(self.cfg['n_agents'])
        self.agent_bus_map = {i: (i % 30) + 2 for i in range(self.cfg['n_agents'])}
        
        # Initialize Environments & Agents
        self.envs = []
        self.agents = []
        self._init_simulation_objects()

    def _init_simulation_objects(self):
        """Helper to instantiate Environments and Agents."""
        # 1. Create one dummy env to inspect state dimension (crucial for PPO)
        dummy_config = self._get_ev_config(0)
        dummy_env = EVClientEnv(dummy_config)
        dummy_state = dummy_env.get_state(0, 0, [0]*5)
        state_dim = len(dummy_state)
        
        # 2. Update agent config if input_dim is needed (for Neural Networks)
        if 'input_dim' not in self.agent_config:
            self.agent_config['input_dim'] = state_dim

        # 3. Instantiate All Agents and Envs
        for i in range(self.cfg['n_agents']):
            # Environment
            config = self._get_ev_config(i)
            self.envs.append(EVClientEnv(config))
            
            # Agent (Injecting specific class)
            # We use **self.agent_config to unpack dictionary into kwargs
            self.agents.append(self.agent_class(**self.agent_config))

    def _get_ev_config(self, agent_idx):
        """Returns the physics config for a specific agent based on profile."""
        return {
            'capacity': 60.0,
            'max_power': 11.0,
            'initial_soc': self.driver_profiles[agent_idx]['soc_init'],
            'soc_req': self.driver_profiles[agent_idx]['soc_req'],
            't_dep': self.driver_profiles[agent_idx]['duration'],
            'dt': 1.0
        }

    def train(self):
        """Main Training Loop"""
        print(f"--- Starting Training: {self.cfg['n_episodes']} Episodes ---")
        
        for episode in tqdm(range(self.cfg['n_episodes']), desc="Training"):
            self._run_episode(mode='train', episode_idx=episode)
            
            # Epsilon Decay (Specific to Q-Learning, skipped for PPO)
            self._decay_exploration()
            
            # Periodic Logging
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.metrics.episode_rewards[-10:])
                print(f"  Ep {episode+1} | Avg Reward: {avg_reward:.2f}")

    def evaluate(self):
        """Evaluation / Generalization Phase"""
        print("--- Starting Evaluation (Test Phase) ---")
        # Run a single test episode (or multiple if configured)
        total_reward = self._run_episode(mode='test')
        
        # Final Metrics Calculation
        print("\n--- Final Metrics ---")
        sigma_g = self.metrics.compute_stability_metric()
        print(f"-> Grid Stability (sigma_g): {sigma_g:.4f} MW")
        print(f"-> Test Reward: {total_reward:.2f}")
        
        self.metrics.plot_metrics()

    def _run_episode(self, mode='train', episode_idx=0):
        """
        Runs a single episode (Day).
        Args:
            mode: 'train' (Learning enabled) or 'test' (Greedy actions, no learning)
        """
        # Reset Logic
        self.grid.reset()
        total_episode_reward = 0
        episode_satisfactions = []
        
        # Reset EVs
        for i, env in enumerate(self.envs):
            env.soc = self.driver_profiles[i]['soc_init'] if mode=='train' else 0.2
            env.current_step = 0
            
        # --- Daily Cycle Loop ---
        for hour in range(self.cfg['sim_hours']):
            # 1. Global Context
            price, price_forecast = self._get_context_data(hour, mode)
            base_load_mw = np.random.normal(3.5, 0.2)
            
            # 2. Decentralized Decision Making
            current_states = {}
            actions = {}
            grid_injections_mw = {}
            
            for i, agent in enumerate(self.agents):
                # State Observation
                # Note: We pass 0.0 for grid signals initially, updated in next step
                s_t = self.envs[i].get_state(grid_signal=0.0, voltage_dev=0.0, price_forecast=price_forecast)
                current_states[i] = s_t
                
                # Action Selection
                # eval_mode=True disables exploration (epsilon=0 or greedy policy)
                is_eval = (mode == 'test')
                a_idx = agent.get_action(s_t, eval_mode=is_eval)
                
                # Map Action Index to Physics
                p_kw = self._action_to_power(a_idx, self.envs[i])
                actions[i] = (a_idx, p_kw)
                
                # Aggregate Load
                bus = self.agent_bus_map[i]
                grid_injections_mw[bus] = grid_injections_mw.get(bus, 0.0) + (p_kw / 1000.0)

            # 3. Grid Physics Step (Centralized)
            lambda_grid, grid_info = self.grid.step(grid_injections_mw, base_load_mw)
            
            if mode == 'train':
                self.metrics.log_step(base_load_mw + sum(grid_injections_mw.values()))

            # 4. Agent Learning Step
            for i, agent in enumerate(self.agents):
                a_idx, p_kw = actions[i]
                
                # Execute Action & Get Reward
                r_t, done, new_soc = self.envs[i].step(
                    action_power=p_kw,
                    grid_signal=lambda_grid,
                    voltage_dev=grid_info['max_voltage'] - 1.0,
                    price_current=price
                )
                
                # Learning Update (Only in Train mode)
                if mode == 'train':
                    s_next = self.envs[i].get_state(lambda_grid, grid_info['max_voltage']-1.0, price_forecast)
                    # Supports both Q-Learning (4 args) and PPO (5 args with done)
                    try:
                        agent.update(current_states[i], a_idx, r_t, s_next, done=done)
                    except TypeError:
                        # Fallback for agents that don't take 'done' (like basic Q-Table)
                        agent.update(current_states[i], a_idx, r_t, s_next)
                
                total_episode_reward += r_t

        # --- End of Episode Stats ---
        self._calculate_satisfaction()
        self.metrics.log_episode(total_episode_reward, mode=mode)
        return total_episode_reward

    def _get_context_data(self, hour, mode):
        """Helper to fetch prices based on mode."""
        price = DataGenerator.get_iso_ne_price(hour, mode=mode)
        forecast = [DataGenerator.get_iso_ne_price((hour+h)%24, mode=mode) for h in range(5)]
        return price, forecast

    def _action_to_power(self, a_idx, env):
        """Maps discrete action index 0,1,2 to kW."""
        p_max = env._get_max_power(env.soc)
        if a_idx == 0: return -p_max # Discharge
        if a_idx == 1: return 0.0    # Idle
        if a_idx == 2: return p_max  # Charge
        return 0.0

    def _decay_exploration(self):
        """Decays epsilon if the agent has that attribute (Q-Learning)."""
        for agent in self.agents:
            if hasattr(agent, 'epsilon'):
                agent.epsilon = max(0.05, agent.epsilon * 0.95)

    def _calculate_satisfaction(self):
        """Logs satisfaction metrics at end of episode."""
        sats = []
        for i, env in enumerate(self.envs):
            req = self.driver_profiles[i]['soc_req']
            final = env.soc
            ratio = min(1.0, final / req) if req > 0 else 1.0
            sats.append(ratio)
        self.metrics.log_satisfaction(sats)