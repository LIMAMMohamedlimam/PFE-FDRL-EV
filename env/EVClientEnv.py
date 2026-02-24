# import numpy as np

# class EVClientEnv:
#     """
#     Handles the physical modeling and MDP formulation for a single EV.
#     Based on Section 1.1 (Physical Model) and 2.3 (MDP Formulation).
#     """
#     def __init__(self, config):
#         # Vehicle Parameters [cite: 13-17]
#         self.capacity = config.get('capacity', 50.0)  # Ci in kWh
#         self.eta = config.get('efficiency', 0.95)     # eta
#         self.max_power_base = config.get('max_power', 7.0) # u_bar
#         self.dt = config.get('dt', 1.0)               # Delta t (hours)
        
#         # User Requirements [cite: 24, 25]
#         self.soc_req = config.get('soc_req', 0.9)     # Required SOC at departure
#         self.t_dep = config.get('t_dep', 10)          # Departure time step
        
#         # Grid/Penalty weights [cite: 98, 121]
#         self.alpha_satisfaction = config.get('alpha', 100.0) 
#         self.beta_grid = config.get('beta', 1.0)
        
#         # State variables
#         self.soc = config.get('initial_soc', 0.2)
#         self.current_step = 0
        
#         # External signals (Mock data for Price and Grid Signal)
#         self.forecast_horizon = config.get('H', 5)
        
#     def _get_max_power(self, soc):
#         """
#         Calculates P_max based on non-linear charging constraint.
#         Formula: P_max = u_bar * (1 - alpha * SOC) 
#         """
#         alpha_constraint = 0.1 # Example coefficient
#         return self.max_power_base * (1 - alpha_constraint * soc)

#     def get_state(self, grid_signal, voltage_dev, price_forecast):
#         """
#         Constructs the state vector s_{i,t}.
#         Definition: s_{i,t} = [SOC, t_remaining, Price_Forecast, Grid_Signal, Voltage_Dev]
#         [cite: 85, 121]
#         """
#         t_remaining = self.t_dep - self.current_step
        
#         # Flatten the forecast into the state vector
#         state = np.array([
#             self.soc,
#             t_remaining,
#             grid_signal,      # lambda_grid
#             voltage_dev       # delta_v
#         ] + list(price_forecast)) # Lambda_{t:t+H}
        
#         return state

#     def step(self, action_power, grid_signal, voltage_dev, price_current):
#         """
#         Executes one step in the environment.
#         Args:
#             action_power: P_{i,t} (Action)
#         Returns:
#             next_state, reward, done, info
#         """
#         # 1. Apply Constraints [cite: 21, 22]
#         p_max = self._get_max_power(self.soc)
#         # Clip power to feasible range [-P_max, P_max] (assuming symmetrical V2G)
#         p_act = np.clip(action_power, -p_max, p_max)
        
#         # 2. Update Battery Dynamics 
#         # SOC_{t+1} = SOC_t + (eta * P * dt) / C
#         delta_soc = (self.eta * p_act * self.dt) / self.capacity
#         prev_soc = self.soc
#         self.soc = np.clip(self.soc + delta_soc, 0.0, 1.0)
        
#         self.current_step += 1
        
#         # 3. Calculate Reward [cite: 86-91, 121]
#         # Term 1: Energy Cost
#         r_cost = -(price_current * p_act * self.dt)
        
#         # Term 2: Grid Penalty (Voltage/Congestion)
#         # Modeled as penalty on voltage deviation or grid signal
#         r_grid = -self.beta_grid * (voltage_dev**2 + grid_signal * abs(p_act))
        
#         # Term 3: Satisfaction Penalty (Only at departure)
#         r_satisfaction = 0
#         done = False
        
#         if self.current_step >= self.t_dep:
#             done = True
#             if self.soc < self.soc_req:
#                 # Penalty: alpha * (SOC_req - SOC_final)^2 [cite: 91]
#                 r_satisfaction = -self.alpha_satisfaction * ((self.soc_req - self.soc)**2)
        
#         total_reward = r_cost + r_grid + r_satisfaction
        
#         return total_reward, done, self.soc





import numpy as np
import math

# We use absolute imports to be robust across the modular structure
from utils.config_loader import get_config
from utils.reward_functions import compute_reward

class EVClientEnv:
    """
    Handles the physical modeling and MDP formulation for a single EV.
    Configured via env.yaml and reward.yaml.
    """
    def __init__(self, override_env_config=None):
        """
        Args:
            override_env_config (dict): Dynamic episode-specific parameters
                                        (e.g., initial_soc, soc_req, t_dep)
                                        that override the base env.yaml constants.
        """
        # Load base configs
        self.env_config = get_config('env')
        self.reward_config = get_config('reward')
        
        # Merge overrides (for driver-specific parameters)
        cfg = self.env_config.copy()
        if override_env_config:
            cfg.update(override_env_config)

        # --- Physical Parameters ---
        self.capacity        = cfg.get('capacity', 60.0)
        self.eta             = cfg.get('eta', 0.95)
        self.max_power_base  = cfg.get('max_power', 11.0)
        self.dt              = cfg.get('dt', 1.0)
        
        # --- CC-CV Profile Coefficient ---
        self.alpha_constraint = cfg.get('alpha_constraint', 0.05)

        # --- User Requirements ---
        self.soc_req     = cfg.get('soc_req', 0.9)
        self.initial_soc = cfg.get('initial_soc', 0.2)
        self.t_dep       = cfg.get('t_dep', 12)

        # --- State Variables ---
        self.soc          = self.initial_soc
        self.current_step = 0

    def _get_max_power(self, soc):
        """
        Non-linear CC-CV charging constraint.
        P_max = ū · (1 − α · SOC)
        """
        return self.max_power_base * (1 - self.alpha_constraint * soc)

    def get_state(self, grid_signal, voltage_dev, price_forecast, ev_total_mw=0.0, delta_ev_mw=0.0):
        """
        Constructs and normalises the state vector s_{i,t}.
        s = [SOC, t_sin, t_cos, t_remaining, grid, voltage, ev_total, ev_delta, *prices]
        [cite: 85, 121]
        """
        s_soc = self.soc

        # Cyclic time encoding (captures hour-of-day periodicity)
        hour_of_day = self.current_step % 24
        t_sin = math.sin(2 * math.pi * hour_of_day / 24.0)
        t_cos = math.cos(2 * math.pi * hour_of_day / 24.0)

        # Remaining time, normalised to [0, 1] over a 24-h window
        t_remaining_norm = (self.t_dep - self.current_step) / 24.0

        # Grid & voltage — clipped and scaled
        s_grid = np.clip(grid_signal, 0.0, 1.0)
        s_volt = np.clip(voltage_dev * 10.0, -1.0, 1.0)

        # Price forecast (normalised; assumed max ≈ $0.50/kWh)
        s_prices = [p / 0.50 for p in price_forecast]

        # Aggregate EV load signals (broadcast from EdgeAggregator)
        s_ev_total = np.clip(ev_total_mw / 0.25,  0.0,  1.0)
        s_ev_delta = np.clip(delta_ev_mw / 0.10, -1.0,  1.0)

        state = np.array([
            s_soc, t_sin, t_cos, t_remaining_norm,
            s_grid, s_volt, s_ev_total, s_ev_delta
        ] + s_prices, dtype=np.float32)

        return state

    def step(self, action_power, grid_signal, voltage_dev, price_current):
        """
        Execute one MDP step.

        Args:
            action_power  : P_{i,t} (kW) — raw agent output scaled to physical range
            grid_signal   : λ_grid ∈ [0, 1] — congestion level
            voltage_dev   : Δv (p.u.) — voltage deviation
            price_current : electricity price ($/kWh)

        Returns:
            total_reward (float) : RL reward
            done         (bool)  : episode finished flag
            soc          (float) : updated state of charge
            energy_cost  (float) : economic cost for this step (≥ 0, charging only)
        """
        # ── 1. Physics & Constraints ──────────────────────────────────────────
        p_max_phys = self._get_max_power(self.soc)

        # Clip to feasible range; negative = V2G discharge
        p_act = np.clip(action_power, -p_max_phys, p_max_phys)

        # Energy transfer (kWh)
        energy_transfer = p_act * self.dt

        # Battery dynamics: SOC_{t+1} = SOC_t + (η · P · Δt) / C  [cite: 13]
        delta_soc = (self.eta * energy_transfer) / self.capacity
        prev_soc  = self.soc
        self.soc  = np.clip(self.soc + delta_soc, 0.0, 1.0)
        self.current_step += 1

        # ── 2. Economic Cost (tracking metric, separate from reward) ──────────
        # Only positive transfers (charging) incur real cost;
        # V2G discharge revenue is conservatively ignored here.
        energy_cost = max(0.0, energy_transfer) * price_current

        # ── 3. Structured Reward ──────────────────────────────────────────────
        total_reward, done = compute_reward(
            soc=self.soc, 
            prev_soc=prev_soc, 
            soc_req=self.soc_req, 
            t_dep=self.t_dep, 
            current_step=self.current_step,
            energy_transfer=energy_transfer, 
            price_current=price_current, 
            grid_signal=grid_signal,
            reward_config=self.reward_config
        )

        return total_reward, done, self.soc, energy_cost









