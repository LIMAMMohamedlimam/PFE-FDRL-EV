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

class EVClientEnv:
    """
    Handles the physical modeling and MDP formulation for a single EV.
    Updated for PPO Stability: Normalized States & Scaled Rewards.
    """
    def __init__(self, config):
        # --- Physical Parameters ---
        # [cite: 13-17]
        self.capacity = config.get('capacity', 60.0)    # C_i (kWh)
        self.eta = config.get('efficiency', 0.95)       # eta
        self.max_power_base = config.get('max_power', 11.0) # u_bar (kW)
        self.dt = config.get('dt', 1.0)                 # Delta t (hours)
        
        # --- User Requirements ---
        # [cite: 24, 25]
        self.soc_req = config.get('soc_req', 0.9)       # Target SOC
        self.initial_soc = config.get('initial_soc', 0.2)
        self.t_dep = config.get('t_dep', 12)            # Duration (hours)
        
        # --- State Variables ---
        self.soc = self.initial_soc
        self.current_step = 0
        
        # --- Reward Weights (Scaled for PPO) ---
        # We reduced these significantly from 100.0 to prevent gradient explosion
        self.w_satisfaction = 20.0  # Bonus for hitting target
        self.w_grid = 1.0           # Penalty for grid stress
        self.w_cost = 1.0           # Penalty for electricity cost
        
    def _get_max_power(self, soc):
        """
        Calculates P_max based on non-linear charging constraint.
        Formula: P_max = u_bar * (1 - alpha * SOC) 
        [cite: 22]
        """
        # Alpha coefficient represents CC-CV curve (Constant Current - Constant Voltage)
        alpha_constraint = 0.05 
        return self.max_power_base * (1 - alpha_constraint * soc)

    def get_state(self, grid_signal, voltage_dev, price_forecast):
        """
        Constructs and NORMALIZES the state vector s_{i,t}.
        State: [SOC, Norm_Time_Sin, Norm_Time_Cos, Norm_Grid, Norm_Voltage, Norm_Prices...]
        [cite: 85, 121]
        """
        # 1. SOC is already [0, 1] [cite: 13]
        s_soc = self.soc 
        
        # 2. Time Encoding (Cyclic)
        # We map the 24h cycle to Sin/Cos to help NN understand periodicity
        # t_dep is usually relative, but here we treat current_step as hour of day context
        hour_of_day = self.current_step % 24
        t_sin = math.sin(2 * math.pi * hour_of_day / 24.0)
        t_cos = math.cos(2 * math.pi * hour_of_day / 24.0)
        
        # 3. Remaining Time (Normalized)
        # 0.0 means "right now", 1.0 means "plenty of time" (assuming max stay ~24h)
        t_remaining_norm = (self.t_dep - self.current_step) / 24.0
        
        # 4. Grid & Voltage (Clipped & Scaled)
        # Grid signal typically 0 (safe) to 1 (congested). We clip to be safe.
        s_grid = np.clip(grid_signal, 0.0, 1.0)
        # Voltage dev is small (e.g. 0.05 p.u.). We scale it up so NN sees it.
        s_volt = np.clip(voltage_dev * 10.0, -1.0, 1.0)
        
        # 5. Price Forecast (Normalized)
        # Assuming max price is roughly $0.50/kWh. 
        # We want input to be roughly 0.0 to 1.0
        s_prices = [p / 0.50 for p in price_forecast]
        
        # Combine
        state = np.array([
            s_soc, 
            t_sin, 
            t_cos, 
            t_remaining_norm, 
            s_grid, 
            s_volt
        ] + s_prices, dtype=np.float32)
        
        return state

    def step(self, action_power, grid_signal, voltage_dev, price_current):
        """
        Executes one step.
        Input: action_power (kW) -> Unscaled continuous action from PPO
        [cite: 10-12]
        """
        # --- 1. Physics & Constraints ---
        
        # Get physical limit [cite: 22]
        p_max_phys = self._get_max_power(self.soc)
        
        # Clip action to physical limits [-Pmax, +Pmax]
        # (Negative = V2G/Discharge, Positive = Charge)
        p_act = np.clip(action_power, -p_max_phys, p_max_phys)
        
        # Calculate Energy Transfer (kWh)
        energy_transfer = p_act * self.dt
        
        # Update SOC 
        # SOC_{t+1} = SOC_t + (eta * P * dt) / C
        delta_soc = (self.eta * energy_transfer) / self.capacity
        
        # Check boundaries [0, 1]
        prev_soc = self.soc
        self.soc = np.clip(self.soc + delta_soc, 0.0, 1.0)
        
        self.current_step += 1
        
        # --- 2. Reward Calculation (Scaled for RL Stability) ---
        # 
        
        # A. Continuous Progress Reward (The Carrot)
        # We reward INCREASING soc if we are below target.
        # This fixes the "agent does nothing" bug.
        reward_soc_gain = 0.0
        if self.soc < self.soc_req:
            # +5.0 reward for every 10% SOC gained (Scale: ~0 to +1.0 per step)
            reward_soc_gain = (self.soc - prev_soc) * 50.0 
            if reward_soc_gain < 0: reward_soc_gain = 0 # Don't reward discharging yet
            
        # B. Cost Penalty (The Stick)
        # Price ~0.15, Energy ~10kWh -> Cost ~1.5. 
        # We negate it.
        reward_cost = -(price_current * energy_transfer) * self.w_cost
        
        # C. Grid Penalty
        # If grid_signal is high (congestion), penalize heavy charging
        reward_grid = -self.w_grid * abs(energy_transfer) * grid_signal
        
        # D. Terminal Satisfaction (The Goal)
        done = False
        reward_terminal = 0.0
        
        if self.current_step >= self.t_dep:
            done = True
            
            if self.soc >= self.soc_req:
                # Big Bonus for success
                reward_terminal = 10.0
            else:
                # Soft Penalty for failure (Not -10,000!)
                # Ex: Missing 20% SOC -> 0.2 * 20.0 = -4.0
                reward_terminal = -self.w_satisfaction * (self.soc_req - self.soc)

        # Total Reward
        # Expected range per step: -2 to +2
        # Expected range per episode: -10 to +20
        total_reward = reward_soc_gain + reward_cost + reward_grid + reward_terminal
        
        return total_reward, done, self.soc





















# import numpy as np
# import math

# class EVClientEnv:
#     """
#     Handles the physical modeling and MDP formulation for a single EV.
#     Strategy: Penalty Minimization (Lagrangian approach).
#     """
#     def __init__(self, config):
#         # --- Physical Parameters ---
#         self.capacity = config.get('capacity', 60.0)    # [cite: 16]
#         self.eta = config.get('efficiency', 0.95)       # [cite: 15]
#         self.max_power_base = config.get('max_power', 11.0) # [cite: 22]
#         self.dt = config.get('dt', 1.0)                 # [cite: 17]
        
#         # --- User Requirements ---
#         self.soc_req = config.get('soc_req', 0.9)       # [cite: 25]
#         self.initial_soc = config.get('initial_soc', 0.2)
#         self.t_dep = config.get('t_dep', 12)            # [cite: 39]
        
#         # --- State Variables ---
#         self.soc = self.initial_soc
#         self.current_step = 0
        
#         # --- Penalty Weights (Hyperparameters) ---
#         # "alpha" from Eq (11) in your PDF [cite: 91]
#         # We assume alpha ~ 100.0, but we scale it down for PPO stability.
#         self.w_tracking = 10.0   # Penalty for being empty (SOC < SOC_req)
#         self.w_cost = 2.0        # Penalty for high energy prices
#         self.w_grid = 1.5        # Penalty for grid congestion
#         self.w_terminal = 50.0   # Big final penalty if we fail

#     def _get_max_power(self, soc):
#         """
#         Calculates P_max based on non-linear charging constraint[cite: 22].
#         """
#         alpha_constraint = 0.05 
#         return self.max_power_base * (1 - alpha_constraint * soc)

#     def get_state(self, grid_signal, voltage_dev, price_forecast):
#         """
#         Constructs and Normalizes state s_{i,t}[cite: 85].
#         """
#         # 1. SOC [0, 1]
#         s_soc = self.soc 
        
#         # 2. Cyclic Time (Sin/Cos)
#         hour_of_day = self.current_step % 24
#         t_sin = math.sin(2 * math.pi * hour_of_day / 24.0)
#         t_cos = math.cos(2 * math.pi * hour_of_day / 24.0)
        
#         # 3. Remaining Time (Normalized)
#         t_remaining_norm = (self.t_dep - self.current_step) / 24.0
        
#         # 4. Grid Signals (Clipped)
#         s_grid = np.clip(grid_signal, 0.0, 1.0)
#         s_volt = np.clip(voltage_dev * 10.0, -1.0, 1.0) # Scaled for visibility
        
#         # 5. Price Forecast (Normalized)
#         s_prices = [p / 0.50 for p in price_forecast]
        
#         state = np.array([
#             s_soc, 
#             t_sin, 
#             t_cos, 
#             t_remaining_norm, 
#             s_grid, 
#             s_volt
#         ] + s_prices, dtype=np.float32)
        
#         return state

#     def step(self, action_power, grid_signal, voltage_dev, price_current):
#         """
#         Executes one step.
#         Minimizes: Cost + Grid_Stress + Deviation_from_Target
#         """
#         # --- 1. Physics & Constraints ---
#         p_max_phys = self._get_max_power(self.soc)
        
#         # Clip action to physical limits [-Pmax, +Pmax] [cite: 21]
#         p_act = np.clip(action_power, -p_max_phys, p_max_phys)
        
#         # Energy Transfer
#         energy_transfer = p_act * self.dt # kWh
        
#         # Update SOC [cite: 12]
#         delta_soc = (self.eta * energy_transfer) / self.capacity
#         prev_soc = self.soc
#         self.soc = np.clip(self.soc + delta_soc, 0.0, 1.0)
        
#         self.current_step += 1
        
#         # --- 2. Penalty-Based Reward Calculation ---
        
#         # A. Tracking Penalty (Dense) [cite: 91]
#         # Instead of waiting for the end, we penalize the gap at EVERY step.
#         # This creates a "gradient" of pain that forces the agent to charge.
#         # R_track = - w * (Target - Current)^2
#         soc_gap = max(0, self.soc_req - self.soc)
#         reward_tracking = -self.w_tracking * (soc_gap ** 2)
        
#         # B. Energy Cost Penalty [cite: 89]
#         # We penalize spending money.
#         # If Price is high, this penalty is high -> Agent pauses charging.
#         # If Price is low, this penalty is low -> Agent charges to reduce 'tracking penalty'.
#         cost = price_current * energy_transfer
#         reward_cost = -self.w_cost * cost 
        
#         # C. Grid Congestion Penalty [cite: 87, 100]
#         # Penalize charging during congestion (grid_signal > 0)
#         reward_grid = -self.w_grid * abs(energy_transfer) * grid_signal
        
#         # D. Terminal Penalty (The "Failure" Check)
#         done = False
#         reward_terminal = 0.0
        
#         if self.current_step >= self.t_dep:
#             done = True
#             # If we missed the target, apply a final heavy penalty
#             # This corresponds to the user being angry.
#             if self.soc < self.soc_req:
#                 reward_terminal = -self.w_terminal * ((self.soc_req - self.soc) ** 2)
#             else:
#                 # Optional: Small positive value to mark "perfect 0 penalty" state
#                 reward_terminal = 10.0 

#         # Total Reward (All Negative = "Cost Function")
#         # Typical Value: -5.0 (Early) -> -0.1 (Late/Success)
#         total_reward = reward_tracking + reward_cost + reward_grid + reward_terminal
        
#         return total_reward, done, self.soc