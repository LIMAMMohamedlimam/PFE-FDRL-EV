# utils/reward_functions.py
import numpy as np

def compute_reward(soc, prev_soc, soc_req, t_dep, current_step,
                   energy_transfer, price_current, grid_signal,
                   reward_config):
    """
    Computes the structured 5-component reward for the EV charging environment.
    All scalar weights are injected via reward_config (reward.yaml).
    
    Returns:
        total_reward (float): The RL reward signal
        done (bool): Whether the episode is finished
    """
    # Load weights from config
    w_dist = reward_config.get('w_target_tracking', 2.0)
    w_prog = reward_config.get('w_progress', 5.0)
    w_cost = reward_config.get('w_cost', 0.5)
    w_grid = reward_config.get('w_grid', 0.3)
    w_terminal_bonus = reward_config.get('terminal_success_bonus', 15.0)
    w_terminal_penalty = reward_config.get('terminal_failure_weight', 25.0)

    # A. Distance penalty - continuous pull toward target
    soc_deficit = max(0.0, soc_req - soc)
    reward_distance = -w_dist * soc_deficit

    # B. SOC progress reward - positive reinforcement for charging
    soc_delta = soc - prev_soc
    reward_progress = w_prog * soc_delta if (soc_delta > 0 and soc < soc_req) else 0.0

    # C. Cost penalty - moderate disincentive for expensive energy
    reward_cost = -w_cost * max(0.0, energy_transfer) * price_current

    # D. Grid penalty - avoid charging during congestion
    reward_grid = -w_grid * abs(energy_transfer) * grid_signal

    # E. Terminal reward - evaluated at departure
    done = False
    reward_terminal = 0.0
    if current_step >= t_dep:
        done = True
        if soc >= soc_req:
            reward_terminal = w_terminal_bonus
        else:
            reward_terminal = -w_terminal_penalty * (soc_req - soc)

    total_reward = reward_distance + reward_progress + reward_cost + reward_grid + reward_terminal
    
    return total_reward, done
