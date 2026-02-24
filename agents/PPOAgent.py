import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from agents.BaseAgent import BaseAgent
from utils.device_utils import get_device, device_info

# 

class ActorCritic(nn.Module):
    """
    Continuous Actor-Critic Network.
    - Actor: Outputs Mean (mu) and Std Dev (sigma) for a Gaussian distribution.
    - Critic: Outputs Value estimate V(s).
    """
    def __init__(self, input_dim, action_dim, hidden_dim=64, std_init=0.5):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor (optional, usually separate is more stable)
        
        # --- CRITIC (Value Function) ---
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # --- ACTOR (Policy Function) ---
        # 1. Mean (mu) - Determines the center of the action
        self.actor_mu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 2. Standard Deviation (sigma) - Determines exploration width
        # We learn log_std to ensure stability (std = exp(log_std))
        self.actor_log_std = nn.Parameter(torch.ones(1, action_dim) * np.log(std_init))

    def act(self, state):
        action_mu = self.actor_mu(state)
        action_std = self.actor_log_std.exp().expand_as(action_mu)
        dist = Normal(action_mu, action_std)

        z = dist.sample()
        action = torch.tanh(z)

        # Tanh-squash correction
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return action.item(), log_prob, entropy
    
    def evaluate(self, state, action):
        action_mu = self.actor_mu(state)
        action_std = self.actor_log_std.exp().expand_as(action_mu)
        dist = Normal(action_mu, action_std)

        # Inverse tanh (atanh) to recover pre-squash action
        action_clamped = torch.clamp(action, -0.999999, 0.999999)
        z = torch.atanh(action_clamped)

        log_prob = dist.log_prob(z) - torch.log(1 - action_clamped.pow(2) + 1e-6)
        action_logprobs = log_prob.sum(dim=-1)

        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

class PPOAgent(BaseAgent):
    """
    Continuous PPO Agent.
    Action Space: [-1.0, 1.0] (Scaled to P_max in environment)

    GPU: Networks and tensors are automatically placed on the best available
    device (CUDA → MPS → CPU) via device_utils.get_device().
    """
    def __init__(self, input_dim, action_dim=1, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4, update_timestep=2000):
        # --- Device ---
        self.device = get_device()

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.update_timestep = update_timestep
        
        # Action dim is usually 1 (Power) for EV charging
        self.policy = ActorCritic(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(input_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_is_terminals = []
        
        self.current_log_prob = None
        self.time_step = 0

    def get_action(self, state, eval_mode=False):
        """
        Returns a continuous action value in range [-1, 1].
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if eval_mode:
            with torch.no_grad():
                # In eval mode, use the Mean directly (Deterministic)
                mu = self.policy.actor_mu(state_tensor)
                action = torch.tanh(mu).item()
            return action

        # Training mode: Sample from Normal Distribution
        with torch.no_grad():
            action, log_prob, _ = self.policy_old.act(state_tensor)
        
        self.current_log_prob = log_prob
        return action

    def update(self, state, action, reward, next_state, done=False):
        """
        Standard PPO buffer storage. Tensors stored on device.
        """
        self.buffer_states.append(torch.FloatTensor(state).to(self.device))
        self.buffer_actions.append(torch.FloatTensor([action]).to(self.device))
        self.buffer_logprobs.append(self.current_log_prob)
        self.buffer_rewards.append(reward)
        self.buffer_is_terminals.append(done)

        self.time_step += 1

        if self.time_step % self.update_timestep == 0:
            self._ppo_update()

    def _ppo_update(self):
        # 1. Monte Carlo Estimate of Rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer_rewards), reversed(self.buffer_is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if rewards.std() > 1e-5:
             rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 2. Convert list to tensors (already on device)
        old_states = torch.stack(self.buffer_states).detach()
        old_actions = torch.stack(self.buffer_actions).detach().squeeze()
        old_logprobs = torch.stack(self.buffer_logprobs).detach().squeeze()

        # Handle simplified shape case (if batch size is small or 1D)
        if len(old_actions.shape) == 0: old_actions = old_actions.unsqueeze(0)
        if len(old_logprobs.shape) == 0: old_logprobs = old_logprobs.unsqueeze(0)

        # 3. Optimize policy
        for _ in range(self.K_epochs):
            # Reshape action to [Batch, 1] for the network
            actions_for_eval = old_actions.unsqueeze(1) 
            
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, actions_for_eval)
            
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()   
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_logprobs.clear()
        self.buffer_rewards.clear()
        self.buffer_is_terminals.clear()

    def get_parameters(self):
        """Always returns CPU numpy arrays so FL aggregation works regardless of device."""
        return {k: v.cpu().numpy() for k, v in self.policy.state_dict().items()}

    def set_parameters(self, parameters):
        new_state_dict = {k: torch.from_numpy(v).to(self.device) for k, v in parameters.items()}
        self.policy.load_state_dict(new_state_dict)
        self.policy_old.load_state_dict(new_state_dict)