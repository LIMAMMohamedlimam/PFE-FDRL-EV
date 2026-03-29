import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
from agents.BaseAgent import BaseAgent
from utils.device_utils import get_device, device_info
from utils.config_loader import get_config
from utils.lora import (
    apply_lora, get_lora_parameters, get_lora_state_dict,
    load_lora_state_dict, get_lora_config, count_parameters,
)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular buffer for SAC off-policy learning."""

    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        """Sample a random batch and return tensors on the given device."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Neural Network Components
# ---------------------------------------------------------------------------

class GaussianPolicy(nn.Module):
    """
    Stochastic Gaussian Actor for SAC.
    Outputs a tanh-squashed action in [-1, 1].
    """
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        """Sample action; return (action, log_prob)."""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()                      # Reparameterisation trick
        action = torch.tanh(z)

        # Log-prob with tanh-squash correction
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        """Return mean action (for evaluation)."""
        mu, _ = self.forward(state)
        return torch.tanh(mu)


class TwinQNetwork(nn.Module):
    """Two independent Q-networks (clipped double-Q trick)."""

    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SACAgent(BaseAgent):
    """
    Soft Actor-Critic (SAC) agent with automatic entropy tuning.
    Action space: continuous [-1, 1], scaled to P_max in the environment.
    Follows the same BaseAgent interface as PPO/Q-Learning for FL compatibility.

    GPU: Networks and tensors are automatically placed on the best available
    device (CUDA → MPS → CPU) via device_utils.get_device().

    LoRA: When use_lora=True, base network weights are frozen and only
    low-rank adapter matrices are trainable. This reduces the number of
    trainable parameters and communication cost in federated learning.
    """

    def __init__(self, input_dim, action_dim=1, alpha_init=0.2, use_lora=False, **kwargs):
        # --- Device ---
        self.device = get_device()
        self.action_dim = action_dim

        # --- LoRA flag: constructor arg > env var > yaml ---
        self.use_lora = use_lora
        if not self.use_lora:
            lora_cfg = get_lora_config()
            self.use_lora = lora_cfg.get('enabled', False)
        self._lora_cfg = get_lora_config() if self.use_lora else {}

        # --- Load Configs ---
        cfg = get_config('sac')
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.005)
        self.batch_size = cfg.get('batch_size', 256)
        self.warmup_steps = cfg.get('warmup_steps', 2000)
        
        lr = float(cfg.get('lr', 3e-4))
        hidden_dim = cfg.get('hidden_dim', 128)
        buffer_capacity = cfg.get('buffer_capacity', 100000)
        self.target_entropy_scale = cfg.get('target_entropy_scale', -0.5)
        self.update_every = cfg.get('update_every', 1)

        # --- Networks (moved to device) ---
        self.actor = GaussianPolicy(input_dim, action_dim, hidden_dim).to(self.device)
        self.critic = TwinQNetwork(input_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinQNetwork(input_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # --- Apply LoRA if enabled ---
        if self.use_lora:
            lora_rank = self._lora_cfg.get('rank', 4)
            lora_alpha = self._lora_cfg.get('alpha', 8)
            lora_targets = self._lora_cfg.get('target_modules', [])

            apply_lora(self.actor, rank=lora_rank, alpha=lora_alpha, target_modules=lora_targets)
            apply_lora(self.critic, rank=lora_rank, alpha=lora_alpha, target_modules=lora_targets)
            # Target critic must mirror the LoRA structure
            apply_lora(self.critic_target, rank=lora_rank, alpha=lora_alpha, target_modules=lora_targets)
            self.critic_target.load_state_dict(self.critic.state_dict())

            # Move LoRA layers to device (they were built on CPU from base weights)
            self.actor = self.actor.to(self.device)
            self.critic = self.critic.to(self.device)
            self.critic_target = self.critic_target.to(self.device)

            actor_trainable = count_parameters(self.actor, trainable_only=True)
            critic_trainable = count_parameters(self.critic, trainable_only=True)
            print(f"[SAC] LoRA ENABLED  (rank={lora_rank}, α={lora_alpha})")
            print(f"      Actor  trainable params: {actor_trainable:,}")
            print(f"      Critic trainable params: {critic_trainable:,}")

        # --- Optimisers ---
        # When LoRA is active, only optimize LoRA parameters
        if self.use_lora:
            self.actor_optim = optim.Adam(get_lora_parameters(self.actor), lr=lr)
            self.critic_optim = optim.Adam(get_lora_parameters(self.critic), lr=lr)
        else:
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # --- Automatic entropy tuning (log_alpha on device) ---
        self.target_entropy = self.target_entropy_scale * float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = alpha_init
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

        # --- Replay buffer ---
        self.buffer = ReplayBuffer(buffer_capacity)

        # --- Step counter ---
        self.total_steps = 0

    # ----- BaseAgent interface -----

    def get_action(self, state, eval_mode=False):
        """Return a scalar action in [-1, 1]."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if eval_mode:
                action = self.actor.deterministic(state_t)
            else:
                action, _ = self.actor.sample(state_t)
        return action.squeeze().item()

    def update(self, state, action, reward, next_state, done=False):
        """Store transition and learn if enough data is available."""
        self.buffer.push(state, action, reward, next_state, float(done))
        self.total_steps += 1

        if len(self.buffer) < self.warmup_steps:
            return
        if self.total_steps % self.update_every != 0:
            return

        self._learn()

    def get_parameters(self):
        """Return model state dicts as numpy (for FL aggregation).

        When LoRA is active, returns ONLY the LoRA adapter weights,
        reducing communication cost in federated learning.
        Always uses .cpu() so FL aggregation (numpy) works regardless of device.
        """
        if self.use_lora:
            # Return only LoRA weights with actor./critic. prefixes
            params = {}
            params.update(get_lora_state_dict(self.actor, prefix='actor.'))
            params.update(get_lora_state_dict(self.critic, prefix='critic.'))
            return params
        else:
            params = {}
            for k, v in self.actor.state_dict().items():
                params[f"actor.{k}"] = v.cpu().numpy()
            for k, v in self.critic.state_dict().items():
                params[f"critic.{k}"] = v.cpu().numpy()
            return params

    def set_parameters(self, parameters):
        """Load aggregated parameters (for FL aggregation).

        When LoRA is active, loads ONLY LoRA adapter weights.
        """
        if self.use_lora:
            # Split by prefix and load LoRA weights
            actor_params = {k: v for k, v in parameters.items() if k.startswith('actor.')}
            critic_params = {k: v for k, v in parameters.items() if k.startswith('critic.')}

            if actor_params:
                load_lora_state_dict(self.actor, actor_params, prefix='actor.', device=self.device)
            if critic_params:
                load_lora_state_dict(self.critic, critic_params, prefix='critic.', device=self.device)
                load_lora_state_dict(self.critic_target, critic_params, prefix='critic.', device=self.device)
        else:
            actor_sd = {}
            critic_sd = {}
            for k, v in parameters.items():
                tensor = torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray) else v.to(self.device)
                if k.startswith("actor."):
                    actor_sd[k[len("actor."):]] = tensor
                elif k.startswith("critic."):
                    critic_sd[k[len("critic."):]] = tensor

            if actor_sd:
                self.actor.load_state_dict(actor_sd)
            if critic_sd:
                self.critic.load_state_dict(critic_sd)
                self.critic_target.load_state_dict(critic_sd)

    # ----- Internal SAC learning -----

    def _learn(self):
        if len(self.buffer) < self.batch_size:
            return
        # All tensors already on self.device from ReplayBuffer.sample()
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, self.device
        )

        # ---- 1. Critic loss (clipped double-Q) ----
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            td_target = rewards + self.gamma * (1 - dones) * q_target

        q1_pred, q2_pred = self.critic(states, actions)
        critic_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # ---- 2. Actor loss ----
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # ---- 3. Entropy temperature (alpha) auto-tune ----
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item()

        # ---- 4. Soft-update target critic ----
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
