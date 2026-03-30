"""
lora.py — Low-Rank Adaptation (LoRA) utilities
================================================
Reusable module providing parameter-efficient fine-tuning for PyTorch models.

LoRA freezes the pre-trained base weights W₀ and injects trainable low-rank
decomposition matrices so that the effective weight becomes:

    W_eff = W₀ + (B @ A) * (alpha / rank)

where A ∈ ℝ^(rank × in_features), B ∈ ℝ^(out_features × rank).

Key design choices:
  - LoRALinear wraps an existing nn.Linear, preserving bias if present.
  - apply_lora() walks a model tree and replaces targeted layers in-place.
  - get_lora_state_dict() / load_lora_state_dict() scope FL aggregation
    to only the tiny LoRA matrices, reducing communication cost.
  - The module is framework-agnostic within this project: both SACAgent
    and PPOAgent use the same apply_lora() → get_lora_parameters() API.

References:
  Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021
  https://arxiv.org/abs/2106.09685
"""

import os
import math
import torch
import torch.nn as nn
from utils.config_loader import get_config


# ---------------------------------------------------------------------------
# Core LoRA Layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank adapter.

    The base weight is frozen; only lora_A and lora_B are trainable.
    Forward:  y = x @ W_frozen^T + bias + x @ A^T @ B^T * scaling
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        """
        Args:
            base_linear: The original nn.Linear layer to wrap.
            rank:        Rank of the low-rank decomposition (r).
            alpha:       Scaling factor; effective scale = alpha / rank.
        """
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # ── Frozen base weight ──
        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # ── Trainable LoRA matrices ──
        # A: (rank, in_features) — Kaiming-uniform init for stable gradients
        # B: (out_features, rank)  — zero init so LoRA starts as identity
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear pass (frozen)
        base_out = nn.functional.linear(x, self.weight, self.bias)

        # LoRA delta: x @ A^T @ B^T * scaling
        lora_out = nn.functional.linear(
            nn.functional.linear(x, self.lora_A),  # x @ A^T  → (*, rank)
            self.lora_B,                            # → (*, out_features)
        ) * self.scaling

        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.2f}, "
            f"bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# Model-level utilities
# ---------------------------------------------------------------------------

def apply_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    target_modules: list = None,
) -> nn.Module:
    """
    Walk *model* and replace every nn.Linear whose name contains any
    substring in *target_modules* with a LoRALinear wrapper.

    Operates **in-place** and returns the model for convenience.

    Args:
        model:          PyTorch model to modify.
        rank:           LoRA rank (r).
        alpha:          LoRA scaling factor.
        target_modules: List of substrings to match against layer names.
                        If None, ALL nn.Linear layers are wrapped.

    Returns:
        The same model (modified in-place).
    """
    if target_modules is None:
        target_modules = []  # empty → match everything

    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Match if target_modules is empty (match all) or name contains any target
            if not target_modules or any(t in name for t in target_modules):
                replacements[name] = LoRALinear(module, rank=rank, alpha=alpha)

    # Perform the actual replacement in the model tree
    for name, lora_layer in replacements.items():
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        leaf_name = parts[-1]
        if leaf_name.isdigit():
            parent[int(leaf_name)] = lora_layer
        else:
            setattr(parent, leaf_name, lora_layer)

    return model


def get_lora_parameters(model: nn.Module):
    """
    Return an iterator over only the trainable LoRA parameters.
    Use this to build the optimizer when LoRA is active.
    """
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            yield param


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a model (trainable or all)."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_lora_state_dict(model: nn.Module, prefix: str = '') -> dict:
    """
    Extract only LoRA weights from the model's state dict.
    Returns numpy arrays (CPU) for FL aggregation compatibility.

    Keys are prefixed with *prefix* (e.g. 'actor.' or 'critic.') so
    that SACAgent can namespace actor vs critic LoRA weights.
    """
    lora_sd = {}
    for name, param in model.state_dict().items():
        if 'lora_' in name:
            key = f"{prefix}{name}" if prefix else name
            lora_sd[key] = param.cpu().numpy()
    return lora_sd


def load_lora_state_dict(
    model: nn.Module,
    state_dict: dict,
    prefix: str = '',
    device: torch.device = None,
) -> None:
    """
    Load LoRA weights into a model from a (possibly prefixed) state dict.
    Non-LoRA keys in *state_dict* are silently ignored.
    """
    model_sd = model.state_dict()
    prefix_len = len(prefix)

    for key, value in state_dict.items():
        # Strip prefix if present
        if prefix and key.startswith(prefix):
            model_key = key[prefix_len:]
        else:
            model_key = key

        if 'lora_' in model_key and model_key in model_sd:
            import numpy as np
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
            else:
                tensor = value
            if device is not None:
                tensor = tensor.to(device)
            model_sd[model_key] = tensor

    model.load_state_dict(model_sd)


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------

def get_lora_config() -> dict:
    """
    Load LoRA configuration from lora.yaml, with environment variable override.

    The USE_LORA environment variable overrides the YAML 'enabled' flag:
        USE_LORA=true  → LoRA enabled
        USE_LORA=false → LoRA disabled
        (unset)        → use lora.yaml value
    """
    try:
        cfg = get_config('lora')
    except FileNotFoundError:
        cfg = {}

    # Environment variable override for the 'enabled' flag
    env_val = os.environ.get('USE_LORA', '').lower()
    if env_val in ('true', '1', 'yes'):
        cfg['enabled'] = True
    elif env_val in ('false', '0', 'no'):
        cfg['enabled'] = False

    # Defaults
    cfg.setdefault('enabled', False)
    cfg.setdefault('rank', 4)
    cfg.setdefault('alpha', 8)
    cfg.setdefault('target_modules', [])

    return cfg
