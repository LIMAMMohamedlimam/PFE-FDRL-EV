"""
profile_sac.py — Lightweight profiling for SAC agent performance
=================================================================
Measures wall-clock time for replay buffer operations and _learn() calls.

Run:  python3 scripts/profile_sac.py
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.SACAgent import SACAgent, ReplayBuffer

INPUT_DIM = 13
N_FILL = 3000       # transitions to fill buffer past warmup
N_LEARN = 200       # _learn() calls to benchmark
BATCH_SIZE = 256

print("=" * 60)
print("  SAC Performance Profiling")
print("=" * 60)

# --- 1. ReplayBuffer push + sample benchmark ---
print("\n--- ReplayBuffer Benchmark ---")
buf = ReplayBuffer(capacity=100_000)

# Push benchmark
t0 = time.perf_counter()
for _ in range(N_FILL):
    s = np.random.randn(INPUT_DIM).astype(np.float32)
    a = np.random.uniform(-1, 1)
    r = np.random.randn()
    ns = np.random.randn(INPUT_DIM).astype(np.float32)
    d = 0.0
    buf.push(s, a, r, ns, d)
t_push = time.perf_counter() - t0
print(f"  Push {N_FILL} transitions: {t_push*1000:.1f} ms ({t_push/N_FILL*1e6:.1f} µs/push)")

import torch
device = torch.device('cpu')

# Sample benchmark
t0 = time.perf_counter()
for _ in range(N_LEARN):
    buf.sample(BATCH_SIZE, device)
t_sample = time.perf_counter() - t0
print(f"  Sample {N_LEARN}x batch({BATCH_SIZE}): {t_sample*1000:.1f} ms ({t_sample/N_LEARN*1000:.2f} ms/sample)")

# --- 2. Full _learn() benchmark ---
print("\n--- SAC _learn() Benchmark ---")
agent = SACAgent(input_dim=INPUT_DIM, action_dim=1)
print(f"  Device: {agent.device}")

# Fill buffer past warmup
for _ in range(N_FILL):
    s = np.random.randn(INPUT_DIM).astype(np.float32)
    a = float(np.random.uniform(-1, 1))
    r = float(np.random.randn())
    ns = np.random.randn(INPUT_DIM).astype(np.float32)
    agent.buffer.push(s, a, r, ns, 0.0)

# Warm up JIT / caches
agent._learn()
agent._learn()

t0 = time.perf_counter()
for _ in range(N_LEARN):
    agent._learn()
t_learn = time.perf_counter() - t0
print(f"  {N_LEARN} _learn() calls: {t_learn*1000:.1f} ms ({t_learn/N_LEARN*1000:.2f} ms/call)")

# --- 3. get_action benchmark ---
print("\n--- get_action() Benchmark ---")
N_ACTIONS = 2000
state = np.random.randn(INPUT_DIM).astype(np.float32)

t0 = time.perf_counter()
for _ in range(N_ACTIONS):
    agent.get_action(state, eval_mode=False)
t_act = time.perf_counter() - t0
print(f"  {N_ACTIONS} get_action() calls: {t_act*1000:.1f} ms ({t_act/N_ACTIONS*1000:.2f} ms/call)")

print(f"\n{'=' * 60}")
print(f"  DONE — all timings are wall-clock on {agent.device}")
print(f"{'=' * 60}")
