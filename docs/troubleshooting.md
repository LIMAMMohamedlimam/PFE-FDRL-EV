# Troubleshooting & FAQ

---

## Installation

### `ImportError: No module named 'questionary'`

**Cause:** `questionary` is not listed in `requirement.txt` (only `pandapower` and `questionary` are listed, but pip may not have installed it).

**Fix:**
```bash
pip install questionary flask
```

---

### `ImportError: No module named 'pandapower'`

**Cause:** Virtual environment not activated, or `pip install -r requirement.txt` was not run.

**Fix:**
```bash
source venv/bin/activate
pip install -r requirement.txt
```

---

### PyTorch not installed / CUDA not detected

**Cause:** PyTorch is a transitive dependency and may not install a CUDA-enabled wheel automatically.

**Fix:** Install the correct wheel for your CUDA version from https://pytorch.org/get-started/locally/ before running pip install:

```bash
# Example for CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirement.txt
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## Training

### pandapower power-flow convergence warnings

**Symptom:** `pandapower.LoadflowNotConverged` warning during training; `GridEnv.step()` returns `lambda_grid = 2.0`.

**Cause:** At certain timesteps the combined EV + base load creates an infeasible power flow. This is physically realistic (network overload) and handled gracefully: the grid returns the maximum congestion signal.

**This is not a bug.** The agents learn to avoid causing overloads because `r_D = −w_D · |energy| · λ_grid` penalises charging during congestion.

**If the warnings are frequent:** Reduce `num_agents` or `max_power` in `configs/env.yaml`, or lower `base_load_mw` in the simulation loop.

---

### SAC loss becomes NaN

**Symptom:** Actor or critic loss becomes `nan`; training collapses.

**Likely causes and fixes:**

| Cause | Fix |
|-------|-----|
| Learning rate too high | Lower `lr` in `configs/sac.yaml` (try 1e-4) |
| Warmup too short | Increase `warmup_steps` (try 5 000) |
| Reward scale too large | Reduce `terminal_success_bonus` or `w_progress` in `reward.yaml` |
| Gradient explosion | The code clips gradients to 1.0 (`clip_grad_norm_`); check that LoRA layers also clip correctly |

Quick diagnostic:
```bash
python -c "
import torch
from agents.SACAgent import SACAgent
a = SACAgent(input_dim=13, action_dim=1)
import numpy as np
s = np.random.rand(13).astype(np.float32)
for _ in range(100):
    a.update(s, 0.5, 1.0, s, False)
print('No NaN in 100 updates')
"
```

---

### SAC agent not learning (reward stays flat)

**Symptom:** Training reward does not increase after many episodes.

**Possible causes:**

1. **Buffer not warm yet:** SAC does not update until `len(buffer) >= warmup_steps` (2 000 by default). In dev mode (5 agents, 20 episodes = 5 × 20 × 24 = 2 400 steps), learning only just begins. Use Training mode for a proper convergence curve.

2. **Target entropy too high / too low:** `target_entropy_scale = −0.5` means H̄ = −0.5. If your reward landscape requires more exploration, try −1.0.

3. **Reward not reaching the agent:** Check that `SACAgent.update()` is receiving non-zero rewards. Add a temporary print inside `update()` to verify.

4. **FL overwriting good weights:** If FL is enabled but the global model is not initialized before the first round, `set_parameters()` may load zeros. Ensure `FederatedServer.initialize()` is called with the first agent's parameters.

---

### Q-Learning agent performs poorly

This is expected. Tabular Q-Learning uses discrete actions {idle, half-charge, full-charge} and cannot represent fine-grained continuous control. It exists as a lower-bound baseline, not a competitive method. The state space is also large for a tabular approach, so the Q-table remains sparse.

---

### FedProx not improving over FedAvg

**Cause:** The proximal coefficient `mu_fedprox = 0.01` may be too small or too large. Also verify that `SACAgent.set_fedprox_global()` is being called after each FL round — without it, the proximal term is zero regardless of the config.

**Diagnostic:** Add `print(agent.mu_fedprox, agent._fedprox_global_params is not None)` after setting the global model.

---

### SWIFT selects 0 agents

**Symptom:** `SWIFTScheduler.select_clients()` returns an empty list.

**Cause:** All agents fail the eligibility filter (`t_remaining < min_stay_hours = 2`). This can happen at the end of an episode when most EVs are close to departure.

**Behaviour:** The scheduler has a safety fallback — if `eligible = ∅`, it falls back to `eligible = all agents`. An empty selection should never happen in normal operation.

**If it persists:** Check that `envs[i].t_dep` and `envs[i].current_step` are set correctly before `select_clients()` is called.

---

## Checkpoints

### Checkpoint load fails: key mismatch

**Symptom:** `RuntimeError: Missing key(s)` or `unexpected key(s)` when calling `agent.load_trained_model()`.

**Cause:** Mismatch between the checkpoint format (full vs LoRA) and the current agent configuration.

**Fix:** `load_trained_model()` auto-detects the format:
- Full checkpoints have an `'actor'` key pointing to a state dict.
- LoRA checkpoints have flat keys like `'actor.fc1.lora_A'`.

If you get a mismatch, ensure the agent is constructed with the same `use_lora` flag as when the checkpoint was saved:

```python
# For a LoRA checkpoint:
agent = SACAgent(input_dim=13, action_dim=1, use_lora=True)
agent.load_trained_model('results/trained_models/sac_agent_lora_0_....pth')

# For a full checkpoint:
agent = SACAgent(input_dim=13, action_dim=1, use_lora=False)
agent.load_trained_model('results/trained_models/sac_agent_0.pth')
```

---

### Checkpoint from PyTorch < 2.6 fails to load

**Symptom:** `_pickle.UnpicklingError` or `WeightsOnlyError` when loading an old checkpoint.

**Cause:** PyTorch 2.6+ changed the default for `weights_only`. The `load_trained_model()` method sets `weights_only=False` explicitly to handle this.

**Fix:** Ensure you are calling `agent.load_trained_model(path)` rather than `torch.load(path)` directly.

---

## Configuration

### `KeyError` when accessing config

**Symptom:** `KeyError: 'some_key'` when calling `get_config('sac')['some_key']`.

**Cause:** YAML key does not exist or has a typo.

**Fix:** Use `.get('key', default)` instead of direct indexing, or add the missing key to the YAML file. Check spelling against [`docs/configuration.md`](configuration.md).

---

### Config changes not taking effect

**Cause:** `get_config()` is a singleton — it caches the first load. In long-running interactive sessions, a changed YAML file may not be re-read.

**Fix:** Restart Python (or the simulation) after editing a YAML file.

---

## Results & Output

### `results/simulation_registry.json` is missing entries

**Cause:** The run failed before `EvalMetrics.plot_metrics()` was called (which writes the registry entry).

**Fix:** Check for traceback output after the failed run. The run directory may still exist with partial data.

---

### Mode 16 (multi-seed) takes too long

The full 10-seed × all-methods run can take 24–48 hours on CPU. Strategies to reduce runtime:

1. Use `--simulation 16` with the "filtered methods" option (HFDRL + 5 key baselines).
2. Use `num_episodes: 500` instead of 1 000 in `configs/training.yaml` for a preliminary run.
3. Run in Development mode (mode=2) with `num_episodes: 20` to verify the pipeline works before committing to the full run.
4. Use a GPU — SAC training is significantly faster with CUDA.

---

### Plots are blank / white

**Cause:** Matplotlib backend issue in headless environments (e.g., SSH without X forwarding).

**Fix:** Set the backend before importing matplotlib:

```bash
export MPLBACKEND=Agg   # non-interactive, saves to file
python main.py 1 --simulation 3
```

Or add `matplotlib.use('Agg')` at the top of the script that fails.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Running in Training mode with dev config edits | 1 000 episodes but only 5 agents | Separate configs: edit `training_dev.yaml` for dev, `training.yaml` for production |
| `use_real_prices: false` in training.yaml | Synthetic prices, results not reproducible | Set `use_real_prices: true` and point `real_prices_csv` at the correct file |
| Forgetting to activate venv | `ModuleNotFoundError` for every import | `source venv/bin/activate` |
| Running mode 16 without enough seeds | Significance tests have no power | Use ≥ 5 seeds for a minimum significance test; 10 seeds preferred |
| Comparing a Dev-mode run against a Training-mode result | Unfair comparison (different episodes, agents, prices) | Always compare runs with identical configs |
