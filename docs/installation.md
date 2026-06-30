# Installation & Environment

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.11 |
| OS | Ubuntu 20.04 / macOS 12 | Ubuntu 22.04 |
| RAM | 4 GB | 16 GB |
| GPU | CPU-only works | CUDA 11.8+ (for faster SAC training) |

The core dependency is **pandapower** for IEEE 33-bus power-flow simulation. PyTorch is required for SAC and PPO agents; it is pulled in transitively by the requirements file.

---

## Installation Steps

### 1. Clone the repository

```bash
git clone <repo-url> pfe_imp
cd pfe_imp
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

The `requirement.txt` pins the two primary runtime dependencies. PyTorch, NumPy, Matplotlib, and other standard packages are installed as transitive dependencies.

If questionary or Flask are not pulled in automatically, install them explicitly:

```bash
pip install questionary flask
```

### 4. (Optional) GPU support

PyTorch will use CUDA automatically if a compatible GPU and driver are detected. Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the output is `True`, SAC/PPO training will run on GPU. The project uses `utils/device_utils.py` to discover CUDA or Apple MPS automatically — no manual device flag is needed.

For a specific CUDA version, install PyTorch with the matching wheel from https://pytorch.org/get-started/locally/ before running `pip install -r requirement.txt`.

---

## Smoke Test

Run the following to confirm that imports, the grid environment, and the config system all load correctly:

```bash
python -c "
from env.GridEnv import GridEnv
from env.EVClientEnv import EVClientEnv
from agents.SACAgent import SACAgent
from utils.config_loader import get_config
cfg = get_config('env')
grid = GridEnv(network_type='case33bw')
print('OK — GridEnv, SACAgent, config system loaded')
"
```

Expected output: `OK — GridEnv, SACAgent, config system loaded`

For a full end-to-end smoke test using the fast dev configuration (20 episodes, 5 agents):

```bash
python main.py 2 --simulation 3   # Dev mode → SAC Training
```

This finishes in under a minute and confirms the training loop, reward pipeline, and metrics logging all work.

---

## Data Files

The project ships with two market price CSV files under `data/`:

| File | Used by | Description |
|------|---------|-------------|
| `data/iso_ne_prices_real.csv` | Training mode | 30-day real ISO-NE hourly wholesale prices |
| `data/iso_ne_prices_dev_test.csv` | Dev mode | Shorter slice for fast iteration |

Both files must have columns `timestamp` and `price`. Prices are min-max normalised to [0, 1] by `utils/MarketPriceLoader.py`.

To disable real prices and use the synthetic generator instead, set `use_real_prices: false` in `configs/training.yaml`.

---

## Known Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImportError: No module named 'questionary'` | Missing optional dep | `pip install questionary` |
| `ImportError: No module named 'flask'` | Missing optional dep | `pip install flask` |
| pandapower power-flow convergence warnings | Normal for heavily loaded bus | Safe to ignore; grid physics still correct |
| CUDA out of memory | Too many agents or large buffer | Reduce `num_agents` or `buffer_capacity` in config |
