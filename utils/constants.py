"""
Domain constants for the HFDRL EV charging project.

Centralised here so physical meaning is documented once and values
stay consistent across env, utils, training, and scripts.
"""

# ── State normalisation (EVClientEnv) ────────────────────────────────────────

# Price forecast: raw $/kWh divided by this to keep state in [0, ~2]
PRICE_NORM_SCALE = 0.50

# EV fleet load (MW) clipped to [0, 1] after dividing by this
EV_TOTAL_NORM_SCALE = 0.25

# Delta EV load (MW) clipped to [-1, 1] after dividing by this
EV_DELTA_NORM_SCALE = 0.10

# ── Voltage bounds (IEEE 1547 / per-unit) ────────────────────────────────────

VOLTAGE_MIN = 0.95  # p.u. lower limit
VOLTAGE_MAX = 1.05  # p.u. upper limit
VOLTAGE_BAND = 0.05  # half-band used to scale lambda_grid signal

# ── Statistical tests ────────────────────────────────────────────────────────

CI_Z_95 = 1.96          # z-score for 95 % confidence interval
ALPHA_SIGNIFICANCE = 0.05  # p-value threshold for significance tests

# ── Convergence detection (MultiSeedRunner) ───────────────────────────────────

CONVERGENCE_WINDOW = 10       # rolling-mean window (episodes)
CONVERGENCE_THRESHOLD = 0.90  # fraction of final mean that counts as converged

# ── Q-Learning exploration schedule (SimulationRunner) ───────────────────────

EPSILON_MIN = 0.05   # minimum exploration rate
EPSILON_DECAY = 0.95  # per-episode multiplicative decay
