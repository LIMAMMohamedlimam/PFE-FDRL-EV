# utils/DriverBehaviorModel.py
"""
Realistic driver behavior simulation for EV charging scenarios.

Provides three archetypal driver profiles (commuter, flexible, night_charger)
with Gaussian-distributed arrival/departure times and configurable SOC
parameters. Output is directly compatible with EVClientEnv.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Literal


# ═══════════════════════════════════════════════════════════════════════════
# Driver type definitions — Gaussian parameters (mean, std) for each event
# ═══════════════════════════════════════════════════════════════════════════

DRIVER_TYPES = {
    "commuter": {
        "description": "Regular 9-to-5 worker, charges overnight at home",
        "home_arrival":     {"mean": 18.0, "std": 1.5},   # arrives home ~18:00
        "home_departure":   {"mean":  7.0, "std": 0.5},   # leaves home ~07:00
        "office_arrival":   {"mean":  8.0, "std": 0.5},   # arrives office ~08:00
        "office_departure": {"mean": 17.0, "std": 1.0},   # leaves office ~17:00
        "soc_init":         {"low": 0.15, "high": 0.35},  # arrives with low SOC
        "soc_req":          {"low": 0.85, "high": 1.00},  # needs near-full charge
        "weight": 0.50,  # population share
    },
    "flexible": {
        "description": "Freelancer / remote worker with irregular schedule",
        "home_arrival":     {"mean": 20.0, "std": 3.0},   # later, more variable
        "home_departure":   {"mean": 10.0, "std": 2.0},   # leaves later
        "office_arrival":   {"mean": 11.0, "std": 2.0},
        "office_departure": {"mean": 19.0, "std": 2.5},
        "soc_init":         {"low": 0.10, "high": 0.50},
        "soc_req":          {"low": 0.70, "high": 0.95},
        "weight": 0.30,
    },
    "night_charger": {
        "description": "Shift worker / ride-share driver, charges during the day",
        "home_arrival":     {"mean":  6.0, "std": 1.5},   # arrives home early morning
        "home_departure":   {"mean": 18.0, "std": 2.0},   # departs evening
        "office_arrival":   {"mean": 19.0, "std": 1.5},
        "office_departure": {"mean":  5.0, "std": 1.0},
        "soc_init":         {"low": 0.05, "high": 0.30},  # heaviest usage
        "soc_req":          {"low": 0.80, "high": 1.00},
        "weight": 0.20,
    },
}

DriverType = Literal["commuter", "flexible", "night_charger"]


# ═══════════════════════════════════════════════════════════════════════════
# Core class
# ═══════════════════════════════════════════════════════════════════════════

class DriverBehaviorModel:
    """
    Generates realistic daily EV driver schedules using Gaussian arrival
    and departure distributions for three archetypal driver types.

    The output format is directly compatible with ``EVClientEnv`` —
    each profile dict contains the keys expected by the training loop:
    ``t_start``, ``t_dep``, ``duration``, ``soc_init``, ``soc_req``.
    """

    def __init__(self, rng_seed: int | None = None):
        """
        Args:
            rng_seed: Optional seed for reproducibility.
        """
        self.rng = np.random.default_rng(rng_seed)

    # ── Sampling helpers ──────────────────────────────────────────────────

    def sample_driver_type(self) -> str:
        """
        Samples a driver type using the population weights defined in
        DRIVER_TYPES.

        Returns:
            One of ``'commuter'``, ``'flexible'``, ``'night_charger'``.
        """
        types = list(DRIVER_TYPES.keys())
        weights = [DRIVER_TYPES[t]["weight"] for t in types]
        return self.rng.choice(types, p=weights)

    def sample_home_arrival(self, driver_type: DriverType) -> int:
        """Returns sampled home arrival hour (0-23)."""
        cfg = DRIVER_TYPES[driver_type]["home_arrival"]
        return self._clamp_hour(self.rng.normal(cfg["mean"], cfg["std"]))

    def sample_home_departure(self, driver_type: DriverType) -> int:
        """Returns sampled home departure hour (0-23)."""
        cfg = DRIVER_TYPES[driver_type]["home_departure"]
        return self._clamp_hour(self.rng.normal(cfg["mean"], cfg["std"]))

    def sample_office_arrival(self, driver_type: DriverType) -> int:
        """Returns sampled office arrival hour (0-23)."""
        cfg = DRIVER_TYPES[driver_type]["office_arrival"]
        return self._clamp_hour(self.rng.normal(cfg["mean"], cfg["std"]))

    def sample_office_departure(self, driver_type: DriverType) -> int:
        """Returns sampled office departure hour (0-23)."""
        cfg = DRIVER_TYPES[driver_type]["office_departure"]
        return self._clamp_hour(self.rng.normal(cfg["mean"], cfg["std"]))

    # ── Schedule generation ───────────────────────────────────────────────

    def generate_driver_schedule(self, days: int = 1) -> Dict:
        """
        Generates a full multi-day schedule for a single randomly-sampled
        driver, including home and office arrival/departure times per day.

        Args:
            days: Number of days to generate.

        Returns:
            Dictionary with keys:
                - ``driver_type``  (str)
                - ``days``        (list of day-level dicts)
                  Each day dict contains:
                    ``home_arrival``, ``home_departure``,
                    ``office_arrival``, ``office_departure``
        """
        dt = self.sample_driver_type()
        schedule = {
            "driver_type": dt,
            "description": DRIVER_TYPES[dt]["description"],
            "days": [],
        }
        for _ in range(days):
            schedule["days"].append({
                "home_arrival":     self.sample_home_arrival(dt),
                "home_departure":   self.sample_home_departure(dt),
                "office_arrival":   self.sample_office_arrival(dt),
                "office_departure": self.sample_office_departure(dt),
            })
        return schedule

    # ── EVClientEnv-compatible profile generation ─────────────────────────

    def generate_profiles(self, n_drivers: int) -> List[Dict]:
        """
        Generates ``n_drivers`` profiles that are **drop-in compatible**
        with ``DataGenerator.get_nhts_profile()`` and the existing
        training loop.

        Each profile dict contains:
            - ``t_start``   : hour the EV connects (home arrival)
            - ``t_dep``     : hour the EV must depart (home departure next morning)
            - ``duration``  : hours the EV is connected
            - ``soc_init``  : initial state of charge
            - ``soc_req``   : required SOC at departure
            - ``driver_type``: label for analysis

        Returns:
            List of profile dicts.
        """
        profiles: List[Dict] = []
        for _ in range(n_drivers):
            dt = self.sample_driver_type()
            cfg = DRIVER_TYPES[dt]

            t_arrival   = self.sample_home_arrival(dt)
            t_departure = self.sample_home_departure(dt)

            # Duration: overnight charging (arrival evening → departure morning)
            if t_departure < t_arrival:
                stay_duration = (24 - t_arrival) + t_departure
            else:
                stay_duration = t_departure - t_arrival

            # Ensure minimum 1-hour connection
            stay_duration = max(1, stay_duration)

            soc_init = float(self.rng.uniform(cfg["soc_init"]["low"],
                                               cfg["soc_init"]["high"]))
            soc_req  = float(self.rng.uniform(cfg["soc_req"]["low"],
                                               cfg["soc_req"]["high"]))

            profiles.append({
                "t_start":     t_arrival,
                "t_dep":       t_departure,
                "duration":    stay_duration,
                "soc_init":    soc_init,
                "soc_req":     soc_req,
                "driver_type": dt,
            })
        return profiles

    # ── Visualization ─────────────────────────────────────────────────────

    @staticmethod
    def plot_probability_curves(
        n_samples: int = 10_000,
        save_path: str | None = None,
    ) -> None:
        """
        Visualises the Gaussian arrival/departure probability distributions
        for each driver type as a 3×4 subplot grid.

        Args:
            n_samples: Number of Monte-Carlo samples per curve.
            save_path: If provided, saves the figure to this path (PNG).
        """
        events = [
            ("home_arrival",     "Home Arrival"),
            ("home_departure",   "Home Departure"),
            ("office_arrival",   "Office Arrival"),
            ("office_departure", "Office Departure"),
        ]
        colors = {"commuter": "#2196F3", "flexible": "#FF9800",
                  "night_charger": "#9C27B0"}

        fig, axes = plt.subplots(
            len(DRIVER_TYPES), len(events),
            figsize=(16, 9), sharex=True, sharey="row",
        )
        fig.suptitle(
            "Driver Behavior — Gaussian Arrival / Departure Distributions",
            fontsize=14, fontweight="bold",
        )

        hours = np.arange(0, 24, 0.25)

        for row_idx, (dtype, dcfg) in enumerate(DRIVER_TYPES.items()):
            for col_idx, (event_key, event_label) in enumerate(events):
                ax = axes[row_idx, col_idx]
                mu  = dcfg[event_key]["mean"]
                sig = dcfg[event_key]["std"]

                # Gaussian PDF (wrapped around 24h for visual clarity)
                pdf = np.exp(-0.5 * ((hours - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

                ax.fill_between(hours, pdf, alpha=0.35, color=colors[dtype])
                ax.plot(hours, pdf, color=colors[dtype], linewidth=1.5)
                ax.axvline(mu, color=colors[dtype], linestyle="--",
                           linewidth=0.8, alpha=0.7)
                ax.set_xlim(0, 24)
                ax.set_xticks(range(0, 25, 4))

                if row_idx == 0:
                    ax.set_title(event_label, fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{dtype}\n(w={dcfg['weight']:.0%})",
                        fontsize=9, rotation=0, labelpad=70, va="center",
                    )
                if row_idx == len(DRIVER_TYPES) - 1:
                    ax.set_xlabel("Hour of Day")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[DriverBehaviorModel] Saved probability curves → {save_path}")
        plt.show()

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _clamp_hour(value: float) -> int:
        """Clamp a float to a valid hour integer [0, 23]."""
        return int(np.clip(round(value), 0, 23))
