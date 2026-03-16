# utils/MarketPriceLoader.py
"""
Loads real hourly electricity price data from a CSV file (ISO-NE style).
Provides normalized prices, train/test splits, and rolling-window retrieval
for integration with the FDRL-EV simulation pipeline.
"""

import os
import numpy as np
import pandas as pd


class MarketPriceLoader:
    """
    Loads and preprocesses real electricity market price data.

    Expected CSV format:
        timestamp,price
        2021-07-01 00:00,0.035
        2021-07-01 01:00,0.032
        ...

    Prices are min-max normalized to [0, 1] and split 80/20 into
    train / test series for use in RL simulation episodes.
    """

    def __init__(self, csv_path: str):
        """
        Args:
            csv_path: Absolute or project-relative path to the price CSV file.
        """
        # Resolve relative paths against the project root
        if not os.path.isabs(csv_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(project_root, csv_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Price CSV not found: {csv_path}")

        self.csv_path = csv_path

        # Populated by load_prices()
        self._raw_prices: np.ndarray | None = None
        self._normalized: np.ndarray | None = None
        self._train: np.ndarray | None = None
        self._test: np.ndarray | None = None
        self._price_min: float = 0.0
        self._price_max: float = 1.0

        # Hourly averages by mode (lazy-cached)
        self._hourly_avg_train: np.ndarray | None = None
        self._hourly_avg_test: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_prices(self) -> None:
        """
        Reads the CSV, normalizes prices to [0, 1], and creates an 80/20
        train/test split along the time axis (preserving temporal order).
        """
        df = pd.read_csv(self.csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        self._raw_prices = df["price"].values.astype(np.float64)

        # Min-max normalization → [0, 1]
        self._price_min = float(self._raw_prices.min())
        self._price_max = float(self._raw_prices.max())
        denom = self._price_max - self._price_min
        if denom < 1e-12:
            # Constant price edge-case
            self._normalized = np.zeros_like(self._raw_prices)
        else:
            self._normalized = (self._raw_prices - self._price_min) / denom

        # 80/20 temporal split
        split_idx = int(len(self._normalized) * 0.8)
        self._train = self._normalized[:split_idx]
        self._test = self._normalized[split_idx:]

        # Pre-compute hourly averages for fast per-hour lookups
        self._hourly_avg_train = self._compute_hourly_avg(self._train)
        self._hourly_avg_test = self._compute_hourly_avg(self._test)

        print(
            f"[MarketPriceLoader] Loaded {len(self._raw_prices)} hours "
            f"(train={len(self._train)}, test={len(self._test)}) "
            f"from {os.path.basename(self.csv_path)}"
        )

    def get_train_series(self) -> np.ndarray:
        """Returns the normalized training price series."""
        self._check_loaded()
        return self._train

    def get_test_series(self) -> np.ndarray:
        """Returns the normalized test price series."""
        self._check_loaded()
        return self._test

    def get_price_window(self, hour: int, window_size: int = 48) -> np.ndarray:
        """
        Returns a rolling window of *window_size* normalized prices
        starting at the given absolute hour index in the full dataset.

        If the window extends past the end of the dataset it wraps around.

        Args:
            hour:        Absolute hour index into the full normalized series.
            window_size: Number of consecutive hours to return.

        Returns:
            np.ndarray of shape (window_size,) with normalized prices.
        """
        self._check_loaded()
        n = len(self._normalized)
        indices = [(hour + i) % n for i in range(window_size)]
        return self._normalized[indices]

    def get_price(self, hour: int, mode: str = "train") -> float:
        """
        Drop-in replacement for ``DataGenerator.get_iso_ne_price(hour, mode)``.

        Returns the average normalized price for *hour-of-day* (0-23)
        computed over the train or test split respectively.

        Args:
            hour: Hour of day (0-23).
            mode: 'train' or 'test'.

        Returns:
            A float in [0, 1].
        """
        self._check_loaded()
        h = int(hour) % 24
        if mode == "test":
            return float(self._hourly_avg_test[h])
        return float(self._hourly_avg_train[h])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hourly_avg(series: np.ndarray) -> np.ndarray:
        """
        Computes mean price for each hour-of-day (0-23) across all full
        days contained in *series* (assumed to start at hour 0).
        """
        hourly_avg = np.zeros(24)
        counts = np.zeros(24)
        for idx, val in enumerate(series):
            h = idx % 24
            hourly_avg[h] += val
            counts[h] += 1
        # Avoid divide-by-zero for hours with no data
        counts[counts == 0] = 1
        return hourly_avg / counts

    def _check_loaded(self) -> None:
        if self._normalized is None:
            raise RuntimeError(
                "Prices not loaded yet.  Call load_prices() first."
            )
