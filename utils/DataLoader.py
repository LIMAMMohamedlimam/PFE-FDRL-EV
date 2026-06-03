import matplotlib.pyplot as plt
import random
import numpy as np

from utils.config_loader import get_config
from utils.MarketPriceLoader import MarketPriceLoader

# --- Générateurs de Données (Mock + Real) ---

class DataGenerator:
    """Génère les données d'environnement (Prix et Comportement)"""
    
    dev_mode = False
    
    # Lazily-initialized real price loader (shared across all callers)
    _price_loader: MarketPriceLoader | None = None

    @classmethod
    def _ensure_price_loader(cls) -> MarketPriceLoader:
        """Instantiates and caches the MarketPriceLoader on first access."""
        if cls._price_loader is None:
            if cls.dev_mode:
                train_cfg = get_config("training_dev")
            else:
                train_cfg = get_config('training')
            csv_path = train_cfg.get('real_prices_csv', 'data/iso_ne_prices.csv')
            cls._price_loader = MarketPriceLoader(csv_path)
            cls._price_loader.load_prices()
        return cls._price_loader

    @staticmethod
    def get_iso_ne_price(hour, mode='train'):
        """
        Returns the electricity price for a given hour.

        When ``use_real_prices`` is *True* in ``training.yaml``, prices
        come from the CSV loaded by :class:`MarketPriceLoader`.
        Otherwise the original synthetic ISO-NE profile is used.
        """
        if DataGenerator.dev_mode:
            train_cfg = get_config("training_dev")
        else:
            train_cfg = get_config('training')

        if train_cfg.get('use_real_prices', False):
            # print("Using real prices", train_cfg.get('real_prices_csv', "Error: No CSV path provided"))
            loader = DataGenerator._ensure_price_loader()
            return loader.get_price(hour, mode=mode)
        # else:
            # print("Using synthetic prices")

        # --- Original synthetic fallback (unchanged) ---
        base_price = 0.15

        hour_adjusted = hour
        if mode == 'test':
            hour_adjusted = (hour + 2) % 24  # Décalage de pic (ex: hiver vs été)
            base_price += 0.02  # Prix globalement plus élevés

        if 8 <= hour_adjusted <= 10 or 18 <= hour_adjusted <= 21:
            return base_price * 1.5
        elif 0 <= hour_adjusted <= 5:
            return base_price * 0.6
        return base_price

    @staticmethod
    def get_nhts_profile(n_drivers):
        """
        Simule les profils NHTS 2017: Heure d'arrivée et de départ.
        Arrivée ~ Normal(18h, 2h), Départ ~ Normal(7h, 1h).
        """
        profiles = []
        for _ in range(n_drivers):
            # Simulation simplifiée sur 24h : Arrivée le soir, départ le lendemain matin
            t_arrival = int(np.random.normal(18, 2))  # ~18h00
            t_departure = int(np.random.normal(7, 1)) # ~07h00 (J+1)

            # Correction des bornes (0-23h)
            t_arrival = max(0, min(23, t_arrival))
            t_departure = max(0, min(23, t_departure))

            # Durée de connexion (si départ < arrivée, c'est le lendemain)
            stay_duration = (24 - t_arrival) + t_departure if t_departure < t_arrival else t_departure - t_arrival

            profiles.append({
                't_start': t_arrival,
                't_dep': t_departure, # Heure absolue de départ J+1 pour la simu
                'duration': stay_duration,
                'soc_init': np.random.uniform(0.1, 0.4), # Arrive avec batterie faible
                'soc_req': np.random.uniform(0.8, 1.0)   # Veut repartir plein
            })
        return profiles

    # ── Non-IID archetype tables ──────────────────────────────────────────────
    # Each row: (t_arr_mu, t_arr_sig, t_dep_mu, t_dep_sig, soc_init_lo, soc_req_lo, label)

    # NHTS 2017 calibrated (default) — Bureau of Transportation Statistics
    _NHTS_ARCHETYPES = [
        (17, 1,   8, 0.5, 0.10, 0.90, 'commuter'),     # evening plug-in, early departure
        (19, 3,   9, 2.0, 0.20, 0.70, 'flexible'),      # variable schedule, lower SOC target
        (22, 1,   6, 1.0, 0.10, 0.80, 'night_charger'), # late plug-in, early morning departure
    ]

    # ACN-Data calibrated — Lee et al., 2019, Caltech Adaptive Charging Network
    _ACN_ARCHETYPES = [
        ( 9, 1.5, 17, 1.0, 0.30, 0.85, 'work_day'),   # morning arrival, afternoon departure
        (18, 1.5, 23, 1.5, 0.20, 0.80, 'evening'),     # evening arrival, late-night departure
        (22, 1.0,  7, 1.0, 0.10, 0.85, 'overnight'),   # night plug-in, morning departure
    ]

    _ARCHETYPE_SETS = {
        'nhts': _NHTS_ARCHETYPES,
        'acn':  _ACN_ARCHETYPES,
    }

    @staticmethod
    def get_nhts_profile_noniid(
        n_drivers: int,
        alpha: float,
        n_edges: int = 2,
        archetype_set: str = 'nhts',
    ) -> list:
        """
        Non-IID driver profiles via Dirichlet allocation of 3 archetypes per edge.

        Each FL edge gets its own type-proportion vector sampled from
        Dirichlet([alpha]*3). Lower alpha → stronger inter-edge heterogeneity.
        alpha=1000 ≈ IID (all edges see the same balanced mix).

        Parameters
        ----------
        n_drivers     : Total number of agent profiles to generate.
        alpha         : Dirichlet concentration. Lower = more non-IID.
        n_edges       : Number of FL edges (heterogeneity groups).
        archetype_set : 'nhts' (default, NHTS-2017) or 'acn' (Lee et al., 2019).
        """
        archetypes = DataGenerator._ARCHETYPE_SETS.get(archetype_set,
                                                        DataGenerator._NHTS_ARCHETYPES)
        n_types = len(archetypes)
        agents_per_edge = np.array_split(range(n_drivers), n_edges)
        profiles = [None] * n_drivers
        for edge_agents in agents_per_edge:
            props = np.random.dirichlet([alpha] * n_types)
            for idx in edge_agents:
                chosen = int(np.random.choice(n_types, p=props))
                a = archetypes[chosen]
                t_arr = int(np.clip(np.random.normal(a[0], a[1]), 0, 23))
                t_dep = int(np.clip(np.random.normal(a[2], a[3]), 0, 23))
                stay  = (24 - t_arr + t_dep) if t_dep < t_arr else (t_dep - t_arr)
                profiles[idx] = {
                    't_start':    t_arr,
                    't_dep':      t_dep,
                    'duration':   stay,
                    'soc_init':   float(np.random.uniform(a[4], a[4] + 0.2)),
                    'soc_req':    float(np.random.uniform(a[5], min(a[5] + 0.15, 1.0))),
                    '_archetype': a[6],
                }
        return profiles
