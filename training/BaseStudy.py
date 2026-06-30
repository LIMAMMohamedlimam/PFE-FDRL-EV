"""
Shared utilities for multi-seed study runners.

DwellTimeStudy, LoRANetworkStudy, StressTestStudy, and MultiSeedRunner
all need identical logger setup, progress logging, artifact saving, and
reproducibility note scaffolding. Centralised here to avoid copy-paste.
"""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


class BaseStudy:

    # ── Logger ───────────────────────────────────────────────────────────────

    @staticmethod
    def setup_logger(output_dir: str, logger_name: str) -> logging.Logger:
        """Dual-sink logger: tqdm-safe console (INFO) + file (DEBUG)."""
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        fmt = logging.Formatter(
            '%(asctime)s  %(levelname)-7s  %(message)s', datefmt='%H:%M:%S'
        )

        class TqdmHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    tqdm.write(self.format(record))
                except Exception:
                    self.handleError(record)

        ch = TqdmHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, 'study.log'), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        fh.terminator = '\n'
        logger.addHandler(fh)

        return logger

    # ── Progress log ──────────────────────────────────────────────────────────

    @staticmethod
    def append_progress(progress_path: str, entry: dict) -> None:
        """Append one JSON line to the rolling progress log (fsynced)."""
        with open(progress_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
            f.flush()
            os.fsync(f.fileno())

    # ── Atomic JSON write ─────────────────────────────────────────────────────

    @staticmethod
    def atomic_write_json(path: str, data: dict) -> None:
        """Write data to path atomically via a .tmp file (POSIX-safe)."""
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    # ── Seed artifacts ────────────────────────────────────────────────────────

    @staticmethod
    def save_seed_artifacts(
        result: dict,
        seed_dir: str,
        logger: logging.Logger = None,
    ) -> None:
        """Save reward/cost/satisfaction .npy curves and scalar metrics.json."""
        os.makedirs(seed_dir, exist_ok=True)
        for curve in ('reward_curve', 'cost_curve', 'satisfaction_curve'):
            np.save(
                os.path.join(seed_dir, f'{curve}.npy'),
                np.array(result[curve], dtype=np.float32),
            )
        scalars = {k: v for k, v in result.items() if not isinstance(v, list)}
        scalars['n_reward_episodes'] = len(result['reward_curve'])
        with open(os.path.join(seed_dir, 'metrics.json'), 'w') as f:
            json.dump(scalars, f, indent=2)
        if logger:
            logger.debug(f'  saved metrics -> {seed_dir}/metrics.json')

    # ── Reproducibility notes ─────────────────────────────────────────────────

    @staticmethod
    def base_repro_notes(seeds: list, dev_mode: bool) -> dict:
        """Return the common fields for every reproducibility_notes.json."""
        return {
            'timestamp':      datetime.now().isoformat(),
            'seeds':          seeds,
            'n_seeds':        len(seeds),
            'dev_mode':       dev_mode,
            'python_version': sys.version,
            'numpy_version':  np.__version__,
            'torch_version':  torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'rng_control': {
                'python_random': 'random.seed(seed)',
                'numpy':         'np.random.seed(seed)',
                'torch':         'torch.manual_seed(seed)',
                'torch_cuda':    'torch.cuda.manual_seed_all(seed)',
                'cudnn':         'deterministic=True',
                'env_hash':      'PYTHONHASHSEED=str(seed)',
            },
            'ci_formula': 'CI_95 = 1.96 * std / sqrt(n)',
        }

    @staticmethod
    def write_repro_notes(
        output_dir: str,
        notes: dict,
        logger: logging.Logger = None,
    ) -> None:
        """Write notes dict to reproducibility_notes.json in output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'reproducibility_notes.json')
        with open(path, 'w') as f:
            json.dump(notes, f, indent=2)
        if logger:
            logger.info(f'Reproducibility notes -> {path}')
