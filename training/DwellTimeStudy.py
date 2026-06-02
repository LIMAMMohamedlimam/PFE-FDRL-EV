"""
DwellTimeStudy.py
=================
Dedicated runner for the SWIFT dwell-time analysis (AAAI paper).

Scientific objective
--------------------
Demonstrate that SWIFT better exploits short participation windows and
intermittent agent availability compared to classical FedAvg aggregation.

Experiment design
-----------------
  Dwell-time scenarios : [1h, 2h, 4h, 6h]
  Methods              : FedAvg-SAC | SWIFT-SAC | HFDRL
  Seeds                : 10 (configurable via dwell_time_study.yaml)

All methods use identical environments, seeds, and evaluation protocol.
Only the dwell duration (t_dep = sim_hours) and the FL selection strategy vary.

Usage
-----
    # Full AAAI run (10 seeds, ~hours on GPU)
    python -m training.DwellTimeStudy

    # Quick smoke test
    python -m training.DwellTimeStudy --dev --seeds 0 1

    # Single (method, dwell, seed) triple — fully standalone
    python -m training.DwellTimeStudy --single --method "SWIFT-SAC" --dwell 2 --seed 3

    # Via interactive menu (main.py → option 17)
    python main.py
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from utils.config_loader import get_config
from training.MultiSeedRunner import set_global_seed, extract_seed_metrics
from training.ComparisonPipeline import run_single_experiment


# ─────────────────────────────────────────────────────────────────────────────
# Study constants
# ─────────────────────────────────────────────────────────────────────────────

STUDY_METHODS = [
    {
        'name': 'FedAvg-SAC',
        'policy': 'sac',
        'aggregation': 'fedavg',
        'use_swift': False,
        'use_lora': False,
    },
    {
        'name': 'SWIFT-SAC',
        'policy': 'sac',
        'aggregation': 'fedavg',
        'use_swift': True,
        'use_lora': False,
    },
    {
        'name': 'HFDRL',
        'policy': 'sac',
        'aggregation': 'fedavg',
        'use_swift': True,
        'use_lora': True,
    },
]

METHOD_BY_NAME = {m['name']: m for m in STUDY_METHODS}


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logger(output_dir: str) -> logging.Logger:
    """Create a logger that writes to both stdout and a log file immediately."""
    logger = logging.getLogger('dwell_study')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s  %(levelname)-7s  %(message)s',
                            datefmt='%H:%M:%S')

    # Console handler — use tqdm.write so it doesn't break progress bars
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

    # File handler — flushed after every record
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'study.log')
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    fh.terminator = '\n'
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_study_cfg() -> dict:
    try:
        return get_config('dwell_time_study')
    except Exception:
        return {}


def _method_kwargs(method: dict, dwell_hours: int, swift_min_stay: float) -> dict:
    return dict(
        policy=method['policy'],
        aggregation=method['aggregation'],
        use_swift=method['use_swift'],
        use_lora=method['use_lora'],
        dwell_time_hours=dwell_hours,
        swift_min_stay_override=swift_min_stay if method['use_swift'] else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed runner  (used by both single-triple and full-study paths)
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_seed(
    method: dict,
    dwell_hours: int,
    swift_min_stay: float,
    seed: int,
    seed_dir: str,
    dev_mode: bool,
    logger: logging.Logger,
    show_progress: bool = False,
    tqdm_position: int = None,
    tqdm_leave: bool = False,
) -> dict:
    """Run one (method, dwell, seed) triple. Returns metric dict or None on failure."""
    set_global_seed(seed)
    kwargs = _method_kwargs(method, dwell_hours, swift_min_stay)

    t0 = time.time()
    try:
        metrics = run_single_experiment(
            verbose=False,
            progress_enabled=show_progress,
            dev_mode=dev_mode,
            tqdm_position=tqdm_position,
            tqdm_leave=tqdm_leave,
            **kwargs,
        )
    except Exception as exc:
        logger.error(f"FAILED {method['name']} dwell={dwell_hours}h seed={seed}: {exc}",
                     exc_info=True)
        return None

    wall_time = time.time() - t0
    result = extract_seed_metrics(metrics, seed)
    result['wall_time_s'] = round(wall_time, 2)
    result['dwell_hours'] = dwell_hours

    os.makedirs(seed_dir, exist_ok=True)

    np.save(os.path.join(seed_dir, 'reward_curve.npy'),
            np.array(result['reward_curve'], dtype=np.float32))
    np.save(os.path.join(seed_dir, 'cost_curve.npy'),
            np.array(result['cost_curve'], dtype=np.float32))
    np.save(os.path.join(seed_dir, 'satisfaction_curve.npy'),
            np.array(result['satisfaction_curve'], dtype=np.float32))

    scalars = {k: v for k, v in result.items() if not isinstance(v, list)}
    scalars['n_reward_episodes'] = len(result['reward_curve'])
    with open(os.path.join(seed_dir, 'metrics.json'), 'w') as f:
        json.dump(scalars, f, indent=2)

    logger.debug(f"  saved metrics -> {seed_dir}/metrics.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Progress log  (written after every seed, readable while study runs)
# ─────────────────────────────────────────────────────────────────────────────

def _append_progress(progress_path: str, entry: dict) -> None:
    """Append one JSON line to the rolling progress log."""
    with open(progress_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _update_aggregated_raw(raw_path: str, all_results: dict) -> None:
    """Rewrite aggregated_raw.json with current state (called after each method)."""
    agg_raw = {}
    for dwell_h, methods_dict in all_results.items():
        agg_raw[str(dwell_h)] = {}
        for mname, seed_list in methods_dict.items():
            agg_raw[str(dwell_h)][mname] = [
                {k: v for k, v in r.items() if not isinstance(v, list)}
                for r in seed_list
            ]
    tmp = raw_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(agg_raw, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, raw_path)  # atomic on POSIX


# ─────────────────────────────────────────────────────────────────────────────
# Single-triple entry point  (fully standalone — no full study needed)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_triple(
    method_name: str,
    dwell_hours: int,
    seed: int,
    dev_mode: bool = False,
    output_base: str = None,
) -> dict:
    """
    Run exactly one (method, dwell_hours, seed) combination independently.

    Parameters
    ----------
    method_name  : One of 'FedAvg-SAC', 'SWIFT-SAC', 'HFDRL'
    dwell_hours  : One of 1, 2, 4, 6
    seed         : Integer seed
    dev_mode     : Use training_dev.yaml
    output_base  : Root directory (defaults to results/dwell_time_study)

    Returns
    -------
    Metric dict (scalars + curves) or None on failure.
    """
    if method_name not in METHOD_BY_NAME:
        raise ValueError(f"Unknown method '{method_name}'. Choose from: {list(METHOD_BY_NAME)}")

    cfg = _load_study_cfg()
    if output_base is None:
        output_base = cfg.get('output_base', 'results/dwell_time_study')

    swift_min_stay_map = {int(k): float(v) for k, v in
                          cfg.get('swift_min_stay', {1: 0.0, 2: 0.5, 4: 1.0, 6: 1.5, 8: 2.0}).items()}
    swift_min_stay = swift_min_stay_map.get(dwell_hours, 0.0)

    method = METHOD_BY_NAME[method_name]
    safe_mname = method_name.replace(' ', '_').replace('/', '-')

    seed_dir = os.path.join(
        output_base, 'singles',
        f'dwell_{dwell_hours}h', safe_mname, f'seed_{seed}'
    )
    os.makedirs(seed_dir, exist_ok=True)

    logger = _setup_logger(seed_dir)
    logger.info(f"Single-triple run: method={method_name}  dwell={dwell_hours}h  seed={seed}"
                f"  mode={'dev' if dev_mode else 'full'}")
    logger.info(f"Output -> {seed_dir}")

    result = _run_one_seed(
        method=method,
        dwell_hours=dwell_hours,
        swift_min_stay=swift_min_stay,
        seed=seed,
        seed_dir=seed_dir,
        dev_mode=dev_mode,
        logger=logger,
        show_progress=True,
    )

    if result is not None:
        scalars = {k: v for k, v in result.items() if not isinstance(v, list)}
        reward = scalars.get('test_reward')
        reward_str = f"{reward:.3f}" if reward is not None else "N/A"
        logger.info(f"Done — test_reward={reward_str}  wall={result['wall_time_s']:.1f}s")
    else:
        logger.error("Run failed — check study.log for details.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full study pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_dwell_time_study(
    seeds: list = None,
    dwell_hours_list: list = None,
    dev_mode: bool = False,
    verbose: bool = True,
    output_base: str = None,
    method_filter: list = None,
) -> tuple:
    """
    Run the full dwell-time study with on-the-go logging and incremental saves.

    Parameters
    ----------
    seeds           : List of random seeds. None → read from dwell_time_study.yaml.
    dwell_hours_list: List of dwell durations. None → read from config.
    dev_mode        : Use training_dev.yaml (fewer episodes, faster).
    verbose         : Print per-seed lines in addition to tqdm bars.
    output_base     : Root output directory. None → config default.
    method_filter   : If given, only run methods in this list.

    Returns
    -------
    (all_results, output_dir)
    """
    cfg = _load_study_cfg()

    if seeds is None:
        seeds = cfg.get('seeds', [0, 1, 2, 3, 4])
    if dwell_hours_list is None:
        dwell_hours_list = cfg.get('dwell_hours', [1, 2, 4, 6, 8])
    if output_base is None:
        output_base = cfg.get('output_base', 'results/dwell_time_study')

    swift_min_stay_map = {int(k): float(v) for k, v in
                          cfg.get('swift_min_stay', {1: 0.0, 2: 0.5, 4: 1.0, 6: 1.5, 8: 2.0}).items()}

    methods = [m for m in STUDY_METHODS
               if method_filter is None or m['name'] in method_filter]

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_base, timestamp)
    agg_dir    = os.path.join(output_dir, 'aggregated')
    for sub in ('tables', 'plots', 'statistics'):
        os.makedirs(os.path.join(agg_dir, sub), exist_ok=True)

    logger = _setup_logger(output_dir)
    progress_path = os.path.join(output_dir, 'progress.jsonl')
    raw_path      = os.path.join(output_dir, 'aggregated_raw.json')

    _save_reproducibility_notes(output_dir, seeds, dwell_hours_list, dev_mode, logger)

    n_total = len(dwell_hours_list) * len(methods) * len(seeds)
    logger.info('=' * 60)
    logger.info('DWELL-TIME STUDY')
    logger.info(f"Scenarios : {dwell_hours_list}h")
    logger.info(f"Methods   : {[m['name'] for m in methods]}")
    logger.info(f"Seeds     : {seeds}")
    logger.info(f"Total runs: {n_total}  ({'dev' if dev_mode else 'full'} mode)")
    logger.info(f"Output    : {output_dir}")
    logger.info('=' * 60)

    all_results: dict = {}

    overall_bar = tqdm(
        total=n_total,
        desc='Overall',
        unit='run',
        ncols=90,
        position=0,
        leave=True,
    )

    for dwell_h in tqdm(dwell_hours_list, desc='Dwell scenarios', unit='h',
                        ncols=90, position=1, leave=False):
        swift_min_stay = swift_min_stay_map.get(dwell_h, 0.0)
        all_results[dwell_h] = {}

        dwell_dir = os.path.join(output_dir, f'dwell_{dwell_h}h')
        os.makedirs(dwell_dir, exist_ok=True)

        logger.info(f"── Dwell = {dwell_h}h  (SWIFT min_stay={swift_min_stay}h) ──")

        for method in tqdm(methods, desc=f'Methods @{dwell_h}h', unit='method',
                           ncols=90, position=2, leave=False):
            mname = method['name']
            safe_mname = mname.replace(' ', '_').replace('/', '-')
            all_results[dwell_h][mname] = []
            method_dir = os.path.join(dwell_dir, safe_mname)

            for seed in tqdm(seeds, desc=f'{mname}', unit='seed',
                             ncols=90, position=3, leave=False):
                seed_dir = os.path.join(method_dir, f'seed_{seed}')

                logger.debug(f"  starting {mname} dwell={dwell_h}h seed={seed}")

                result = _run_one_seed(
                    method=method,
                    dwell_hours=dwell_h,
                    swift_min_stay=swift_min_stay,
                    seed=seed,
                    seed_dir=seed_dir,
                    dev_mode=dev_mode,
                    logger=logger,
                    show_progress=True,
                    tqdm_position=4,
                    tqdm_leave=False,
                )

                status = 'ok' if result is not None else 'failed'
                if result is not None:
                    all_results[dwell_h][mname].append(result)
                    scalars = {k: v for k, v in result.items() if not isinstance(v, list)}
                    logger.info(
                        f"  [{mname}] dwell={dwell_h}h seed={seed}  "
                        f"test_reward={scalars.get('test_reward', float('nan')):.3f}  "
                        f"wall={result['wall_time_s']:.1f}s"
                    )
                else:
                    logger.warning(f"  [{mname}] dwell={dwell_h}h seed={seed}  FAILED")

                # Write progress immediately after each seed
                _append_progress(progress_path, {
                    'ts': datetime.now().isoformat(),
                    'method': mname,
                    'dwell_h': dwell_h,
                    'seed': seed,
                    'status': status,
                    'wall_time_s': result['wall_time_s'] if result else None,
                    'test_reward': (
                        {k: v for k, v in result.items() if not isinstance(v, list)}
                        .get('test_reward') if result else None
                    ),
                })
                overall_bar.update(1)

            n_ok = len(all_results[dwell_h][mname])
            logger.info(f"  ✓ {mname} @ {dwell_h}h: {n_ok}/{len(seeds)} seeds complete")

            # Incrementally persist aggregated results after every method
            _update_aggregated_raw(raw_path, all_results)
            logger.debug(f"  aggregated_raw.json updated -> {raw_path}")

    overall_bar.close()

    logger.info(f"\nRaw results: {raw_path}")
    logger.info(f"Progress log: {progress_path}")

    # ── Post-processing: stats + plots ────────────────────────────────────────
    try:
        from scripts.generate_dwell_plots import generate_all_dwell_plots
        logger.info("Running post-processing (plots + tables)…")
        generate_all_dwell_plots(all_results, agg_dir, cfg)
        logger.info(f"Plots and tables written to {agg_dir}/")
    except Exception as exc:
        logger.warning(f"Post-processing failed: {exc}", exc_info=True)

    return all_results, output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_reproducibility_notes(
    output_dir: str,
    seeds: list,
    dwell_hours: list,
    dev_mode: bool,
    logger: logging.Logger,
) -> None:
    notes = {
        'timestamp':       datetime.now().isoformat(),
        'study':           'SWIFT Dwell-Time Analysis',
        'seeds':           seeds,
        'n_seeds':         len(seeds),
        'dwell_scenarios': dwell_hours,
        'methods':         [m['name'] for m in STUDY_METHODS],
        'dev_mode':        dev_mode,
        'python_version':  sys.version,
        'numpy_version':   np.__version__,
        'torch_version':   torch.__version__,
        'cuda_available':  torch.cuda.is_available(),
        'rng_control': {
            'python_random': 'random.seed(seed)',
            'numpy':         'np.random.seed(seed)',
            'torch':         'torch.manual_seed(seed)',
            'torch_cuda':    'torch.cuda.manual_seed_all(seed)',
            'cudnn':         'deterministic=True',
            'env_hash':      'PYTHONHASHSEED=str(seed)',
        },
        'ci_formula': 'CI_95 = 1.96 * std / sqrt(n)',
        'design_note': (
            'dwell_time_hours controls both sim_hours (episode length) and '
            'all agents t_dep. Each episode models one complete EV charging '
            'session of the given duration. SWIFT utility scoring selects the '
            'top 60% of agents by staleness + SOC gap + diversity each round.'
        ),
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'reproducibility_notes.json')
    with open(path, 'w') as f:
        json.dump(notes, f, indent=2)
    logger.info(f"Reproducibility notes -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SWIFT Dwell-Time Study')
    parser.add_argument('--dev', action='store_true',
                        help='Use dev config (fewer episodes, faster)')

    # Full-study options
    parser.add_argument('--seeds',   type=int, nargs='+',
                        help='Override seed list  (e.g. --seeds 0 1 2)')
    parser.add_argument('--dwell',   type=int, nargs='+',
                        help='Override dwell hours (e.g. --dwell 1 2 4 6)')
    parser.add_argument('--methods', type=str, nargs='+',
                        help='Run only these methods (e.g. --methods "FedAvg-SAC" "SWIFT-SAC")')

    # Single-triple mode
    parser.add_argument('--single', action='store_true',
                        help='Run exactly one (method, dwell, seed) triple and exit')
    parser.add_argument('--method', type=str,
                        help='[--single] Method name: FedAvg-SAC | SWIFT-SAC | HFDRL')
    parser.add_argument('--dwell-single', type=int, dest='dwell_single',
                        help='[--single] Dwell hours: 1 | 2 | 4 | 6')
    parser.add_argument('--seed-single', type=int, dest='seed_single',
                        help='[--single] Seed integer')
    parser.add_argument('--output-base', type=str, dest='output_base',
                        help='Root output directory (overrides config)')

    args = parser.parse_args()

    if args.single:
        missing = [f for f, v in [('--method', args.method),
                                   ('--dwell-single', args.dwell_single),
                                   ('--seed-single', args.seed_single)] if v is None]
        if missing:
            parser.error(f"--single requires: {', '.join(missing)}")
        run_single_triple(
            method_name=args.method,
            dwell_hours=args.dwell_single,
            seed=args.seed_single,
            dev_mode=args.dev,
            output_base=args.output_base,
        )
    else:
        run_dwell_time_study(
            seeds=args.seeds,
            dwell_hours_list=args.dwell,
            dev_mode=args.dev,
            method_filter=args.methods,
            output_base=args.output_base,
        )
