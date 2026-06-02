"""
StressTestStudy.py
==================
Robustness stress tests for the HFDRL paper (AAAI — Robustness Analysis section).

Two sub-studies share this module:

  Sub-study A — Forecast Error
      Sweep Gaussian noise σ over the 5-hour price forecast (state[8:13]).
      The reward is computed on true prices; only the agent's observation is noised.
      Methods : FedAvg-SAC | SWIFT-SAC | HFDRL
      Scenarios: σ ∈ [0.0, 0.02, 0.05, 0.10, 0.20]  ($/kWh)

  Sub-study B — Non-IID Data
      Sweep Dirichlet concentration α over 3 driver archetypes, one allocation
      per FL edge. Lower α → stronger inter-edge heterogeneity.
      Methods : FedAvg-SAC | SWIFT-SAC | HFDRL
      Scenarios: α ∈ [1000, 10, 1.0, 0.5, 0.1]

Both sub-studies use 5 seeds (configurable) and follow the DwellTimeStudy
persistence pattern: per-seed npy + json, incremental progress.jsonl,
atomic aggregated_raw.json, post-processing plots + tables.

Usage
-----
    # Full study (both sub-studies, 5 seeds)
    python -m training.StressTestStudy

    # Forecast error only
    python -m training.StressTestStudy --sub forecast_error

    # Non-IID only, with ACN archetypes
    python -m training.StressTestStudy --sub non_iid --archetype acn

    # Single run
    python -m training.StressTestStudy --single --sub forecast_error \\
        --scenario 0.05 --method HFDRL --seed 0

    # Via interactive menu (main.py → option 19)
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

SUB_STUDIES = ('forecast_error', 'non_iid', 'both')


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup (identical to DwellTimeStudy)
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logger(output_dir: str) -> logging.Logger:
    logger = logging.getLogger('stress_study')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s  %(levelname)-7s  %(message)s',
                            datefmt='%H:%M:%S')

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
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    try:
        return get_config('stress_test_study')
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _append_progress(progress_path: str, entry: dict) -> None:
    with open(progress_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _update_aggregated_raw(raw_path: str, all_results: dict) -> None:
    """Rewrite aggregated_raw.json atomically (called after every method)."""
    agg_raw = {}
    for scenario_key, methods_dict in all_results.items():
        agg_raw[str(scenario_key)] = {}
        for mname, seed_list in methods_dict.items():
            agg_raw[str(scenario_key)][mname] = [
                {k: v for k, v in r.items() if not isinstance(v, list)}
                for r in seed_list
            ]
    tmp = raw_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(agg_raw, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, raw_path)


def _save_reproducibility_notes(
    output_dir: str,
    seeds: list,
    sub_study: str,
    scenarios: list,
    dev_mode: bool,
    archetype_set: str,
    logger: logging.Logger,
) -> None:
    notes = {
        'timestamp':      datetime.now().isoformat(),
        'study':          'Robustness Stress Tests',
        'sub_study':      sub_study,
        'seeds':          seeds,
        'n_seeds':        len(seeds),
        'scenarios':      scenarios,
        'methods':        [m['name'] for m in STUDY_METHODS],
        'dev_mode':       dev_mode,
        'archetype_set':  archetype_set,
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
        'design_notes': {
            'forecast_error': (
                'forecast_noise_std (σ in $/kWh) added to state[8:13] price forecast. '
                'Actual reward price is NOT noised. Same σ applied in train and test loops.'
            ),
            'non_iid': (
                'Dirichlet([α,α,α]) sampled per FL edge to determine driver-type proportions. '
                'Lower α → more skewed → stronger inter-edge heterogeneity. '
                f'Archetype set: {archetype_set}.'
            ),
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'reproducibility_notes.json')
    with open(path, 'w') as f:
        json.dump(notes, f, indent=2)
    logger.info(f"Reproducibility notes -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed runner  (shared by both sub-studies)
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_seed(
    method: dict,
    extra_kwargs: dict,
    seed: int,
    seed_dir: str,
    dev_mode: bool,
    logger: logging.Logger,
    show_progress: bool = False,
    tqdm_position: int = None,
    tqdm_leave: bool = False,
) -> dict:
    """Run one (method, scenario, seed) triple. Returns metric dict or None on failure."""
    set_global_seed(seed)

    kwargs = dict(
        policy=method['policy'],
        aggregation=method['aggregation'],
        use_swift=method['use_swift'],
        use_lora=method['use_lora'],
        verbose=False,
        progress_enabled=show_progress,
        dev_mode=dev_mode,
        tqdm_position=tqdm_position,
        tqdm_leave=tqdm_leave,
        **extra_kwargs,
    )

    t0 = time.time()
    try:
        metrics = run_single_experiment(**kwargs)
    except Exception as exc:
        logger.error(f"FAILED {method['name']} seed={seed}: {exc}", exc_info=True)
        return None

    wall_time = time.time() - t0
    result = extract_seed_metrics(metrics, seed)
    result['wall_time_s'] = round(wall_time, 2)

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

    logger.debug(f"  saved -> {seed_dir}/metrics.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Single-run entry point  (fully standalone)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_stress_run(
    sub_study: str,
    scenario_value: float,
    method_name: str,
    seed: int,
    dev_mode: bool = False,
    output_base: str = None,
    archetype_set: str = 'nhts',
) -> dict:
    """
    Run exactly one (sub_study, scenario, method, seed) combination.

    Parameters
    ----------
    sub_study      : 'forecast_error' or 'non_iid'
    scenario_value : σ ($/kWh) for forecast_error, or α for non_iid
    method_name    : One of 'FedAvg-SAC', 'SWIFT-SAC', 'HFDRL'
    seed           : Integer random seed
    dev_mode       : Use training_dev.yaml
    output_base    : Root directory (defaults to results/stress_test_study)
    archetype_set  : 'nhts' (default) or 'acn' — only used for non_iid
    """
    if method_name not in METHOD_BY_NAME:
        raise ValueError(f"Unknown method '{method_name}'. Choose from: {list(METHOD_BY_NAME)}")
    if sub_study not in ('forecast_error', 'non_iid'):
        raise ValueError(f"sub_study must be 'forecast_error' or 'non_iid', got '{sub_study}'")

    cfg = _load_cfg()
    if output_base is None:
        output_base = cfg.get('output_base', 'results/stress_test_study')

    method = METHOD_BY_NAME[method_name]
    safe_mname = method_name.replace(' ', '_').replace('/', '-')

    if sub_study == 'forecast_error':
        scenario_key = f'noise_{scenario_value:.2f}'
        extra_kwargs = {'forecast_noise_std': float(scenario_value)}
    else:
        scenario_key = f'alpha_{scenario_value}'
        extra_kwargs = {
            'non_iid_alpha': float(scenario_value),
            'archetype_set': archetype_set,
        }

    seed_dir = os.path.join(
        output_base, 'singles', sub_study, scenario_key, safe_mname, f'seed_{seed}'
    )
    os.makedirs(seed_dir, exist_ok=True)

    logger = _setup_logger(seed_dir)
    logger.info(f"Single run: sub={sub_study}  scenario={scenario_key}  "
                f"method={method_name}  seed={seed}  "
                f"mode={'dev' if dev_mode else 'full'}")
    logger.info(f"Output -> {seed_dir}")

    result = _run_one_seed(
        method=method,
        extra_kwargs=extra_kwargs,
        seed=seed,
        seed_dir=seed_dir,
        dev_mode=dev_mode,
        logger=logger,
        show_progress=True,
    )

    if result is not None:
        reward = result.get('test_reward')
        reward_str = f"{reward:.3f}" if reward is not None else "N/A"
        logger.info(f"Done — test_reward={reward_str}  wall={result['wall_time_s']:.1f}s")
    else:
        logger.error("Run failed — check study.log for details.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sub-study runner  (used internally by run_stress_test_study)
# ─────────────────────────────────────────────────────────────────────────────

def _run_sub_study(
    sub_study: str,
    scenarios: list,
    methods: list,
    seeds: list,
    output_dir: str,
    dev_mode: bool,
    logger: logging.Logger,
    archetype_set: str = 'nhts',
) -> dict:
    """
    Core loop for one sub-study (forecast_error or non_iid).

    Returns
    -------
    all_results : {scenario_key: {method_name: [seed_dicts]}}
    """
    sub_dir      = os.path.join(output_dir, sub_study)
    agg_dir      = os.path.join(sub_dir, 'aggregated')
    progress_path = os.path.join(sub_dir, 'progress.jsonl')
    raw_path      = os.path.join(sub_dir, 'aggregated_raw.json')

    for sub in ('tables', 'plots', 'statistics'):
        os.makedirs(os.path.join(agg_dir, sub), exist_ok=True)

    n_total = len(scenarios) * len(methods) * len(seeds)
    label = 'Forecast Error' if sub_study == 'forecast_error' else 'Non-IID Data'

    logger.info('─' * 60)
    logger.info(f'{label} sub-study')
    logger.info(f"Scenarios : {scenarios}")
    logger.info(f"Methods   : {[m['name'] for m in methods]}")
    logger.info(f"Seeds     : {seeds}")
    logger.info(f"Total runs: {n_total}")
    logger.info('─' * 60)

    all_results: dict = {}

    overall_bar = tqdm(
        total=n_total, desc=f'{label}', unit='run',
        ncols=90, position=0, leave=True,
    )

    for scenario in tqdm(scenarios, desc='Scenarios', unit='sc',
                         ncols=90, position=1, leave=False):
        if sub_study == 'forecast_error':
            scenario_key = f'noise_{scenario:.2f}'
            extra_kwargs = {'forecast_noise_std': float(scenario)}
        else:
            scenario_key = f'alpha_{scenario}'
            extra_kwargs = {
                'non_iid_alpha':  float(scenario),
                'archetype_set':  archetype_set,
            }

        all_results[scenario_key] = {}
        scenario_dir = os.path.join(sub_dir, scenario_key)
        os.makedirs(scenario_dir, exist_ok=True)

        logger.info(f"  ── scenario={scenario_key} ──")

        for method in tqdm(methods, desc=f'Methods @{scenario_key}', unit='method',
                           ncols=90, position=2, leave=False):
            mname = method['name']
            safe_mname = mname.replace(' ', '_').replace('/', '-')
            all_results[scenario_key][mname] = []
            method_dir = os.path.join(scenario_dir, safe_mname)

            for seed in tqdm(seeds, desc=mname, unit='seed',
                             ncols=90, position=3, leave=False):
                seed_dir = os.path.join(method_dir, f'seed_{seed}')
                logger.debug(f"  starting {mname} {scenario_key} seed={seed}")

                result = _run_one_seed(
                    method=method,
                    extra_kwargs=extra_kwargs,
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
                    all_results[scenario_key][mname].append(result)
                    scalars = {k: v for k, v in result.items() if not isinstance(v, list)}
                    logger.info(
                        f"  [{mname}] {scenario_key} seed={seed}  "
                        f"test_reward={scalars.get('test_reward', float('nan')):.3f}  "
                        f"wall={result['wall_time_s']:.1f}s"
                    )
                else:
                    logger.warning(f"  [{mname}] {scenario_key} seed={seed}  FAILED")

                _append_progress(progress_path, {
                    'ts':          datetime.now().isoformat(),
                    'sub_study':   sub_study,
                    'scenario':    scenario_key,
                    'method':      mname,
                    'seed':        seed,
                    'status':      status,
                    'wall_time_s': result['wall_time_s'] if result else None,
                    'test_reward': (
                        {k: v for k, v in result.items() if not isinstance(v, list)}
                        .get('test_reward') if result else None
                    ),
                })
                overall_bar.update(1)

            n_ok = len(all_results[scenario_key][mname])
            logger.info(f"  ✓ {mname} @ {scenario_key}: {n_ok}/{len(seeds)} seeds complete")

            _update_aggregated_raw(raw_path, all_results)
            logger.debug(f"  aggregated_raw.json updated -> {raw_path}")

    overall_bar.close()
    logger.info(f"Raw results: {raw_path}")

    # ── Post-processing ───────────────────────────────────────────────────────
    try:
        from scripts.generate_stress_test_plots import generate_sub_study_plots
        cfg = _load_cfg()
        generate_sub_study_plots(all_results, sub_study, agg_dir, cfg)
        logger.info(f"Plots and tables -> {agg_dir}/")
    except Exception as exc:
        logger.warning(f"Post-processing failed: {exc}", exc_info=True)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Full study pipeline  (public entry point)
# ─────────────────────────────────────────────────────────────────────────────

def run_stress_test_study(
    sub_study: str = 'both',
    seeds: list = None,
    dev_mode: bool = False,
    verbose: bool = True,
    output_base: str = None,
    method_filter: list = None,
    archetype_set: str = 'nhts',
    forecast_noise_levels: list = None,
    non_iid_alphas: list = None,
) -> tuple:
    """
    Run the full stress-test study with incremental saves.

    Parameters
    ----------
    sub_study            : 'forecast_error', 'non_iid', or 'both'
    seeds                : List of random seeds. None → read from config.
    dev_mode             : Use training_dev.yaml.
    verbose              : Print per-seed lines in addition to tqdm bars.
    output_base          : Root output directory. None → config default.
    method_filter        : If given, only run methods in this list.
    archetype_set        : 'nhts' (default) or 'acn' — for non_iid sub-study.
    forecast_noise_levels: Override config noise levels.
    non_iid_alphas       : Override config alpha list.

    Returns
    -------
    (results_dict, output_dir)
    results_dict keys: 'forecast_error' and/or 'non_iid'
    """
    cfg = _load_cfg()

    if seeds is None:
        seeds = cfg.get('seeds', [0, 1, 2, 42, 123])
    if output_base is None:
        output_base = cfg.get('output_base', 'results/stress_test_study')
    if forecast_noise_levels is None:
        forecast_noise_levels = cfg.get('forecast_noise_levels', [0.0, 0.02, 0.05, 0.10, 0.20])
    if non_iid_alphas is None:
        non_iid_alphas = cfg.get('non_iid_alphas', [1000.0, 10.0, 1.0, 0.5, 0.1])
    if archetype_set == 'nhts':
        archetype_set = cfg.get('archetype_set', 'nhts')

    methods = [m for m in STUDY_METHODS
               if method_filter is None or m['name'] in method_filter]

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_base, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    logger = _setup_logger(output_dir)

    # Determine which sub-studies to run
    run_forecast = sub_study in ('forecast_error', 'both')
    run_noniid   = sub_study in ('non_iid', 'both')

    all_scenarios = []
    if run_forecast:
        all_scenarios += [('forecast_error', float(v)) for v in forecast_noise_levels]
    if run_noniid:
        all_scenarios += [('non_iid', float(a)) for a in non_iid_alphas]

    _save_reproducibility_notes(
        output_dir, seeds, sub_study, all_scenarios, dev_mode, archetype_set, logger
    )

    logger.info('=' * 60)
    logger.info('STRESS TEST STUDY')
    logger.info(f"Sub-studies : {sub_study}")
    logger.info(f"Methods     : {[m['name'] for m in methods]}")
    logger.info(f"Seeds       : {seeds}")
    logger.info(f"Output      : {output_dir}")
    logger.info('=' * 60)

    results_dict = {}

    if run_forecast:
        results_dict['forecast_error'] = _run_sub_study(
            sub_study='forecast_error',
            scenarios=forecast_noise_levels,
            methods=methods,
            seeds=seeds,
            output_dir=output_dir,
            dev_mode=dev_mode,
            logger=logger,
            archetype_set=archetype_set,
        )

    if run_noniid:
        results_dict['non_iid'] = _run_sub_study(
            sub_study='non_iid',
            scenarios=non_iid_alphas,
            methods=methods,
            seeds=seeds,
            output_dir=output_dir,
            dev_mode=dev_mode,
            logger=logger,
            archetype_set=archetype_set,
        )

    logger.info(f"\n✓ Stress test study complete — output: {output_dir}")
    return results_dict, output_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robustness Stress Test Study')
    parser.add_argument('--dev', action='store_true',
                        help='Use dev config (fewer episodes, faster)')
    parser.add_argument('--sub', type=str, default='both',
                        choices=['forecast_error', 'non_iid', 'both'],
                        help='Which sub-study to run (default: both)')
    parser.add_argument('--archetype', type=str, default='nhts',
                        choices=['nhts', 'acn'],
                        help='Driver archetype set for non-IID (default: nhts)')
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='Override seed list')
    parser.add_argument('--noise-levels', type=float, nargs='+', dest='noise_levels',
                        help='Override forecast noise levels')
    parser.add_argument('--alphas', type=float, nargs='+',
                        help='Override Dirichlet alpha list')
    parser.add_argument('--methods', type=str, nargs='+',
                        help='Restrict methods (e.g. --methods HFDRL "SWIFT-SAC")')
    parser.add_argument('--output-base', type=str, dest='output_base',
                        help='Root output directory')

    # Single-run mode
    parser.add_argument('--single', action='store_true',
                        help='Run exactly one (sub, scenario, method, seed) and exit')
    parser.add_argument('--scenario', type=float,
                        help='[--single] scenario value: σ for forecast_error, α for non_iid')
    parser.add_argument('--method', type=str,
                        help='[--single] method name: FedAvg-SAC | SWIFT-SAC | HFDRL')
    parser.add_argument('--seed', type=int,
                        help='[--single] integer seed')

    args = parser.parse_args()

    if args.single:
        missing = [f for f, v in [('--sub', args.sub == 'both'),
                                   ('--scenario', args.scenario is None),
                                   ('--method', args.method is None),
                                   ('--seed', args.seed is None)] if v]
        if missing:
            parser.error(f"--single: please provide --scenario, --method, --seed "
                         f"and set --sub to 'forecast_error' or 'non_iid'")
        if args.sub == 'both':
            parser.error("--single requires --sub forecast_error or --sub non_iid (not 'both')")
        run_single_stress_run(
            sub_study=args.sub,
            scenario_value=args.scenario,
            method_name=args.method,
            seed=args.seed,
            dev_mode=args.dev,
            output_base=args.output_base,
            archetype_set=args.archetype,
        )
    else:
        run_stress_test_study(
            sub_study=args.sub,
            seeds=args.seeds,
            dev_mode=args.dev,
            method_filter=args.methods,
            output_base=args.output_base,
            archetype_set=args.archetype,
            forecast_noise_levels=args.noise_levels,
            non_iid_alphas=args.alphas,
        )
