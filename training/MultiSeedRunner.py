"""
MultiSeedRunner.py
==================
Run every method from training.yaml over multiple random seeds to produce
statistically rigorous results for AAAI publication.

Usage (pipeline):
    from training.MultiSeedRunner import run_multiseed_pipeline
    all_results, out_dir = run_multiseed_pipeline(seeds=[0,1,2,3,4])

Usage (standalone):
    python -m training.MultiSeedRunner --dev

Output structure:
    results/multi_seed/<timestamp>/
        reproducibility_notes.json
        aggregated_raw.json
        <method_name>/
            seed_<N>/
                metrics.json
                reward_curve.npy
                cost_curve.npy
                satisfaction_curve.npy
        aggregated/
            tables/  plots/  statistics/   (written by StatisticalAnalysis)
"""

import os
import sys
import json
import random
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from utils.config_loader import get_config
from utils.constants import CONVERGENCE_WINDOW, CONVERGENCE_THRESHOLD
from training.BaseStudy import BaseStudy
from training.ComparisonPipeline import run_single_experiment, _method_cfg_to_kwargs

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Seed management
# ─────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    """Fix all RNG sources for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction from EvalMetrics
# ─────────────────────────────────────────────────────────────────────────────

def _convergence_episode(
    rewards: list,
    window: int = CONVERGENCE_WINDOW,
    threshold: float = CONVERGENCE_THRESHOLD,
) -> int:
    """
    First episode where the rolling mean reaches `threshold` × final rolling mean.
    Returns -1 if the curve never converges by this criterion.
    """
    if len(rewards) < window + 1:
        return -1
    arr = np.array(rewards, dtype=float)
    smoothed = np.convolve(arr, np.ones(window) / window, mode='valid')
    final_avg = float(np.mean(smoothed[-window:])) if len(smoothed) >= window else float(smoothed[-1])
    if final_avg == 0.0:
        return -1
    target = threshold * abs(final_avg)
    for i, v in enumerate(smoothed):
        if abs(v) >= target:
            return int(i + window)
    return -1


def extract_seed_metrics(metrics, seed: int) -> dict:
    """
    Flatten all relevant scalar and curve data from an EvalMetrics object.

    Returns a dict with:
      - scalar metrics  (for statistical analysis)
      - curve arrays    (for learning-curve plots)
    """
    rewards    = list(metrics.episode_rewards)
    costs      = list(metrics.episode_costs)
    test_rew   = list(metrics.test_rewards)
    sat        = list(metrics.satisfaction_history)
    volt_viols = list(metrics.voltage_violations)
    comm_ms    = list(metrics.comm_overhead_ms)
    final_soc  = list(metrics.final_soc_per_test)

    mean_test_reward      = float(np.mean(test_rew))          if test_rew    else float('nan')
    mean_final_soc        = float(np.mean(final_soc))         if final_soc   else float('nan')
    mean_cost             = float(np.mean(costs[-20:]))        if costs       else float('nan')
    voltage_violation_rate = float(np.mean(volt_viols))       if volt_viols  else float('nan')
    mean_comm_ms          = float(np.mean(comm_ms))           if comm_ms     else float('nan')
    conv_ep               = _convergence_episode(rewards)

    return {
        # ── scalars ──────────────────────────────────────────────────────
        'seed':                   seed,
        'test_reward':            mean_test_reward,
        'final_soc':              mean_final_soc,
        'charging_cost':          mean_cost,
        'voltage_violation_rate': voltage_violation_rate,
        'comm_overhead_ms':       mean_comm_ms,
        'convergence_episode':    conv_ep,
        # ── curves (for plotting) ─────────────────────────────────────────
        'reward_curve':      rewards,
        'cost_curve':        costs,
        'satisfaction_curve': sat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-method multi-seed runner
# ─────────────────────────────────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


def run_multiseed_experiment(
    method_name: str,
    method_kwargs: dict,
    seeds: list,
    output_dir: str,
    dev_mode: bool = False,
    verbose: bool = True,
) -> list:
    """
    Run one method over all seeds; persist per-seed data; return list of metric dicts.
    """
    safe_name = method_name.replace(' ', '_').replace('/', '-')
    method_dir = os.path.join(output_dir, safe_name)
    seed_results = []

    seed_pbar = tqdm(
        enumerate(seeds), total=len(seeds),
        desc=f"  Seeds ({method_name})",
        unit="seed",
        leave=True,
        disable=not verbose,
    )

    method_t0 = time.time()

    for idx, seed in seed_pbar:
        seed_label = f"seed_{seed}"
        seed_dir   = os.path.join(method_dir, seed_label)
        os.makedirs(seed_dir, exist_ok=True)

        seed_pbar.set_postfix_str(f"seed={seed}")
        if verbose:
            tqdm.write(f"    ▶ [{method_name}] Starting seed={seed}  ({idx+1}/{len(seeds)})")

        set_global_seed(seed)

        try:
            t_start = time.time()
            metrics = run_single_experiment(
                verbose=verbose,
                progress_enabled=verbose,
                dev_mode=dev_mode,
                **method_kwargs,
            )
            wall_time = time.time() - t_start

            result = extract_seed_metrics(metrics, seed)
            result['wall_time_s'] = wall_time

            BaseStudy.save_seed_artifacts(result, seed_dir)
            seed_results.append(result)

            # ── per-seed summary ────────────────────────────────────────────
            if verbose:
                sat_vals = result.get('satisfaction_curve', [])
                avg_sat = float(np.mean(sat_vals[-10:])) if sat_vals else float('nan')
                tqdm.write(
                    f"    ✓ [{method_name}] seed={seed} done in {_format_duration(wall_time)} │ "
                    f"TestReward={result['test_reward']:.2f}  "
                    f"FinalSOC={result['final_soc']:.3f}  "
                    f"Cost=${result['charging_cost']:.2f}  "
                    f"ConvEp={result['convergence_episode']}  "
                    f"Satisfaction={avg_sat:.3f}"
                )

        except Exception as exc:
            tqdm.write(f"    ✗ [FAILED] {method_name} seed={seed}: {exc}")
            import traceback; traceback.print_exc()

    seed_pbar.close()

    # ── method-level summary ────────────────────────────────────────────────
    method_elapsed = time.time() - method_t0
    if verbose and seed_results:
        test_rews = [r['test_reward'] for r in seed_results]
        costs     = [r['charging_cost'] for r in seed_results]
        socs      = [r['final_soc'] for r in seed_results]
        times     = [r['wall_time_s'] for r in seed_results]
        tqdm.write(
            f"\n    ── {method_name} summary ({len(seed_results)}/{len(seeds)} seeds, "
            f"total {_format_duration(method_elapsed)}) ──\n"
            f"       TestReward : {np.mean(test_rews):8.2f} ± {np.std(test_rews):.2f}\n"
            f"       FinalSOC   : {np.mean(socs):8.3f} ± {np.std(socs):.3f}\n"
            f"       Cost       : ${np.mean(costs):7.2f} ± {np.std(costs):.2f}\n"
            f"       WallTime   : {np.mean(times):8.1f}s ± {np.std(times):.1f}s per seed"
        )

    return seed_results


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_multiseed_pipeline(
    seeds: list = None,
    dev_mode: bool = False,
    verbose: bool = True,
    output_base: str = None,
    method_filter: list = None,
) -> tuple:
    """
    Run every method in training.yaml over multiple seeds.

    Args:
        seeds:         List of integer seeds.  If None, read from multiseed.yaml.
        dev_mode:      Use training_dev.yaml parameters (fewer episodes).
        verbose:       Print per-seed progress.
        output_base:   Root output path; defaults to multiseed.yaml setting.
        method_filter: If given, only run methods whose names appear in this list.

    Returns:
        (all_results, output_dir)
        all_results: dict {method_name: [seed_metric_dicts]}
        output_dir:  path to the timestamped run directory
    """
    ms_cfg = _load_multiseed_cfg()
    if seeds is None:
        seeds = ms_cfg.get('seeds', [0, 1, 2, 3, 4])
    if output_base is None:
        output_base = ms_cfg.get('output_base', 'results/multi_seed')

    # Always read methods from training.yaml (dev only changes episode count)
    train_cfg = get_config('training')
    methods   = train_cfg.get('methods', [])

    if method_filter:
        methods = [m for m in methods if m.get('name') in method_filter]

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_base, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    agg_dir    = os.path.join(output_dir, 'aggregated')
    for sub in ('tables', 'plots', 'statistics'):
        os.makedirs(os.path.join(agg_dir, sub), exist_ok=True)

    _save_reproducibility_notes(output_dir, seeds, dev_mode)

    n_seeds   = len(seeds)
    n_methods = len(methods)
    total_runs = n_methods * n_seeds
    logger.info('=' * 65)
    logger.info(f"MULTI-SEED PIPELINE: {n_methods} methods × {n_seeds} seeds = {total_runs} runs")
    logger.info(f"Seeds : {seeds}")
    logger.info(f"Mode  : {'dev' if dev_mode else 'full'}")
    logger.info(f"Output: {output_dir}")
    logger.info('=' * 65)

    pipeline_t0 = time.time()
    all_results = {}

    method_pbar = tqdm(
        methods, desc="Methods", unit="method",
        leave=True, disable=not verbose,
    )

    for m_idx, method in enumerate(method_pbar):
        name   = method.get('name', 'unknown')
        kwargs = _method_cfg_to_kwargs(method)
        method_pbar.set_description(f"Methods [{name}]")
        tqdm.write(f"\n{'─'*60}")
        tqdm.write(f"  ▶ Method {m_idx+1}/{n_methods}: {name}")
        tqdm.write(f"{'─'*60}")
        seed_results = run_multiseed_experiment(
            method_name=name,
            method_kwargs=kwargs,
            seeds=seeds,
            output_dir=output_dir,
            dev_mode=dev_mode,
            verbose=verbose,
        )
        all_results[name] = seed_results

        elapsed = time.time() - pipeline_t0
        completed_methods = m_idx + 1
        if completed_methods < n_methods:
            eta = (elapsed / completed_methods) * (n_methods - completed_methods)
            tqdm.write(f"  ✓ {name}: {len(seed_results)}/{n_seeds} seeds │ "
                       f"Pipeline ETA: ~{_format_duration(eta)}")
        else:
            tqdm.write(f"  ✓ {name}: {len(seed_results)}/{n_seeds} seeds")

    method_pbar.close()

    # ── save aggregated raw JSON (scalars only) ────────────────────────────
    agg_raw = {}
    for name, results in all_results.items():
        agg_raw[name] = [{k: v for k, v in r.items() if not isinstance(v, list)}
                         for r in results]
    with open(os.path.join(output_dir, 'aggregated_raw.json'), 'w') as f:
        json.dump(agg_raw, f, indent=2)

    pipeline_elapsed = time.time() - pipeline_t0
    logger.info('=' * 65)
    logger.info(f"PIPELINE COMPLETE in {_format_duration(pipeline_elapsed)}")
    logger.info(f"Raw results saved to {output_dir}/aggregated_raw.json")
    logger.info('=' * 65)

    # ── run statistical analysis + plotting ───────────────────────────────
    try:
        from utils.StatisticalAnalysis import StatisticalAnalysis
        from scripts.generate_multiseed_plots import generate_all_plots

        logger.info("Running statistical analysis & plotting...")
        analysis = StatisticalAnalysis(all_results, agg_dir, ms_cfg)
        stats    = analysis.compute_and_save()
        generate_all_plots(all_results, stats, agg_dir, ms_cfg)
        logger.info(f"Statistical analysis and plots written to {agg_dir}/")
    except Exception as exc:
        logger.warning(f"Post-processing failed: {exc}", exc_info=True)

    return all_results, output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_multiseed_cfg() -> dict:
    try:
        return get_config('multiseed')
    except Exception:
        return {}


def _save_reproducibility_notes(output_dir: str, seeds: list, dev_mode: bool) -> None:
    notes = BaseStudy.base_repro_notes(seeds, dev_mode)
    notes['cudnn_deterministic'] = (
        torch.backends.cudnn.deterministic if torch.cuda.is_available() else 'N/A'
    )
    BaseStudy.write_repro_notes(output_dir, notes, logger)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-seed statistical evaluation')
    parser.add_argument('--dev',     action='store_true', help='Use dev config (fewer episodes)')
    parser.add_argument('--seeds',   type=int, nargs='+', help='Override seeds list')
    parser.add_argument('--methods', type=str, nargs='+', help='Run only these method names')
    args = parser.parse_args()

    run_multiseed_pipeline(
        seeds=args.seeds,
        dev_mode=args.dev,
        method_filter=args.methods,
    )
