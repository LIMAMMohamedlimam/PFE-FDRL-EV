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

    # Via interactive menu (main.py → option 17)
    python main.py
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime

import numpy as np
import torch

from utils.config_loader import get_config
from training.MultiSeedRunner import set_global_seed, extract_seed_metrics
from training.ComparisonPipeline import run_single_experiment


# ─────────────────────────────────────────────────────────────────────────────
# Study constants
# ─────────────────────────────────────────────────────────────────────────────

# Methods in the same format as training.yaml (converted to kwargs internally)
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


def _load_study_cfg() -> dict:
    try:
        return get_config('dwell_time_study')
    except Exception:
        return {}


def _method_kwargs(method: dict, dwell_hours: int, swift_min_stay: float) -> dict:
    """Build run_single_experiment kwargs for one method at one dwell scenario."""
    return dict(
        policy=method['policy'],
        aggregation=method['aggregation'],
        use_swift=method['use_swift'],
        use_lora=method['use_lora'],
        dwell_time_hours=dwell_hours,
        swift_min_stay_override=swift_min_stay if method['use_swift'] else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_seed(
    method: dict,
    dwell_hours: int,
    swift_min_stay: float,
    seed: int,
    seed_dir: str,
    dev_mode: bool,
    verbose: bool,
) -> dict:
    """Run one (method, dwell, seed) triple. Returns metric dict or None on failure."""
    set_global_seed(seed)
    kwargs = _method_kwargs(method, dwell_hours, swift_min_stay)

    t0 = time.time()
    try:
        metrics = run_single_experiment(
            verbose=False,
            progress_enabled=False,
            dev_mode=dev_mode,
            **kwargs,
        )
    except Exception as exc:
        print(f"      [FAILED] {method['name']} dwell={dwell_hours}h seed={seed}: {exc}")
        import traceback; traceback.print_exc()
        return None

    wall_time = time.time() - t0
    result = extract_seed_metrics(metrics, seed)
    result['wall_time_s'] = wall_time
    result['dwell_hours'] = dwell_hours

    os.makedirs(seed_dir, exist_ok=True)

    # Persist curves as .npy
    np.save(os.path.join(seed_dir, 'reward_curve.npy'),
            np.array(result['reward_curve'], dtype=np.float32))
    np.save(os.path.join(seed_dir, 'cost_curve.npy'),
            np.array(result['cost_curve'], dtype=np.float32))
    np.save(os.path.join(seed_dir, 'satisfaction_curve.npy'),
            np.array(result['satisfaction_curve'], dtype=np.float32))

    # Persist scalars as JSON
    scalars = {k: v for k, v in result.items() if not isinstance(v, list)}
    scalars['n_reward_episodes'] = len(result['reward_curve'])
    with open(os.path.join(seed_dir, 'metrics.json'), 'w') as f:
        json.dump(scalars, f, indent=2)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
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
    Run the full dwell-time study.

    Parameters
    ----------
    seeds           : List of random seeds. None → read from dwell_time_study.yaml.
    dwell_hours_list: List of dwell durations. None → read from config.
    dev_mode        : Use training_dev.yaml (fewer episodes, faster).
    verbose         : Print progress.
    output_base     : Root output directory. None → config default.
    method_filter   : If given, only run methods in this list.

    Returns
    -------
    (all_results, output_dir)
    all_results : {dwell_h: {method_name: [seed_metric_dicts]}}
    output_dir  : timestamped path where everything is saved
    """
    cfg = _load_study_cfg()

    if seeds is None:
        seeds = cfg.get('seeds', [0, 1, 2, 3, 4])
    if dwell_hours_list is None:
        dwell_hours_list = cfg.get('dwell_hours', [1, 2, 4, 6])
    if output_base is None:
        output_base = cfg.get('output_base', 'results/dwell_time_study')

    swift_min_stay_map = cfg.get('swift_min_stay', {
        1: 0.0, 2: 0.5, 4: 1.0, 6: 1.5
    })
    # YAML keys are strings; convert to int
    swift_min_stay_map = {int(k): float(v) for k, v in swift_min_stay_map.items()}

    methods = [m for m in STUDY_METHODS
               if method_filter is None or m['name'] in method_filter]

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_base, timestamp)
    agg_dir    = os.path.join(output_dir, 'aggregated')
    for sub in ('tables', 'plots', 'statistics'):
        os.makedirs(os.path.join(agg_dir, sub), exist_ok=True)

    _save_reproducibility_notes(output_dir, seeds, dwell_hours_list, dev_mode)

    n_total = len(dwell_hours_list) * len(methods) * len(seeds)
    print(f"\n{'='*68}")
    print(f"  DWELL-TIME STUDY")
    print(f"  Scenarios : {dwell_hours_list}h")
    print(f"  Methods   : {[m['name'] for m in methods]}")
    print(f"  Seeds     : {seeds}")
    print(f"  Total runs: {n_total}  ({'dev' if dev_mode else 'full'} mode)")
    print(f"  Output    : {output_dir}")
    print(f"{'='*68}\n")

    # {dwell_h: {method_name: [seed_metrics]}}
    all_results: dict = {}

    for dwell_h in dwell_hours_list:
        swift_min_stay = swift_min_stay_map.get(dwell_h, 0.0)
        dwell_key = dwell_h
        all_results[dwell_key] = {}

        dwell_dir = os.path.join(output_dir, f'dwell_{dwell_h}h')
        os.makedirs(dwell_dir, exist_ok=True)

        print(f"\n── Dwell = {dwell_h}h (SWIFT min_stay={swift_min_stay}h) ──")

        for method in methods:
            mname = method['name']
            safe_mname = mname.replace(' ', '_').replace('/', '-')
            all_results[dwell_key][mname] = []

            method_dir = os.path.join(dwell_dir, safe_mname)

            for idx, seed in enumerate(seeds):
                seed_dir = os.path.join(method_dir, f'seed_{seed}')
                if verbose:
                    print(f"  [{mname}] dwell={dwell_h}h  seed={seed}  ({idx+1}/{len(seeds)})")

                result = _run_one_seed(
                    method=method,
                    dwell_hours=dwell_h,
                    swift_min_stay=swift_min_stay,
                    seed=seed,
                    seed_dir=seed_dir,
                    dev_mode=dev_mode,
                    verbose=verbose,
                )
                if result is not None:
                    all_results[dwell_key][mname].append(result)

            n_ok = len(all_results[dwell_key][mname])
            print(f"  ✓ {mname} @ {dwell_h}h: {n_ok}/{len(seeds)} seeds")

    # ── Save aggregated raw JSON ───────────────────────────────────────────────
    agg_raw = {}
    for dwell_h, methods_dict in all_results.items():
        agg_raw[str(dwell_h)] = {}
        for mname, seed_list in methods_dict.items():
            agg_raw[str(dwell_h)][mname] = [
                {k: v for k, v in r.items() if not isinstance(v, list)}
                for r in seed_list
            ]
    raw_path = os.path.join(output_dir, 'aggregated_raw.json')
    with open(raw_path, 'w') as f:
        json.dump(agg_raw, f, indent=2)
    print(f"\n-> Raw results saved to {raw_path}")

    # ── Post-processing: stats + plots ────────────────────────────────────────
    try:
        from scripts.generate_dwell_plots import generate_all_dwell_plots
        generate_all_dwell_plots(all_results, agg_dir, cfg)
        print(f"\n✓ Plots and tables written to {agg_dir}/")
    except Exception as exc:
        print(f"\n[WARNING] Post-processing failed: {exc}")
        import traceback; traceback.print_exc()

    return all_results, output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_reproducibility_notes(
    output_dir: str,
    seeds: list,
    dwell_hours: list,
    dev_mode: bool,
) -> None:
    notes = {
        'timestamp':      datetime.now().isoformat(),
        'study':          'SWIFT Dwell-Time Analysis',
        'seeds':          seeds,
        'n_seeds':        len(seeds),
        'dwell_scenarios': dwell_hours,
        'methods':        [m['name'] for m in STUDY_METHODS],
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
    print(f"-> Reproducibility notes saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SWIFT Dwell-Time Study')
    parser.add_argument('--dev',    action='store_true',
                        help='Use dev config (fewer episodes, faster)')
    parser.add_argument('--seeds',  type=int, nargs='+',
                        help='Override seed list (e.g. --seeds 0 1 2)')
    parser.add_argument('--dwell',  type=int, nargs='+',
                        help='Override dwell hours (e.g. --dwell 1 2 4 6)')
    parser.add_argument('--methods', type=str, nargs='+',
                        help='Run only these methods (e.g. --methods "FedAvg-SAC" "SWIFT-SAC")')
    args = parser.parse_args()

    run_dwell_time_study(
        seeds=args.seeds,
        dwell_hours_list=args.dwell,
        dev_mode=args.dev,
        method_filter=args.methods,
    )
