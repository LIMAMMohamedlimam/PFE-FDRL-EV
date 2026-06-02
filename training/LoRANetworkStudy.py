"""
LoRANetworkStudy.py
===================
Dedicated runner for the LoRA Network Constraints study (AAAI paper).

Scientific objective
--------------------
Justify LoRA's existence by showing it becomes essential under realistic
low-bandwidth FL conditions.  The communication savings from LoRA far
outweigh its marginal reward degradation at 1–10 Mbps, while at 100 Mbps
the benefit is smaller but still acceptable.

Study design — Two-phase approach
-----------------------------------
  Phase A : RL training
      2 methods × N seeds via ComparisonPipeline.run_single_experiment()
      → reward quality, convergence speed, SoC, charging cost

  Phase B : Analytical communication overhead
      Create throwaway SACAgent instances, measure get_parameters() size,
      apply formula per bandwidth scenario — NO re-training needed.
      formula: time_s = (size_MB × 8 / bw_mbps) + (latency_ms / 1000)

NOTE: Bandwidth scenarios ONLY affect the analytical comm model (Phase B).
      RL training quality is bandwidth-independent in this simulator.

Usage
-----
    # Full AAAI run (10 seeds, both methods)
    python -m training.LoRANetworkStudy

    # Quick smoke test (dev mode, 2 seeds)
    python -m training.LoRANetworkStudy --dev --seeds 0 1

    # Single (method, seed) pair — fully standalone
    python -m training.LoRANetworkStudy --single --method "HFDRL + LoRA" --seed 0

    # Group run — only LoRA method
    python -m training.LoRANetworkStudy --dev --group lora_only

    # Via interactive menu (main.py → option 18)
    python main.py
"""

import os
import sys
import json
import time
import math
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
        'name': 'HFDRL (no LoRA)',
        'policy': 'sac',
        'aggregation': 'fedavg',
        'use_swift': True,
        'use_lora': False,
    },
    {
        'name': 'HFDRL + LoRA',
        'policy': 'sac',
        'aggregation': 'fedavg',
        'use_swift': True,
        'use_lora': True,
    },
]

METHOD_BY_NAME = {m['name']: m for m in STUDY_METHODS}

_DEFAULT_SCENARIOS = [
    {'name': '1_Mbps',   'agent_bw': 1,          'agent_lat': 100, 'label': '1 Mbps'},
    {'name': '5_Mbps',   'agent_bw': 5,           'agent_lat': 50,  'label': '5 Mbps'},
    {'name': '10_Mbps',  'agent_bw': 10,          'agent_lat': 20,  'label': '10 Mbps'},
    {'name': '100_Mbps', 'agent_bw': 100,         'agent_lat': 5,   'label': '100 Mbps'},
    {'name': 'variable', 'agent_bw': 'variable',  'bw_range': [1, 5, 10],
     'agent_lat': 50, 'label': 'Variable'},
]


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logger(output_dir: str) -> logging.Logger:
    """Dual-sink logger: tqdm-safe console (INFO) + file (DEBUG), flush every record."""
    logger = logging.getLogger('lora_network_study')
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
        return get_config('lora_network_study')
    except Exception:
        return {}


def _resolve_method_group(method_filter, cfg: dict) -> list:
    """Resolve method list from filter string, group key, or None (→ full)."""
    if method_filter is None:
        return list(STUDY_METHODS)

    run_groups = cfg.get('run_groups', {})

    if isinstance(method_filter, str):
        if method_filter in run_groups:
            names = run_groups[method_filter]
            return [m for m in STUDY_METHODS if m['name'] in names]
        method_filter = [method_filter]

    if isinstance(method_filter, list):
        # Check if it's a single group key
        if len(method_filter) == 1 and method_filter[0] in run_groups:
            names = run_groups[method_filter[0]]
            return [m for m in STUDY_METHODS if m['name'] in names]
        # Otherwise treat as method names
        return [m for m in STUDY_METHODS if m['name'] in method_filter]

    return list(STUDY_METHODS)


def _safe_name(name: str) -> str:
    return (name.replace(' ', '_').replace('/', '-')
                .replace('(', '').replace(')', '').replace('+', 'plus'))


# ─────────────────────────────────────────────────────────────────────────────
# Progress log helpers
# ─────────────────────────────────────────────────────────────────────────────

def _append_progress(progress_path: str, entry: dict) -> None:
    """Append one JSON line to the rolling progress log (fsynced)."""
    with open(progress_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _update_aggregated_raw(raw_path: str, all_results: dict) -> None:
    """Rewrite aggregated_raw.json atomically (scalar-only view)."""
    agg_raw = {}
    for mname, seed_list in all_results.items():
        agg_raw[mname] = [
            {k: v for k, v in r.items() if not isinstance(v, list)}
            for r in seed_list
        ]
    tmp = raw_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(agg_raw, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, raw_path)


# ─────────────────────────────────────────────────────────────────────────────
# Phase A — per-seed RL training runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_seed(
    method: dict,
    seed: int,
    seed_dir: str,
    dev_mode: bool,
    logger: logging.Logger,
    show_progress: bool = False,
) -> dict | None:
    """
    Run one (method, seed) pair via ComparisonPipeline.

    Returns metric dict (scalars + curve lists) or None on failure.
    """
    set_global_seed(seed)

    kwargs = dict(
        policy=method['policy'],
        aggregation=method['aggregation'],
        use_swift=method['use_swift'],
        use_lora=method['use_lora'],
    )

    t0 = time.time()
    try:
        metrics = run_single_experiment(
            verbose=False,
            progress_enabled=show_progress,
            dev_mode=dev_mode,
            **kwargs,
        )
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
# Phase B — model size measurement + analytical comm overhead
# ─────────────────────────────────────────────────────────────────────────────

def _measure_model_sizes(logger: logging.Logger) -> dict:
    """
    Instantiate throwaway SACAgent per method, call get_parameters(),
    measure byte size.  This is the EXACT payload transmitted in FL rounds.

    Returns {method_name: int_bytes}
    """
    from agents.SACAgent import SACAgent
    from env.EVClientEnv import EVClientEnv
    from network_sim.network_simulator import measure_params_size

    # Build a dummy env to probe state dimension (same pattern as ComparisonPipeline)
    dummy_env = EVClientEnv({
        'capacity': 60.0, 'max_power': 11.0,
        'initial_soc': 0.5, 'soc_req': 0.8, 't_dep': 10, 'dt': 1.0,
    })
    input_dim = len(dummy_env.get_state(0.0, 0.0, [0.0] * 5))

    sizes = {}
    for method in STUDY_METHODS:
        agent = SACAgent(input_dim=input_dim, action_dim=1,
                         use_lora=method['use_lora'])
        params = agent.get_parameters()
        size_b = measure_params_size(params)
        sizes[method['name']] = size_b
        logger.info(
            f"  Model size  [{method['name']}]: "
            f"{size_b:,} B  ({size_b / 1024:.1f} KiB)"
        )

    nolora = sizes.get('HFDRL (no LoRA)', 1)
    lora   = sizes.get('HFDRL + LoRA',   0)
    if nolora > 0:
        logger.info(
            f"  LoRA reduction: {(1 - lora / nolora) * 100:.1f}% "
            f"({nolora:,} B → {lora:,} B)"
        )
    return sizes


def _compute_comm_overhead(
    model_sizes: dict,
    scenarios: list,
    n_agents: int,
    logger: logging.Logger,
) -> dict:
    """
    Analytically compute communication overhead per (scenario × method).

    For variable bandwidth: average transfer time over bw_range.
    Agents → Cloud (cloud-only topology) — same model as network_sim formula.

    Returns:
      {scenario_name: {method_name: {
          bytes_per_agent_per_round,
          time_per_agent_per_round_s,
          bytes_total_n_agents,
          time_total_n_agents_s,
          bw_mbps_used,
          latency_ms,
      }}}
    """
    result = {}
    for sc in scenarios:
        sc_name = sc['name']
        result[sc_name] = {}
        logger.debug(f"  Computing comm overhead for scenario: {sc_name}")

        bw_list = sc['bw_range'] if sc['agent_bw'] == 'variable' else [sc['agent_bw']]
        lat_ms  = sc['agent_lat']

        for method in STUDY_METHODS:
            mname = method['name']
            if mname not in model_sizes:
                continue
            size_b  = model_sizes[mname]
            size_mb = size_b / (1024 * 1024)

            times = [(size_mb * 8 / bw) + (lat_ms / 1000.0) for bw in bw_list]
            avg_time = float(np.mean(times))

            result[sc_name][mname] = {
                'bytes_per_agent_per_round':  size_b,
                'time_per_agent_per_round_s': round(avg_time, 6),
                'bytes_total_n_agents':       size_b * n_agents,
                'time_total_n_agents_s':      round(avg_time * n_agents, 4),
                'bw_mbps_used':               bw_list,
                'latency_ms':                 lat_ms,
            }

    # Log a quick comparison table
    methods_present = [m['name'] for m in STUDY_METHODS if m['name'] in model_sizes]
    header = f"{'Scenario':<12}" + "".join(f"  {m[:18]:<20}" for m in methods_present)
    logger.info("  Simulated time per agent per round (seconds):")
    logger.info(f"  {header}")
    for sc in scenarios:
        row = f"  {sc['name']:<12}"
        for m in methods_present:
            t = result.get(sc['name'], {}).get(m, {}).get('time_per_agent_per_round_s', float('nan'))
            row += f"  {t:<22.4f}"
        logger.info(row)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Single-pair entry point (standalone / CI debug mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_pair(
    method_name: str,
    seed: int,
    dev_mode: bool = False,
    output_base: str = None,
) -> dict | None:
    """
    Run exactly one (method, seed) pair independently.

    Shows all tqdm bars + live log to console.  Prints a summary table at end.
    Saves artifacts to <output_base>/single_runs/<method_safe>_seed<N>/.

    Returns metric dict or None on failure.
    """
    if method_name not in METHOD_BY_NAME:
        raise ValueError(
            f"Unknown method '{method_name}'. Valid: {list(METHOD_BY_NAME)}"
        )

    cfg = _load_study_cfg()
    if output_base is None:
        output_base = cfg.get('output_base', 'results/lora_network_study')

    method    = METHOD_BY_NAME[method_name]
    safe      = _safe_name(method_name)
    seed_dir  = os.path.join(output_base, 'single_runs', f'{safe}_seed{seed}')
    os.makedirs(seed_dir, exist_ok=True)

    logger = _setup_logger(seed_dir)
    logger.info(f"{'=' * 55}")
    logger.info(f"SINGLE PAIR RUN")
    logger.info(f"  method : {method_name}")
    logger.info(f"  seed   : {seed}")
    logger.info(f"  mode   : {'dev' if dev_mode else 'full'}")
    logger.info(f"  output : {seed_dir}")
    logger.info(f"{'=' * 55}")

    result = _run_one_seed(
        method=method,
        seed=seed,
        seed_dir=seed_dir,
        dev_mode=dev_mode,
        logger=logger,
        show_progress=True,
    )

    if result is not None:
        scalars = {k: v for k, v in result.items() if not isinstance(v, list)}
        logger.info(f"{'─' * 55}")
        logger.info(f"  test_reward      : {scalars.get('test_reward', float('nan')):.4f}")
        logger.info(f"  final_soc        : {scalars.get('final_soc', float('nan')):.4f}")
        logger.info(f"  charging_cost    : {scalars.get('charging_cost', float('nan')):.4f}")
        logger.info(f"  conv_episode     : {scalars.get('convergence_episode', -1)}")
        logger.info(f"  wall_time        : {result['wall_time_s']:.1f}s")
        logger.info(f"{'─' * 55}")
    else:
        logger.error("Run FAILED — check study.log for details.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility notes
# ─────────────────────────────────────────────────────────────────────────────

def _save_reproducibility_notes(
    output_dir: str,
    seeds: list,
    methods: list,
    scenarios: list,
    dev_mode: bool,
    logger: logging.Logger,
) -> None:
    notes = {
        'timestamp':        datetime.now().isoformat(),
        'study':            'LoRA Network Constraints Study',
        'study_design':     (
            'Two-phase: '
            'Phase A = RL training (2 methods × N seeds via ComparisonPipeline); '
            'Phase B = analytical comm overhead from measured model sizes per BW scenario.'
        ),
        'formula':          'time_s = (size_MB * 8 / bw_mbps) + (latency_ms / 1000)',
        'note_bandwidth':   (
            'Bandwidth scenarios are ANALYTICAL ONLY — no real network throttling. '
            'RL training quality (reward, SoC, cost) is identical across all BW scenarios.'
        ),
        'seeds':            seeds,
        'n_seeds':          len(seeds),
        'methods':          [m['name'] for m in methods],
        'bandwidth_scenarios': [sc['name'] for sc in scenarios],
        'n_training_runs':  len(methods) * len(seeds),
        'dev_mode':         dev_mode,
        'python_version':   sys.version,
        'numpy_version':    np.__version__,
        'torch_version':    torch.__version__,
        'cuda_available':   torch.cuda.is_available(),
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
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'reproducibility_notes.json')
    with open(path, 'w') as f:
        json.dump(notes, f, indent=2)
    logger.info(f"Reproducibility notes -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Full study pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_lora_network_study(
    seeds: list = None,
    dev_mode: bool = False,
    verbose: bool = True,
    output_base: str = None,
    method_filter=None,
    no_plots: bool = False,
) -> tuple:
    """
    Run the full LoRA Network Constraints Study.

    Parameters
    ----------
    seeds         : Seed list.  None → read from lora_network_study.yaml.
    dev_mode      : Use training_dev.yaml (fewer episodes, faster).
    verbose       : Print per-seed lines in addition to tqdm bars.
    output_base   : Root output directory.  None → config default.
    method_filter : str group key ('lora_only'/'nolora_only'/'full'),
                    list of method names, or None (runs all).
    no_plots      : Skip post-processing plot generation.

    Returns
    -------
    (all_results, comm_analysis, output_dir)
      all_results   : {method_name: [seed_metric_dicts]}
      comm_analysis : {'model_sizes': {...}, 'bandwidth_scenarios': {...}}
      output_dir    : Path to timestamped run directory
    """
    cfg = _load_study_cfg()

    if seeds is None:
        seeds = cfg.get('seeds', [0, 1, 2, 3, 4])
    if output_base is None:
        output_base = cfg.get('output_base', 'results/lora_network_study')

    scenarios = cfg.get('bandwidth_scenarios', _DEFAULT_SCENARIOS)
    n_agents  = cfg.get('n_agents', 20)
    methods   = _resolve_method_group(method_filter, cfg)

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_base, timestamp)
    agg_dir    = os.path.join(output_dir, 'aggregated')
    comm_dir   = os.path.join(output_dir, 'comm_analysis')
    for sub in ('tables', 'plots', 'statistics'):
        os.makedirs(os.path.join(agg_dir, sub), exist_ok=True)
    os.makedirs(comm_dir, exist_ok=True)

    logger        = _setup_logger(output_dir)
    progress_path = os.path.join(output_dir, 'progress.jsonl')
    raw_path      = os.path.join(output_dir, 'aggregated_raw.json')

    _save_reproducibility_notes(output_dir, seeds, methods, scenarios, dev_mode, logger)

    n_total = len(methods) * len(seeds)

    logger.info('=' * 60)
    logger.info('LORA NETWORK CONSTRAINTS STUDY')
    logger.info(f"Methods   : {[m['name'] for m in methods]}")
    logger.info(f"Seeds     : {seeds}")
    logger.info(f"BW scen.  : {[sc['name'] for sc in scenarios]}")
    logger.info(f"Total runs: {n_total}  ({'dev' if dev_mode else 'full'} mode)")
    logger.info(f"Output    : {output_dir}")
    logger.info('=' * 60)

    all_results: dict = {}

    # ── Phase A: RL training ──────────────────────────────────────────────────
    logger.info('\n── Phase A: RL Training ──')

    overall_bar = tqdm(
        total=n_total,
        desc='Overall',
        unit='run',
        ncols=90,
        position=0,
        leave=True,
    )

    for method in tqdm(methods, desc='Methods', unit='method',
                       ncols=90, position=1, leave=False):
        mname    = method['name']
        safe     = _safe_name(mname)
        all_results[mname] = []
        method_dir = os.path.join(output_dir, safe)

        logger.info(f"── Method: {mname} ──")

        for seed in tqdm(seeds, desc=mname[:22], unit='seed',
                         ncols=90, position=2, leave=False):
            seed_dir = os.path.join(method_dir, f'seed_{seed}')

            logger.debug(f"  starting {mname} seed={seed}")
            t_seed = time.time()

            result = _run_one_seed(
                method=method,
                seed=seed,
                seed_dir=seed_dir,
                dev_mode=dev_mode,
                logger=logger,
            )

            status = 'ok' if result is not None else 'failed'
            if result is not None:
                all_results[mname].append(result)
                scalars = {k: v for k, v in result.items() if not isinstance(v, list)}
                if verbose:
                    logger.info(
                        f"  [{mname}] seed={seed}  "
                        f"reward={scalars.get('test_reward', float('nan')):.3f}  "
                        f"wall={result['wall_time_s']:.1f}s"
                    )
            else:
                logger.warning(f"  [{mname}] seed={seed}  FAILED")

            _append_progress(progress_path, {
                'ts':           datetime.now().isoformat(),
                'phase':        'A',
                'method':       mname,
                'seed':         seed,
                'status':       status,
                'wall_time_s':  result['wall_time_s'] if result else None,
                'test_reward':  (
                    {k: v for k, v in result.items() if not isinstance(v, list)}
                    .get('test_reward') if result else None
                ),
            })
            overall_bar.update(1)

        n_ok = len(all_results[mname])
        logger.info(f"  ✓ {mname}: {n_ok}/{len(seeds)} seeds complete")
        _update_aggregated_raw(raw_path, all_results)
        logger.debug(f"  aggregated_raw.json updated -> {raw_path}")

    overall_bar.close()

    logger.info(f"\nRaw results -> {raw_path}")
    logger.info(f"Progress log -> {progress_path}")

    # ── Phase B: Communication overhead ──────────────────────────────────────
    logger.info('\n── Phase B: Analytical Communication Overhead ──')

    model_sizes = _measure_model_sizes(logger)
    model_sizes_path = os.path.join(comm_dir, 'model_sizes.json')
    with open(model_sizes_path, 'w') as f:
        json.dump(model_sizes, f, indent=2)
    logger.info(f"Model sizes -> {model_sizes_path}")

    comm_analysis = _compute_comm_overhead(model_sizes, scenarios, n_agents, logger)
    bw_path = os.path.join(comm_dir, 'bandwidth_scenarios.json')
    with open(bw_path, 'w') as f:
        json.dump(comm_analysis, f, indent=2)
    logger.info(f"BW scenario overhead -> {bw_path}")

    # Also append Phase B summary to progress log
    _append_progress(progress_path, {
        'ts':     datetime.now().isoformat(),
        'phase':  'B',
        'status': 'ok',
        'model_sizes': model_sizes,
    })

    combined_comm = {'model_sizes': model_sizes, 'bandwidth_scenarios': comm_analysis}

    # ── Post-processing: plots + tables ──────────────────────────────────────
    if not no_plots:
        try:
            _proj_root = str(Path(__file__).resolve().parent.parent)
            if _proj_root not in sys.path:
                sys.path.insert(0, _proj_root)
            from scripts.generate_lora_network_plots import generate_all_lora_network_plots
            logger.info('\n── Post-processing (plots + tables) ──')
            generate_all_lora_network_plots(all_results, combined_comm, agg_dir, cfg)
            logger.info(f"Plots and tables -> {agg_dir}/")
        except Exception as exc:
            logger.warning(f"Post-processing failed: {exc}", exc_info=True)

    logger.info('\n' + '=' * 60)
    logger.info('STUDY COMPLETE')
    logger.info(f"Output -> {output_dir}")
    logger.info('=' * 60)

    return all_results, combined_comm, output_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LoRA Network Constraints Study',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--dev', action='store_true',
                        help='Use dev config (fewer episodes, faster)')

    # Full-study options
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='Override seed list  (e.g. --seeds 0 1 2)')
    parser.add_argument('--methods', type=str, nargs='+',
                        help='Run only these methods by name\n'
                             '  e.g. --methods "HFDRL + LoRA"')
    parser.add_argument('--group', type=str,
                        choices=['lora_only', 'nolora_only', 'full'],
                        help='Run a preset method group from config\n'
                             '  lora_only | nolora_only | full (default)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip post-processing plot generation')
    parser.add_argument('--output-base', type=str, dest='output_base',
                        help='Root output directory (overrides config)')

    # Single-pair mode
    parser.add_argument('--single', action='store_true',
                        help='Run exactly one (method, seed) pair and exit')
    parser.add_argument('--method', type=str,
                        help='[--single] Method: "HFDRL (no LoRA)" | "HFDRL + LoRA"')
    parser.add_argument('--seed', type=int,
                        help='[--single] Seed integer')

    args = parser.parse_args()

    if args.single:
        missing = [name for name, val in [('--method', args.method), ('--seed', args.seed)]
                   if val is None]
        if missing:
            parser.error(f"--single requires: {', '.join(missing)}")
        run_single_pair(
            method_name=args.method,
            seed=args.seed,
            dev_mode=args.dev,
            output_base=args.output_base,
        )
    else:
        method_filter = args.group or args.methods
        run_lora_network_study(
            seeds=args.seeds,
            dev_mode=args.dev,
            method_filter=method_filter,
            no_plots=args.no_plots,
            output_base=args.output_base,
        )
