"""
StatisticalAnalysis.py
======================
Aggregate multi-seed results into publication-quality statistical summaries:
  - Mean, Std, 95% CI for every metric
  - Paired t-test and Wilcoxon signed-rank vs HFDRL
  - LaTeX + CSV tables (Metric | Mean | Std | 95% CI)
  - JSON significance report

Usage (from MultiSeedRunner):
    analysis = StatisticalAnalysis(all_results, agg_dir, ms_cfg)
    stats    = analysis.compute_and_save()

Usage (standalone, reload from disk):
    analysis = StatisticalAnalysis.from_disk('results/multi_seed/<timestamp>')
    stats    = analysis.compute_and_save()
"""

import os
import json
import math
import csv

import numpy as np
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

HFDRL_NAME = 'SAC HFedAvg SWIFT LoRA'   # our proposed method

METRICS = [
    'test_reward',
    'final_soc',
    'charging_cost',
    'voltage_violation_rate',
    'comm_overhead_ms',
    'convergence_episode',
]

METRIC_DISPLAY = {
    'test_reward':            'Test Reward',
    'final_soc':              'Final SoC Ratio',
    'charging_cost':          'Charging Cost (\\$)',
    'voltage_violation_rate': 'Volt. Viol. Rate',
    'comm_overhead_ms':       'Comm. Overhead (ms)',
    'convergence_episode':    'Convergence Ep.',
}

METRIC_DISPLAY_PLAIN = {
    'test_reward':            'Test Reward',
    'final_soc':              'Final SoC Ratio',
    'charging_cost':          'Charging Cost ($)',
    'voltage_violation_rate': 'Volt. Viol. Rate',
    'comm_overhead_ms':       'Comm. Overhead (ms)',
    'convergence_episode':    'Convergence Ep.',
}

# Higher-is-better flag (for significance direction in tables)
METRIC_HIGHER_BETTER = {
    'test_reward':            True,
    'final_soc':              True,
    'charging_cost':          False,
    'voltage_violation_rate': False,
    'comm_overhead_ms':       False,
    'convergence_episode':    False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class StatisticalAnalysis:
    """
    Compute and save statistical summaries for multi-seed HFDRL experiments.

    Parameters
    ----------
    all_results : dict {method_name: [seed_metric_dicts]}
    output_dir  : base directory for tables/, statistics/ sub-folders
    ms_cfg      : dict from multiseed.yaml
    """

    def __init__(self, all_results: dict, output_dir: str, ms_cfg: dict = None):
        self.all_results = all_results
        self.output_dir  = output_dir
        self.ms_cfg      = ms_cfg or {}
        self.ci_z        = float(self.ms_cfg.get('ci_z', 1.96))
        self.alpha       = float(self.ms_cfg.get('alpha', 0.05))
        self.stats       = {}                # populated by compute()

    # ── constructor from disk ─────────────────────────────────────────────────

    @classmethod
    def from_disk(cls, run_dir: str, ms_cfg: dict = None):
        """Reload all_results from aggregated_raw.json inside a multi-seed run dir."""
        json_path = os.path.join(run_dir, 'aggregated_raw.json')
        with open(json_path) as f:
            raw = json.load(f)
        # raw dict has scalar metrics only; curves are unavailable but not needed here
        agg_dir = os.path.join(run_dir, 'aggregated')
        os.makedirs(os.path.join(agg_dir, 'tables'),     exist_ok=True)
        os.makedirs(os.path.join(agg_dir, 'statistics'), exist_ok=True)
        return cls(raw, agg_dir, ms_cfg)

    # ── compute ───────────────────────────────────────────────────────────────

    def compute_and_save(self) -> dict:
        """Compute all statistics, save tables and significance report."""
        self.stats = self._compute_stats()
        self._save_latex_table()
        self._save_csv_table()
        self._save_significance_table()
        self._save_full_stats_json()
        return self.stats

    def _compute_stats(self) -> dict:
        """
        Returns:
            {method_name: {metric: {mean, std, n, ci95, ci_low, ci_high, values}}}
        """
        results = {}
        for method, seed_list in self.all_results.items():
            results[method] = {}
            for metric in METRICS:
                values = [
                    float(r[metric])
                    for r in seed_list
                    if isinstance(r.get(metric), (int, float))
                       and not math.isnan(float(r[metric]))
                ]
                n = len(values)
                if n == 0:
                    results[method][metric] = {
                        'mean': float('nan'), 'std': float('nan'), 'n': 0,
                        'ci95': float('nan'), 'ci_low': float('nan'),
                        'ci_high': float('nan'), 'values': [],
                    }
                    continue
                mean  = float(np.mean(values))
                std   = float(np.std(values, ddof=1)) if n > 1 else 0.0
                ci95  = float(self.ci_z * std / math.sqrt(n))
                results[method][metric] = {
                    'mean':    mean,
                    'std':     std,
                    'n':       n,
                    'ci95':    ci95,
                    'ci_low':  mean - ci95,
                    'ci_high': mean + ci95,
                    'values':  values,
                }
        return results

    # ── significance tests ────────────────────────────────────────────────────

    def compute_significance(self, metric: str = 'test_reward') -> dict:
        """
        Compare each method against HFDRL using paired t-test and Wilcoxon.

        Both tests are PAIRED — same seeds, so each seed's value is matched.
        Returns dict {method: {t_stat, t_pval, w_stat, w_pval, significant, direction}}
        """
        if HFDRL_NAME not in self.all_results:
            return {}

        hfdrl_seeds = self.all_results[HFDRL_NAME]
        hfdrl_map   = {r['seed']: float(r[metric])
                       for r in hfdrl_seeds
                       if isinstance(r.get(metric), (int, float))}

        sig_results = {}
        for method, seed_list in self.all_results.items():
            if method == HFDRL_NAME:
                continue
            method_map = {r['seed']: float(r[metric])
                          for r in seed_list
                          if isinstance(r.get(metric), (int, float))}

            shared_seeds = sorted(set(hfdrl_map) & set(method_map))
            if len(shared_seeds) < 2:
                sig_results[method] = {
                    'n_paired': len(shared_seeds),
                    'note': 'insufficient paired samples',
                }
                continue

            hfdrl_vals  = np.array([hfdrl_map[s]  for s in shared_seeds])
            method_vals = np.array([method_map[s] for s in shared_seeds])

            # Paired t-test
            t_stat, t_pval = scipy_stats.ttest_rel(hfdrl_vals, method_vals)

            # Wilcoxon signed-rank (only if any difference exists)
            diff = hfdrl_vals - method_vals
            if np.all(diff == 0):
                w_stat, w_pval = float('nan'), 1.0
            else:
                try:
                    w_stat, w_pval = scipy_stats.wilcoxon(diff)
                    w_stat, w_pval = float(w_stat), float(w_pval)
                except Exception:
                    w_stat, w_pval = float('nan'), float('nan')

            hb     = METRIC_HIGHER_BETTER.get(metric, True)
            hfdrl_better = (float(np.mean(hfdrl_vals)) > float(np.mean(method_vals))) if hb \
                      else (float(np.mean(hfdrl_vals)) < float(np.mean(method_vals)))

            sig_results[method] = {
                'n_paired':     len(shared_seeds),
                'hfdrl_mean':   float(np.mean(hfdrl_vals)),
                'baseline_mean': float(np.mean(method_vals)),
                't_stat':       float(t_stat),
                't_pval':       float(t_pval),
                'w_stat':       w_stat,
                'w_pval':       w_pval,
                'significant':  bool(t_pval < self.alpha),
                'hfdrl_better': hfdrl_better,
            }

        return sig_results

    # ── LaTeX table ───────────────────────────────────────────────────────────

    def _save_latex_table(self) -> str:
        """Save publication-ready LaTeX table (booktabs style)."""
        if not self.stats:
            return ''

        lines = [
            r'\begin{table}[t]',
            r'\centering',
            r'\caption{Performance comparison across methods (mean $\pm$ std, 95\% CI over '
            f'{self._n_seeds()} seeds).' + r'}',
            r'\label{tab:multiseed_results}',
            r'\setlength{\tabcolsep}{4pt}',
            r'\begin{tabular}{l' + 'r' * len(METRICS) + '}',
            r'\toprule',
        ]

        # Header row
        header_cells = ['Method'] + [METRIC_DISPLAY.get(m, m) for m in METRICS]
        lines.append(' & '.join(header_cells) + r' \\')
        lines.append(r'\midrule')

        # Identify best values per metric (for bold formatting)
        best = {}
        for metric in METRICS:
            hb = METRIC_HIGHER_BETTER.get(metric, True)
            vals = [(m, s[metric]['mean'])
                    for m, s in self.stats.items()
                    if not math.isnan(s[metric]['mean'])]
            if vals:
                best[metric] = max(vals, key=lambda x: x[1] if hb else -x[1])[0]

        # Data rows
        method_order = _sorted_methods(list(self.stats.keys()))
        for method in method_order:
            s    = self.stats[method]
            name = method.replace('_', r'\_')
            cells = [name]
            for metric in METRICS:
                ms = s[metric]
                if math.isnan(ms['mean']):
                    cells.append('--')
                    continue
                mean_str = _fmt(ms['mean'], metric)
                ci_str   = _fmt(ms['ci95'], metric)
                std_str  = _fmt(ms['std'],  metric)
                val = f"{mean_str} $\\pm$ {std_str}"
                if method == best.get(metric):
                    val = r'\textbf{' + val + '}'
                cells.append(val)
            lines.append(' & '.join(cells) + r' \\')

        lines += [
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ]

        path = os.path.join(self.output_dir, 'tables', 'performance_table.tex')
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"-> LaTeX table saved to {path}")
        return path

    # ── CSV table ─────────────────────────────────────────────────────────────

    def _save_csv_table(self) -> str:
        """Save CSV table: Method | Metric | Mean | Std | 95% CI."""
        path = os.path.join(self.output_dir, 'tables', 'performance_table.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Metric', 'Mean', 'Std', 'CI_95', 'CI_Low', 'CI_High', 'N_seeds'])
            for method in _sorted_methods(list(self.stats.keys())):
                for metric in METRICS:
                    ms = self.stats[method][metric]
                    writer.writerow([
                        method,
                        METRIC_DISPLAY_PLAIN.get(metric, metric),
                        _safe(ms['mean']),
                        _safe(ms['std']),
                        _safe(ms['ci95']),
                        _safe(ms['ci_low']),
                        _safe(ms['ci_high']),
                        ms['n'],
                    ])
        print(f"-> CSV table saved to {path}")
        return path

    # ── Significance table ─────────────────────────────────────────────────────

    def _save_significance_table(self) -> str:
        """Save LaTeX + CSV significance tables (HFDRL vs baselines)."""
        sig = self.compute_significance('test_reward')
        if not sig:
            return ''

        # ── LaTeX ─────────────────────────────────────────────────────────────
        lines = [
            r'\begin{table}[t]',
            r'\centering',
            r'\caption{Statistical significance: HFDRL vs baselines on test reward '
            r'(paired $t$-test and Wilcoxon signed-rank, $\alpha=0.05$).}',
            r'\label{tab:significance}',
            r'\begin{tabular}{lrrrrcc}',
            r'\toprule',
            r'Baseline & HFDRL Mean & Baseline Mean & $t$-stat & $p$-value (t) & $p$-value (W) & Sig. \\',
            r'\midrule',
        ]
        for method in _sorted_methods(list(sig.keys())):
            s = sig[method]
            if 'note' in s:
                lines.append(f"{method.replace('_', r'_')} & \multicolumn{{6}}{{c}}{{{s['note']}}} \\\\")
                continue
            star = r'\textbf{*}' if s['significant'] else ''
            method_tex = method.replace('_', r'\_')
            lines.append(
                f"{method_tex} & "
                f"{s['hfdrl_mean']:.2f} & "
                f"{s['baseline_mean']:.2f} & "
                f"{s['t_stat']:.3f} & "
                f"{s['t_pval']:.4f} & "
                f"{s['w_pval']:.4f} & "
                f"{star} \\\\"
            )
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

        tex_path = os.path.join(self.output_dir, 'tables', 'significance_table.tex')
        with open(tex_path, 'w') as f:
            f.write('\n'.join(lines))

        # ── CSV ───────────────────────────────────────────────────────────────
        csv_path = os.path.join(self.output_dir, 'tables', 'significance_table.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Baseline', 'HFDRL_Mean', 'Baseline_Mean',
                             't_stat', 't_pval', 'w_stat', 'w_pval',
                             'Significant', 'HFDRL_Better', 'N_paired'])
            for method, s in sig.items():
                if 'note' in s:
                    writer.writerow([method] + [''] * 9)
                    continue
                writer.writerow([
                    method,
                    _safe(s.get('hfdrl_mean')),
                    _safe(s.get('baseline_mean')),
                    _safe(s.get('t_stat')),
                    _safe(s.get('t_pval')),
                    _safe(s.get('w_stat')),
                    _safe(s.get('w_pval')),
                    s.get('significant', ''),
                    s.get('hfdrl_better', ''),
                    s.get('n_paired', ''),
                ])

        print(f"-> Significance tables saved to {tex_path}, {csv_path}")
        self._save_sig_json(sig)
        return tex_path

    def _save_sig_json(self, sig: dict) -> None:
        path = os.path.join(self.output_dir, 'statistics', 'significance_report.json')
        with open(path, 'w') as f:
            json.dump(sig, f, indent=2)
        print(f"-> Significance report saved to {path}")

    def _save_full_stats_json(self) -> None:
        """Save full statistics (without raw value lists) to JSON."""
        out = {}
        for method, metrics_dict in self.stats.items():
            out[method] = {}
            for metric, ms in metrics_dict.items():
                out[method][metric] = {k: v for k, v in ms.items() if k != 'values'}
        path = os.path.join(self.output_dir, 'statistics', 'full_stats.json')
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"-> Full statistics saved to {path}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _n_seeds(self) -> int:
        for v in self.all_results.values():
            if v:
                return len(v)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sorted_methods(names: list) -> list:
    """Put HFDRL first, then alphabetical."""
    priority = [HFDRL_NAME]
    rest     = [n for n in sorted(names) if n not in priority]
    return [n for n in priority if n in names] + rest


def _fmt(value: float, metric: str) -> str:
    """Format a numeric value for LaTeX display."""
    if math.isnan(value):
        return '--'
    if metric in ('test_reward', 'charging_cost', 'comm_overhead_ms'):
        return f'{value:.2f}'
    if metric == 'final_soc':
        return f'{value:.3f}'
    if metric == 'voltage_violation_rate':
        return f'{value:.4f}'
    if metric == 'convergence_episode':
        return f'{int(value)}' if value >= 0 else '--'
    return f'{value:.3f}'


def _safe(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    if isinstance(v, float):
        return f'{v:.6g}'
    return str(v)
