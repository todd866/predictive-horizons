#!/usr/bin/env python3
"""Generate chemotaxis comparison figure from results JSON."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

RESULTS = Path(__file__).parent.parent / 'results' / 'chemotaxis_results.json'

with open(RESULTS) as f:
    data = json.load(f)

meta = data['metadata']
runs = data['runs']

models = ['connectome_only', 'plus_eph', 'plus_npp', 'trilayer']
labels = ['Connectome\nonly', '+ Ephaptic', '+ Neuropeptide', 'Trilayer']
colors = ['#4477AA', '#66CCEE', '#CCBB44', '#EE6677']

# Extract CI per model
ci_by_model = {}
for m in models:
    ci_by_model[m] = [r['chemotaxis_index'] for r in runs if r['model'] == m]

# Paired differences (same heading/seed across models)
def get_paired(model_a, model_b):
    """Get paired CI values matched by heading and seed."""
    a_dict = {(r['heading'], r['seed']): r['chemotaxis_index']
              for r in runs if r['model'] == model_a}
    b_dict = {(r['heading'], r['seed']): r['chemotaxis_index']
              for r in runs if r['model'] == model_b}
    keys = sorted(set(a_dict) & set(b_dict))
    return np.array([a_dict[k] for k in keys]), np.array([b_dict[k] for k in keys])

fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1.5, 1.5]})

# ── Panel A: CI by model (box + individual points) ────────────────
ax = axes[0]
positions = np.arange(len(models))
for i, (m, label, color) in enumerate(zip(models, labels, colors)):
    cis = ci_by_model[m]
    bp = ax.boxplot([cis], positions=[i], widths=0.5,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor=color, alpha=0.4),
                     medianprops=dict(color='black', linewidth=1.5))
    jitter = 0.1 * np.random.RandomState(42).randn(len(cis))
    ax.scatter(np.full(len(cis), i) + jitter, cis, c=color, s=15,
               alpha=0.6, zorder=3, edgecolors='none')

ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Chemotaxis index', fontsize=11)
ax.set_title('a) Navigation performance', fontsize=11, fontweight='bold', loc='left')
ax.set_ylim(-1.1, 1.1)

# Add mean ± SEM
for i, m in enumerate(models):
    cis = ci_by_model[m]
    mean = np.mean(cis)
    se = np.std(cis) / np.sqrt(len(cis))
    ax.plot(i, mean, 'k_', markersize=12, markeredgewidth=2, zorder=5)

# ── Panel B: Paired differences (trilayer - connectome) ───────────
ax = axes[1]
a_ci, t_ci = get_paired('connectome_only', 'trilayer')
diff = t_ci - a_ci
t_stat, p_val = stats.ttest_rel(t_ci, a_ci)

ax.hist(diff, bins=12, color=colors[3], alpha=0.5, edgecolor=colors[3])
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(np.mean(diff), color='black', linewidth=1.5, label=f'mean = {np.mean(diff):+.3f}')
ax.set_xlabel('$\\Delta$CI (trilayer $-$ connectome)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('b) Paired difference', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=9, frameon=False)
ax.text(0.95, 0.95, f'paired $t$-test\n$p = {p_val:.4f}$',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# ── Panel C: Reversal rate by model ───────────────────────────────
ax = axes[2]
rev_by_model = {}
for m in models:
    rev_by_model[m] = [r['reversal_rate'] for r in runs if r['model'] == m]

means = [np.mean(rev_by_model[m]) for m in models]
sems = [np.std(rev_by_model[m]) / np.sqrt(len(rev_by_model[m])) for m in models]
ax.bar(positions, means, yerr=sems, color=colors, alpha=0.6,
       edgecolor=[c for c in colors], capsize=3)
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Reversals / min', fontsize=10)
ax.set_title('c) Reversal rate', fontsize=11, fontweight='bold', loc='left')

plt.tight_layout()
out = Path(__file__).parent / 'fig_chemotaxis.pdf'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved to {out}')

# Print stats
print(f'\nPaired tests vs connectome_only:')
for m, label in zip(models[1:], labels[1:]):
    a_ci, m_ci = get_paired('connectome_only', m)
    diff = m_ci - a_ci
    t, p = stats.ttest_rel(m_ci, a_ci)
    print(f'  {label.replace(chr(10), " "):20s}: Δ={np.mean(diff):+.4f}, t={t:.3f}, p={p:.4f}')
