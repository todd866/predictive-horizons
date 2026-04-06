# Predictive Horizons of High-Dimensional Biological Systems

**The connectome is the measurement, not the system.**

**Repository:** [todd866/predictive-horizons](https://github.com/todd866/predictive-horizons)
**Paper status:** In preparation for *BioSystems*

## Companion Papers

Paper 5 in the BioSystems series on high-dimensional biological dynamics:

1. **Limits of Falsifiability** — [DOI: 10.1016/j.biosystems.2025.105608](https://doi.org/10.1016/j.biosystems.2025.105608)
2. **Timing Inaccessibility** — [DOI: 10.1016/j.biosystems.2025.105632](https://doi.org/10.1016/j.biosystems.2025.105632)
3. **Intelligence as High-Dimensional Coherence** — [DOI: 10.1016/j.biosystems.2026.105704](https://doi.org/10.1016/j.biosystems.2026.105704)
4. **Coherence Time** — [DOI: 10.1016/j.biosystems.2026.105755](https://doi.org/10.1016/j.biosystems.2026.105755)
5. **Predictive Horizons** — this paper

## Overview

Connectome-only simulations of *C. elegans* carry environmental information but cannot coordinate motor output. We show that field-level coupling — ephaptic, neuropeptide broadcast, proprioceptive feedback — rescues motor coordination on the real 448-neuron connectome, and that this rescue requires criticality.

**Key equation:** The predictive regime is τ\_sync < τ\_env (synchronization outruns environmental change), with anticipatory appearance when τ\_sync < τ\_meas (sync outruns measurement commitment).

## Key Results

- **Motor coordination requires field coupling.** VA-VB correlation: ~0.05 (connectome only) → ~0.6 (full field model). MI is similar across models (~0.5 nats). The connectome carries information; the field coordinates the body.
- **Criticality is required.** The coordination advantage appears only near the synchronization phase transition.
- **The observational gap.** A 625-oscillator network carries ~14 nats of predictive content; a 5-dimensional observer measures ~5 nats. 64% is hidden correlation structure.
- **Coding destroys prediction.** Discrete coding retains only ~15% of the predictive content in continuous oscillatory dynamics.

## Running Simulations

```bash
# Theory figures (alignment, observational gap, code collapse)
cd simulations
python3 generate_figures.py

# C. elegans coordination result (headline figure)
cd simulations/openworm
python3 celegans_coordination.py
```

**Requirements:** numpy, scipy, matplotlib

## Repository structure

```
predictive-horizons/
├── predictive_horizons.tex          # Manuscript
├── predictive_horizons.pdf          # Compiled output
├── highlights.txt                   # BioSystems highlights
├── simulations/
│   ├── generate_figures.py          # Theory simulations (625-oscillator field)
│   ├── openworm/
│   │   ├── celegans_coordination.py # Headline result: 3 models × 3 regimes × 5 seeds
│   │   ├── herm_full_edgelist.csv   # OpenWorm connectome data
│   │   └── neuron_positions.csv     # Neuron positions along body axis
│   └── figures/
│       ├── fig1_alignment.pdf       # Sync manifold formation
│       ├── fig2_observational_gap.pdf
│       ├── fig3_code_collapse.pdf
│       └── fig_coordination.pdf     # C. elegans motor coordination
├── README.md
├── CITATION.cff
└── LICENSE                          # MIT
```

## Citation

```bibtex
@article{todd2026predictive,
  title={Predictive Horizons of High-Dimensional Biological Systems:
         The Connectome Is the Measurement, Not the System},
  author={Todd, Ian},
  journal={BioSystems},
  year={2026},
  note={In preparation}
}
```

## License

MIT License. See [LICENSE](LICENSE).
