# Predictive Horizons of High-Dimensional Biological Systems

**Repository:** [todd866/predictive-horizons](https://github.com/todd866/predictive-horizons)
**Paper status:** In preparation for *BioSystems*

## Companion Papers

This is paper 5 in the BioSystems series on high-dimensional biological dynamics:

1. **Limits of Falsifiability** — [DOI: 10.1016/j.biosystems.2025.105608](https://doi.org/10.1016/j.biosystems.2025.105608)
2. **Timing Inaccessibility** — [DOI: 10.1016/j.biosystems.2025.105632](https://doi.org/10.1016/j.biosystems.2025.105632)
3. **Intelligence as High-Dimensional Coherence** — [DOI: 10.1016/j.biosystems.2026.105704](https://doi.org/10.1016/j.biosystems.2026.105704) | [repo](https://github.com/todd866/intelligence-biosystems)
4. **Coherence Time** — [DOI: 10.1016/j.biosystems.2026.105755](https://doi.org/10.1016/j.biosystems.2026.105755)
5. **Predictive Horizons** — this paper

## One-line thesis

Intelligence and prediction live upstream of code: biological prediction arises from continuous oscillatory synchronization with environmental structure, not from computation over internal models, and coding destroys most of the predictive content.

## Core results

### 1. Sync-Measurement Asymmetry

Three timescales define the predictive regime:

- **τ_sync** — entrainment time (how quickly internal dynamics align with environment through coupling)
- **τ_env** — environmental autocorrelation (how long structure persists)
- **τ_meas** — measurement commitment time (how long to collapse to discrete output)

**Predictive regime:** τ_sync < τ_env (organism keeps up with environment)
**Anticipatory appearance:** τ_sync < τ_meas (organism has incorporated structure before it can be "measured")

### 2. The Observational Gap

A 625-oscillator network coupled to a spatiotemporal environment carries ~14.4 nats of future mutual information in its full internal state. A 5-dimensional observer can verify only ~5.2 nats — 64% is hidden correlation structure invisible to low-dimensional measurement.

### 3. Coding Destroys Prediction

An explicit commit channel (5 dimensions, 4 bits, discrete updates) retains only ~2.5 nats of the 14.4 available — coding destroys ~85% of predictive content. The prediction was in the oscillatory dynamics; the code is its impoverished shadow.

### 4. Prediction Without Coding

Predictive coding mistakes the low-dimensional products of dimensional collapse for the generative mechanism. Synchronization provides a complete account of prediction for organisms without the architecture to implement hierarchical Bayesian inference.

## Repository structure

```
predictive-horizons/
├── predictive_horizons.tex       # Manuscript
├── predictive_horizons.pdf       # Compiled output
├── highlights.txt                # BioSystems highlights
├── simulations/
│   ├── generate_figures.py       # All simulation code
│   └── figures/
│       ├── fig1_alignment.pdf    # Sync manifold formation
│       ├── fig2_observational_gap.pdf  # Hidden correlations
│       └── fig3_code_collapse.pdf      # Coding destroys prediction
├── README.md
├── CITATION.cff
└── LICENSE                       # MIT
```

## Running simulations

```bash
cd simulations
python3 generate_figures.py
```

**Requirements:** numpy, scipy, matplotlib

The script generates three evidentiary figures:
- **Figure 1:** Alignment field snapshots showing sync manifold formation
- **Figure 4:** The observational gap — system MI vs observer MI at different resolutions
- **Figure 5:** Code collapse — continuous state vs explicit commit channel

## Citation

```bibtex
@article{todd2026predictive,
  title={Predictive Horizons of High-Dimensional Biological Systems},
  author={Todd, Ian},
  journal={BioSystems},
  year={2026},
  note={In preparation}
}
```

## License

MIT License. See [LICENSE](LICENSE).
