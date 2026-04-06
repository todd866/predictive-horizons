# Predictive Horizons — Next Session Plan

## The Core Argument (evolved during this session)

**The resource is right there.** Every neuron in C. elegans sits in a continuous bioelectric field — ephaptic coupling, neuropeptide broadcast, proprioceptive feedback. These channels are measured, documented, and systematically ignored by computational neuroscience because the toolchain (NEURON, Brian2, NeuroML, OpenWorm) can't represent them. The connectome is what you see when you look at biology with a graph-shaped instrument. It's the measurement, not the system.

**Of course evolution harnesses it.** The field provides continuous, high-bandwidth, spatially graded coupling. Not harnessing it would be like building a radio and ignoring the antenna.

**Connectome-only simulations can't coordinate.** Our simulation on the real C. elegans connectome (302 neurons, 7000+ synapses) shows: with synaptic connectivity alone, motor neurons are essentially uncorrelated (VA-VB = 0.04). Add ephaptic field coupling and motor coordination jumps to 0.65. The connectome carries information; the field coordinates action.

**This requires criticality.** The effect only appears near the synchronisation transition. Below: nothing coordinates. Above: everything locks. At criticality: avalanche-like propagation converts weak distributed correlations into coordinated motor patterns.

**Prediction requires coordination.** A worm that "knows" where the food is but can't coordinate its muscles to crawl there hasn't predicted anything in any biologically meaningful sense. The MI metric is similar across models (~0.5 nats). The motor coordination metric is where the difference lives. The connectome has environmental information trapped in its neurons with no way to get it out as coherent movement.

## Key Results (what works, what doesn't)

### Keep:
- **Motor coordination at criticality** — VA-VB: 0.04 → 0.65 with ephaptic coupling. THE headline result.
- **Sync-measurement asymmetry** — three timescale framework (τ_sync, τ_env, τ_meas)
- **Observational gap** — system MI >> observer MI (14.4 vs 5.2 nats)
- **Code collapse** — coding destroys 85% of predictive content
- **Prediction Without Coding** section — direct critique of predictive coding
- **"Intelligence lives upstream of code"** closing

### Drop or rethink:
- **MI capacity sweep** — both models scale linearly, no divergence. MI isn't the differentiator.
- **Field PDE simulation** — beautiful kymograph but field-to-neuron coupling too weak quantitatively
- **Original "oscillatory beats discrete 16x"** — unfair comparison, overclaimed

## Plan for Next Session

### 1. Strengthen the C. elegans simulation

**The experiment that works:** Three models at criticality, same connectome:
- A: Connectome only (r = 0.10)
- B: + Ephaptic coupling (r = 0.21)
- C: + Ephaptic + neuropeptide + proprioception (r = 0.26)

**Improvements needed:**
- Run at 3 coupling regimes (below/at/above critical) → shows criticality is required
- Multiple seeds for error bars (currently n=1)
- Use same random seeds across models for fair comparison
- Add travelling wave analysis (not just pairwise VA-VB correlation)
- Proper citations: White et al. 1986, Cook et al. 2019, Varshney et al. 2011

### 2. Restructure paper around the new narrative

**New structure:**

1. **Introduction** — The resource is right there. Why connectome simulations fail (OpenWorm). Measurement selects for what the instrument can see. The connectome is the measurement, not the system.

2. **Theory: Sync-Measurement Asymmetry** — Keep existing Sections 2-3 (three timescales, passive/active/selective sync). Trim to essentials. This provides the formal reason WHY field coupling matters: it's the substrate that enables τ_sync < τ_env.

3. **The C. elegans Test** — Three models, one connectome. MI is the same; motor coordination diverges. The connectome carries information but can't coordinate the body. The field enables the translation from prediction to action.

4. **Criticality** — The coordination advantage only appears near the sync transition. Phase diagram: coupling strength × model type. The sweet spot is at criticality.

5. **The Observational Gap** — Why low-D measurement misses field dynamics (keep existing simulation + figure). The measurement apparatus (electrode arrays, connectome tracing) is low-D — captures spikes and synapses, throws away field dynamics. Same pattern as glial cells being "just structural support" for decades.

6. **Prediction Without Coding** — Coding is dimensional collapse. It's the downstream product, not the mechanism. Keep and trim.

7. **Discussion** — Self-simulation (proprioceptive loop closes prediction-action cycle). Connection to Igamberdiev's anticipatory dynamics. The connectome is necessary (synaptic specificity) but not sufficient (needs the field for coordination). Implications for computational neuroscience methodology.

### 3. Final figure set

1. Alignment snapshots — keep (sync manifold formation)
2. Three-level hierarchy — keep (TikZ schematic)
3. **C. elegans motor coordination** — THE headline figure. Three models, coordination metrics, criticality dependence.
4. Observational gap — keep (hidden correlations)
5. Code collapse — keep (coding destroys prediction)

### 4. Text changes
- New abstract reflecting coordination thesis
- New intro framing (resource is right there, measurement-shaped blindness)
- OpenWorm citations and context
- Update highlights for BioSystems submission
- Fix falsifiability self-cite title
- Revise scope boundaries: sketch self-simulation

## Files

- Main tex: `predictive_horizons.tex`
- **Key simulation:** `simulations/openworm/celegans_highD.py` (the keeper — critical regime three-model comparison)
- Also ran: `celegans_field.py` (field PDE — kymograph is nice but quantitative result weak), `celegans_capacity.py` (MI capacity sweep — null result)
- Connectome data: `simulations/openworm/herm_full_edgelist.csv`
- Neuron positions: `simulations/openworm/neuron_positions.csv`
- Public repo: https://github.com/todd866/predictive-horizons
- Parent submodule: `biosystems/81_predictive_horizons` in highdimensional

## Venue

**BioSystems** — stay the course. Free via USyd Elsevier agreement, Igamberdiev alignment, established track record. If the result turns out exceptionally clean, could consider PLOS Comp Bio or Neural Computation as a follow-up focused purely on the simulation result. But the theory + simulation package fits BioSystems best.
