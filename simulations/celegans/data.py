"""
C. elegans data loading — connectome, positions, neuropeptide/monoamine layers.

Extracts the data-loading logic that was duplicated across every simulation
script in openworm/ into a single reusable function.

Usage::

    from celegans.data import load_worm_data
    wd = load_worm_data("openworm")
    print(wd.N, wd.W_chem.shape)
"""

import csv
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# ── Neuron-class prefixes ─────────────────────────────────────────────

SENSORY_PFX = (
    "ADF", "ADL", "ASE", "ASG", "ASH", "ASI", "ASJ", "ASK",
    "AWA", "AWB", "AWC", "AFD", "BAG", "IL1", "IL2", "OLL",
    "OLQ", "PHA", "PHB", "PLM", "ALM", "AVM", "PVD", "FLP",
)

MOTOR_PFX = (
    "VA", "VB", "VD", "DA", "DB", "DD", "AS",
    "RMD", "RME", "SMD", "SMB",
)

# ── Return type ───────────────────────────────────────────────────────

WormData = namedtuple("WormData", [
    "neurons",    # sorted list of neuron names
    "N",          # number of neurons
    "neuron_idx", # dict  name -> int index
    "W_chem",     # (N,N) chemical synapse adjacency (raw weights)
    "W_gap",      # (N,N) gap junction adjacency (raw, bidirectional)
    "W_npp",      # (N,N) neuropeptide adjacency (raw)
    "W_mono",     # (N,N) monoamine adjacency (raw)
    "pos_3d",     # (N,3) 3-D neuron positions (NeuroPAL atlas)
    "pos_1d",     # (N,)  body-axis position in [0,1]
    "dist_3d",    # (N,N) pairwise 3-D distance matrix
    "sensory",    # list of indices — sensory neurons
    "motor",      # list of indices — motor neurons
    "inter",      # list of indices — interneurons
    "VA",         # specific motor-class indices
    "VB",
    "DA",
    "DB",
    "VD",
    "DD",
])


def load_worm_data(data_dir):
    """Load all C. elegans data from *data_dir* and return a `WormData`.

    Parameters
    ----------
    data_dir : str or Path
        Path to the directory containing ``herm_full_edgelist.csv`` and a
        ``data/`` subdirectory with position and neuropeptide files.

    Returns
    -------
    WormData
        Named tuple with connectome matrices, positions, distances, and
        neuron-class indices.
    """
    data_dir = Path(data_dir)

    # ── 1. Connectome edges ───────────────────────────────────────────
    neurons_set = set()
    chemical_edges = []
    gap_edges = []

    with open(data_dir / "herm_full_edgelist.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row["Source"].strip()
            tgt = row["Target"].strip()
            w = int(row["Weight"].strip())
            typ = row["Type"].strip()
            neurons_set.add(src)
            neurons_set.add(tgt)
            if typ == "chemical":
                chemical_edges.append((src, tgt, w))
            elif typ == "electrical":
                gap_edges.append((src, tgt, w))

    # ── 2. Neuropeptide / monoamine connectomes ───────────────────────
    npp_df = pd.read_csv(data_dir / "data" / "neuropeptide_connectome_LR.csv",
                         index_col=0)
    npp_neurons = list(npp_df.columns)

    mono_path = data_dir / "data" / "monoamine_connectome.csv"
    if mono_path.exists():
        mono_df = pd.read_csv(mono_path, index_col=0)
    else:
        mono_df = pd.DataFrame()

    # ── 3. 3-D neuron positions (NeuroPAL atlas) ─────────────────────
    with open(data_dir / "data" / "neuron_positions_3d.txt") as f:
        header = f.readline().strip().lstrip("#")
        pos3d_names = header.split()
        pos3d_coords = []
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                pos3d_coords.append([float(x) for x in parts])

    pos3d_coords = np.array(pos3d_coords)
    pos3d_map = {
        name: pos3d_coords[i]
        for i, name in enumerate(pos3d_names)
        if i < len(pos3d_coords)
    }

    # ── 4. Unified neuron list ────────────────────────────────────────
    neurons = sorted(neurons_set)
    N = len(neurons)
    neuron_idx = {n: i for i, n in enumerate(neurons)}

    # ── 5. Adjacency matrices ─────────────────────────────────────────
    W_chem = np.zeros((N, N))
    W_gap = np.zeros((N, N))

    for src, tgt, w in chemical_edges:
        W_chem[neuron_idx[src], neuron_idx[tgt]] += w

    for src, tgt, w in gap_edges:
        i, j = neuron_idx[src], neuron_idx[tgt]
        W_gap[i, j] += w
        W_gap[j, i] += w

    # ── 6. Map neuropeptide/monoamine to unified indices ──────────────
    W_npp = np.zeros((N, N))
    W_mono = np.zeros((N, N))

    for src_name in npp_neurons:
        if src_name not in neuron_idx:
            continue
        si = neuron_idx[src_name]
        for tgt_name in npp_neurons:
            if tgt_name not in neuron_idx:
                continue
            ti = neuron_idx[tgt_name]
            if src_name in npp_df.index and tgt_name in npp_df.columns:
                val = npp_df.loc[src_name, tgt_name]
                if val > 0:
                    W_npp[si, ti] = val
            if (not mono_df.empty
                    and src_name in mono_df.index
                    and tgt_name in mono_df.columns):
                val = mono_df.loc[src_name, tgt_name]
                if val > 0:
                    W_mono[si, ti] = val

    # ── 7. 3-D positions / body axis / distance matrix ────────────────
    pos_3d = np.zeros((N, 3))
    rng_pos = np.random.RandomState(0)
    for i, n in enumerate(neurons):
        if n in pos3d_map:
            pos_3d[i] = pos3d_map[n]
        else:
            pos_3d[i] = rng_pos.randn(3) * 0.3

    # Body axis = y-coordinate (anterior-posterior in the atlas)
    y_coords = pos_3d[:, 1]
    pos_1d = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min() + 1e-10)

    dist_3d = squareform(pdist(pos_3d))

    # ── 8. Neuron classification ──────────────────────────────────────
    sensory = [i for i, n in enumerate(neurons) if n.startswith(SENSORY_PFX)]
    motor = [i for i, n in enumerate(neurons) if n.startswith(MOTOR_PFX)]
    inter = [i for i in range(N) if i not in sensory and i not in motor]
    va = [i for i, n in enumerate(neurons) if n.startswith("VA")]
    vb = [i for i, n in enumerate(neurons) if n.startswith("VB")]
    da = [i for i, n in enumerate(neurons) if n.startswith("DA")]
    db = [i for i, n in enumerate(neurons) if n.startswith("DB")]
    vd = [i for i, n in enumerate(neurons) if n.startswith("VD")]
    dd = [i for i, n in enumerate(neurons) if n.startswith("DD")]

    return WormData(
        neurons=neurons,
        N=N,
        neuron_idx=neuron_idx,
        W_chem=W_chem,
        W_gap=W_gap,
        W_npp=W_npp,
        W_mono=W_mono,
        pos_3d=pos_3d,
        pos_1d=pos_1d,
        dist_3d=dist_3d,
        sensory=sensory,
        motor=motor,
        inter=inter,
        VA=va,
        VB=vb,
        DA=da,
        DB=db,
        VD=vd,
        DD=dd,
    )
