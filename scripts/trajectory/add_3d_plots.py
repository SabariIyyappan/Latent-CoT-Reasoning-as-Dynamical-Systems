"""
Add 3-D DMD and PHATE trajectory plots (combined + per-concept) to an
existing analysis tree, without re-running inference.

Why this exists
---------------
Jerome's follow-up review asked for the 3-D versions of DMD and PHATE
projections for both the initial GSM8K experiments (CODI / COCONUT
GPT-2) and the SimCoT experiments (SIM-CoT-CODI / SIM-CoT-COCONUT
GPT-2). The original ``runner.py`` and ``analyze_from_cache.py`` only
emitted 2-D versions; this script computes 3-component DMD and 3-component
PHATE on the cached ``latent_states/all_states.h5`` and renders the new
plots into ``<run_dir>/plots/`` (combined) and ``<run_dir>/plots/per_concept/``.

This is purely additive: existing 2-D plots, dynamic features, stability,
and the DMD eigenvalue / unit-circle plot (``dmd_spectrum.png``) are
left untouched.

Usage
-----
    python scripts/add_3d_plots.py <run_dir> [<run_dir> ...]

Each ``<run_dir>`` must already contain ``latent_states/all_states.h5``
with datasets ``latent_thoughts``, ``correct``, and ``concept``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.dim_reduct import DimReduct
from analysis.plotting import Plotting
from analysis.data_prep import GSM8K_CONCEPT_NAMES


def _load_states(run_dir: Path) -> Dict[str, np.ndarray]:
    h5_path = run_dir / "latent_states" / "all_states.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Expected cached HDF5 at {h5_path}. Re-run extract / inference "
            f"to produce it before adding 3-D plots."
        )
    out: Dict[str, np.ndarray] = {}
    with h5py.File(str(h5_path), "r") as f:
        for key in ("latent_thoughts", "correct", "concept"):
            if key in f:
                out[key] = f[key][:]
    if "latent_thoughts" not in out:
        raise RuntimeError(f"{h5_path} missing required dataset 'latent_thoughts'")
    if "correct" not in out:
        raise RuntimeError(f"{h5_path} missing required dataset 'correct'")
    if "concept" not in out:
        out["concept"] = np.full(len(out["correct"]), 6, dtype=np.int32)
    return out


def _save_3d_reduced(run_dir: Path, name: str, embedding: np.ndarray) -> None:
    out_dir = run_dir / "reduced_states"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_reduced.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("embedding", data=embedding)
        f.attrs["method"] = name
        f.attrs["n_samples"] = embedding.shape[0]
        f.attrs["n_steps"] = embedding.shape[1]
        f.attrs["n_components"] = embedding.shape[2]
    print(f"  saved {path}  shape={embedding.shape}")


def add_3d_plots(run_dir: Path) -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    print(f"\n=== {run_dir} ===")
    t0 = time.time()
    data = _load_states(run_dir)
    states = data["latent_thoughts"]
    correct = data["correct"].astype(bool)
    concept = data["concept"].astype(np.int32)
    print(
        f"  loaded states {states.shape}  correct={int(correct.sum())}/"
        f"{len(correct)}  concepts={np.unique(concept).tolist()}"
    )

    plotting = Plotting(str(run_dir))

    # DimReduct expects a torch tensor (it calls .numpy() inside)
    states_t = torch.from_numpy(states.astype(np.float32))

    # ── DMD with n_components=3 ──────────────────────────────────────────
    print("  computing DMD (n_components=3)...")
    reducer = DimReduct(states_t)
    dmd3 = reducer.perform_dmd(n_components=3)
    _save_3d_reduced(run_dir, "dmd3d", dmd3)

    print("  rendering DMD 3-D combined...")
    plotting.make_3d_trajectory_plot(
        embeddings=dmd3,
        method_name="dmd",
        correct_mask=correct,
    )
    print("  rendering DMD 3-D per-concept...")
    plotting.make_per_concept_3d_trajectory_plots(
        embeddings=dmd3,
        method_name="dmd",
        concept_labels=concept,
        concept_names=GSM8K_CONCEPT_NAMES,
        correct_mask=correct,
    )

    # ── PHATE with n_components=3 ────────────────────────────────────────
    # Use a fresh reducer so DMD eigenvalue side effects don't carry over.
    print("  computing PHATE (n_components=3)...")
    reducer = DimReduct(states_t)
    phate3 = reducer.perform_phate(n_components=3)
    # perform_phate stores under "phate"; rename in-flight to keep the 2-D
    # output (already on disk) untouched.
    _save_3d_reduced(run_dir, "phate3d", phate3)

    print("  rendering PHATE 3-D combined...")
    plotting.make_3d_trajectory_plot(
        embeddings=phate3,
        method_name="phate",
        correct_mask=correct,
    )
    print("  rendering PHATE 3-D per-concept...")
    plotting.make_per_concept_3d_trajectory_plots(
        embeddings=phate3,
        method_name="phate",
        concept_labels=concept,
        concept_names=GSM8K_CONCEPT_NAMES,
        correct_mask=correct,
    )

    print(f"  done {run_dir.name}  elapsed={(time.time()-t0)/60:.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more analysis run directories (each must contain "
        "latent_states/all_states.h5).",
    )
    args = parser.parse_args()
    for d in args.run_dirs:
        add_3d_plots(d.resolve())


if __name__ == "__main__":
    main()
