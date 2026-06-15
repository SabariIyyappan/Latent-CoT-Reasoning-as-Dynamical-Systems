"""
Re-render every perturbation_divergence_*.png across all four trees from
the cached HDF5 arrays, applying the updated y-axis label
(``\\|z_t^{perturbed} - z_t^{clean}\\|_2`` etc., expanded form) without
re-running any inference.

Source data per (tree, sigma):
  divergence              : <tree>/stability_feats/stability.h5
                              key = perturbation_divergence{suffix}
  divergence_std          : same h5, key = perturbation_divergence_std{suffix}
  relative_divergence     : same h5, key = perturbation_relative_divergence{suffix}
  subsample_index         : same h5, key = perturbation_subsample_index{suffix}
                              (only present for the stratified-subsample runs)

Correct-mask source priority (per (tree, sigma) cell):
  1. If divergence shape[0] == N_full (=8792): use <tree>/latent_states/
     all_states.h5:correct directly. Applies to vanilla sigma=0.01 only.
  2. Else if subsample_index is present AND the tree's HDF5 question order
     matches the JSON order used at run time: use correct[subsample_index].
     Applies to all SIM-CoT cases (HDF5 in JSON order, verified).
  3. Else: render without correctness split. Applies to vanilla
     sigma=0.1/1.0 because vanilla all_states.h5 is in DataPrep order, not
     JSON order, and the inline-recomputed mask wasn't saved.

Filename suffix matches what's already on disk so files are overwritten
in place: '' (sigma=0.01), '_sigma0.1', '_sigma1.0'.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.plotting import Plotting


# (paradigm, method, tree_dir, json_order_matches_hdf5_order)
TREES = [
    ("vanilla", "coconut", "results/coconut_gpt2/20260502_191252", False),
    ("vanilla", "codi",    "results/codi_gpt2/20260502_191252",    False),
    ("simcot",  "coconut", "results/simcot_coconut_literal",       True),
    ("simcot",  "codi",    "results/simcot_codi_literal",          True),
]

# (filename_suffix, hdf5_key_suffix, sigma)
SIGMAS = [
    ("",            "",            0.01),
    ("_sigma0.1",   "_sigma0.1",   0.1),
    ("_sigma1.0",   "_sigma1.0",   1.0),
]


def _load_correct_mask(
    states_h5: Path,
    n_div: int,
    subsample_idx: Optional[np.ndarray],
    json_match: bool,
) -> Optional[np.ndarray]:
    """Return the right correctness mask for this (tree, sigma) cell, or
    None if it can't be reconstructed from cached data."""
    with h5py.File(str(states_h5), "r") as f:
        if "correct" not in f:
            return None
        all_correct = f["correct"][:].astype(bool)
    n_full = len(all_correct)

    if n_div == n_full:
        # Full-N case (vanilla sigma=0.01 from runner.py)
        return all_correct

    if subsample_idx is not None and json_match and subsample_idx.max() < n_full:
        # SIM-CoT subsampled case (HDF5 in JSON order)
        return all_correct[subsample_idx]

    # Vanilla subsampled case — order mismatch, no saved mask
    return None


def relabel_one(
    paradigm: str,
    method: str,
    tree_dir: Path,
    json_match: bool,
    filename_suffix: str,
    key_suffix: str,
    sigma: float,
) -> Optional[str]:
    stab = tree_dir / "stability_feats" / "stability.h5"
    states = tree_dir / "latent_states" / "all_states.h5"
    if not stab.is_file() or not states.is_file():
        return None

    with h5py.File(str(stab), "r") as f:
        div_key = f"perturbation_divergence{key_suffix}"
        std_key = f"perturbation_divergence_std{key_suffix}"
        rel_key = f"perturbation_relative_divergence{key_suffix}"
        idx_key = f"perturbation_subsample_index{key_suffix}"
        if div_key not in f:
            return None
        divergence = f[div_key][:]
        div_std = f[std_key][:] if std_key in f else None
        relative = f[rel_key][:] if rel_key in f else None
        subsample_idx = f[idx_key][:] if idx_key in f else None

    correct_mask = _load_correct_mask(
        states, divergence.shape[0], subsample_idx, json_match
    )

    plotter = Plotting(str(tree_dir))
    out_path = plotter.make_perturbation_divergence_plot(
        divergence=divergence,
        divergence_std=div_std,
        correct_mask=correct_mask,
        relative_divergence=relative,
        noise_std=sigma,
        n_perturbations=3,
        filename_suffix=filename_suffix,
    )
    return out_path


def main() -> None:
    written, skipped = 0, 0
    for paradigm, method, tree_str, json_match in TREES:
        tree_dir = PROJECT_ROOT / tree_str
        for filename_suffix, key_suffix, sigma in SIGMAS:
            tag = f"{paradigm:>7s} {method:>7s} σ={sigma}"
            out = relabel_one(
                paradigm, method, tree_dir, json_match,
                filename_suffix, key_suffix, sigma,
            )
            if out is None:
                print(f"  SKIP  {tag}  (no perturbation_divergence{key_suffix} key)")
                skipped += 1
            else:
                # Note whether mask was reconstructable
                marker = "WITH split" if (
                    (sigma == 0.01 and paradigm == "vanilla")  # full N case
                    or paradigm == "simcot"                    # always reconstructable
                ) else "no split (vanilla σ>0.01 mask not cached)"
                print(f"  OK    {tag}  → {out}   [{marker}]")
                written += 1
    print(f"\nDone. {written} written, {skipped} skipped.")


if __name__ == "__main__":
    main()
