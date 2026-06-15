"""
Render consolidated 3-sigma relative-divergence plots matching the SIM-CoT
Figure 20 / 21 layout: a single figure per (paradigm, method) showing the
three sigma values (0.01 / 0.1 / 1.0) side-by-side, each panel containing
correct (green) and incorrect (red) mean curves with +/-1 std bands.

Reads everything from cached HDF5 keys -- no inference re-run.

Source per (tree, sigma):
  divergence_rel  : <tree>/stability_feats/stability.h5
                      key = perturbation_relative_divergence{suffix}
  std (perturb)   : key = perturbation_divergence_std{suffix}
                    rescaled by clean_norm to get relative std (approximation)
  correct_mask    : sigma=0.01: <tree>/latent_states/all_states.h5:correct
                                 (full N case; vanilla and SimCoT both work
                                  IF we have N-aligned divergence at sigma=0.01)
                    sigma=0.1/1.0:
                      preferred: stability.h5: perturbation_correct_mask{suffix}
                                 (added by add_perturbation.py post-edit)
                      fallback for SimCoT: all_states.h5:correct[subsample_index]
                                            (HDF5 in JSON order)
                      fallback for vanilla: all_states.h5:correct -- requires
                                            same length as divergence (full N)
                                            else single-curve

Output: 4 PNGs in repo root figures/ tree
  perturbation_relative_grid_vanilla_codi.png
  perturbation_relative_grid_vanilla_coconut.png
  perturbation_relative_grid_simcot_codi.png
  perturbation_relative_grid_simcot_coconut.png

Each at the corresponding tree's plots/ folder for in-paper inclusion.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# (paradigm, method, tree_dir, json_order_matches)
TREES = [
    ("vanilla", "codi",    "results/codi_gpt2/20260502_191252",    False),
    ("vanilla", "coconut", "results/coconut_gpt2/20260502_191252", False),
    ("simcot",  "codi",    "results/simcot_codi_literal",          True),
    ("simcot",  "coconut", "results/simcot_coconut_literal",       True),
]

# (suffix, label, sigma)
SIGMAS = [
    ("",          0.01),
    ("_sigma0.1", 0.1),
    ("_sigma1.0", 1.0),
]


def _load_arrays(stab: Path, suffix: str):
    """Return (rel_div, mask_or_None, n_div) for the given suffix."""
    with h5py.File(str(stab), "r") as f:
        rel_key = f"perturbation_relative_divergence{suffix}"
        if rel_key not in f:
            return None, None, 0
        rel = f[rel_key][:]
        # Try persistent mask first (added by add_perturbation.py post-edit)
        mask_key = f"perturbation_correct_mask{suffix}"
        mask = f[mask_key][:].astype(bool) if mask_key in f else None
        # Try subsample index for fallback path
        idx_key = f"perturbation_subsample_index{suffix}"
        sub_idx = f[idx_key][:] if idx_key in f else None
    return rel, mask, sub_idx


def _correct_mask_for_panel(
    states_h5: Path,
    stab_h5: Path,
    suffix: str,
    rel_shape: Tuple[int, int],
    json_match: bool,
) -> Optional[np.ndarray]:
    """Find the right correctness mask for this (tree, sigma) panel."""
    n_div = rel_shape[0]

    # 1) Persistent mask written by add_perturbation.py
    with h5py.File(str(stab_h5), "r") as f:
        persistent_key = f"perturbation_correct_mask{suffix}"
        if persistent_key in f:
            return f[persistent_key][:].astype(bool)

    # 2) Full-N case: divergence has same length as states.correct
    with h5py.File(str(states_h5), "r") as f:
        if "correct" not in f:
            return None
        all_correct = f["correct"][:].astype(bool)

    if n_div == len(all_correct):
        return all_correct

    # 3) JSON-order tree (SimCoT) with stored subsample index
    with h5py.File(str(stab_h5), "r") as f:
        idx_key = f"perturbation_subsample_index{suffix}"
        if idx_key in f and json_match:
            sub_idx = f[idx_key][:]
            if sub_idx.max() < len(all_correct):
                return all_correct[sub_idx]

    # 4) Give up
    return None


def _plot_one_panel(ax, rel: np.ndarray, mask: Optional[np.ndarray],
                    sigma: float, n_samples: int):
    """Single sigma panel: correct (green) / incorrect (red) mean +- std."""
    n_steps = rel.shape[1]
    steps = np.arange(n_steps)

    if mask is not None and mask.any() and (~mask).any():
        for sub_mask, color, marker, ls, label in [
            (mask,  "#2ecc71", "o", "-",  f"Correct (n={int(mask.sum()):,})"),
            (~mask, "#e74c3c", "x", "--", f"Incorrect (n={int((~mask).sum()):,})"),
        ]:
            sub = rel[sub_mask]
            m = np.nanmean(sub, axis=0)
            s = np.nanstd(sub, axis=0)
            lo = np.maximum(m - s, 0.0)
            ax.plot(steps, m, marker=marker, linestyle=ls, color=color,
                    linewidth=2.0, markersize=6, label=label, zorder=4)
            ax.fill_between(steps, lo, m + s, color=color, alpha=0.20)
    else:
        m = np.nanmean(rel, axis=0)
        s = np.nanstd(rel, axis=0)
        lo = np.maximum(m - s, 0.0)
        ax.plot(steps, m, "o-", color="#444", linewidth=2.0, markersize=6,
                label="Mean", zorder=4)
        ax.fill_between(steps, lo, m + s, color="#888", alpha=0.20)

    ax.set_title(f"sigma = {sigma}", fontsize=11)
    ax.set_xlabel(f"Latent step t  (t in 0..{n_steps - 1})", fontsize=10)
    ax.set_ylabel(
        r"$\|z_t^{perturbed} - z_t^{clean}\|_2 / \|z_t^{clean}\|_2$",
        fontsize=10,
    )
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.98, 0.02, f"N = {n_samples:,} samples (aggregated)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        color="#555", bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="white", edgecolor="#cccccc",
                                 alpha=0.85),
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)


def render_grid(paradigm: str, method: str, tree_str: str, json_match: bool):
    tree = PROJECT_ROOT / tree_str
    stab = tree / "stability_feats" / "stability.h5"
    states = tree / "latent_states" / "all_states.h5"
    if not stab.is_file():
        print(f"  SKIP {paradigm} {method}: missing stability.h5")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    panels_rendered = 0
    for ax, (suffix, sigma) in zip(axes, SIGMAS):
        rel, _, _ = _load_arrays(stab, suffix)
        if rel is None:
            ax.text(0.5, 0.5, f"(no data for sigma={sigma})",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="#888")
            ax.set_title(f"sigma = {sigma}", fontsize=11)
            ax.axis("off")
            continue
        mask = _correct_mask_for_panel(states, stab, suffix, rel.shape, json_match)
        _plot_one_panel(ax, rel, mask, sigma, rel.shape[0])
        panels_rendered += 1

    if panels_rendered == 0:
        plt.close(fig)
        return None

    method_label = method.upper()
    paradigm_label = "Vanilla" if paradigm == "vanilla" else "SIM-CoT"
    fig.suptitle(
        f"Perturbation Relative Divergence  --  {paradigm_label} {method_label}\n"
        f"(n_perturbations = 3;  growing = sensitive, flat = robust)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    out_path = tree / "plots" / f"perturbation_relative_grid_{paradigm}_{method}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")
    return out_path


def main():
    written = 0
    for paradigm, method, tree_str, json_match in TREES:
        out = render_grid(paradigm, method, tree_str, json_match)
        if out:
            written += 1
    print(f"\nDone. {written} consolidated 3-sigma figures written.")


if __name__ == "__main__":
    main()
