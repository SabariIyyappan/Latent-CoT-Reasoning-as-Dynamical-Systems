"""
Side-by-side comparison of 3-D DMD and PHATE trajectories between the
initial GSM8K experiments (CODI / COCONUT GPT-2) and the SimCoT
experiments (SIM-CoT-CODI / SIM-CoT-COCONUT GPT-2).

Why this exists
---------------
Jerome asked: "is the separation present in initial exps also seen in
SIMCOT version for both methods?". This script reads the cached
``reduced_states/{dmd3d,phate3d}_reduced.h5`` for the four canonical
trees and writes a single 4 x 2 figure pair (one for each reduction)
with the initial result on the left and the SimCoT result on the right.
Each subplot is a 3-D scatter using the same viridis-by-token-step
encoding as ``analysis/plotting.make_3d_trajectory_plot``.

Outputs
-------
    results/compare_initial_vs_simcot/
        compare_dmd_3d.png       (Coconut + CODI, initial vs SimCoT)
        compare_phate_3d.png     (Coconut + CODI, initial vs SimCoT)

This script does NOT recompute reductions; it uses what
scripts/add_3d_plots.py already produced and saved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Default tree assignments (override with --paths if these move).
DEFAULT_TREES: Dict[Tuple[str, str], Path] = {
    ("coconut", "initial"): Path("results/coconut_gpt2/20260502_191252"),
    ("coconut", "simcot"):  Path("results/simcot_coconut_literal"),
    ("codi",    "initial"): Path("results/codi_gpt2/20260502_191252"),
    ("codi",    "simcot"):  Path("results/simcot_codi_literal"),
}


def _load_reduced(run_dir: Path, key: str) -> np.ndarray:
    p = run_dir / "reduced_states" / f"{key}_reduced.h5"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}. Run scripts/add_3d_plots.py on {run_dir} first."
        )
    with h5py.File(str(p), "r") as f:
        return f["embedding"][:]


def _load_meta(run_dir: Path) -> Dict[str, np.ndarray]:
    p = run_dir / "latent_states" / "all_states.h5"
    out: Dict[str, np.ndarray] = {}
    with h5py.File(str(p), "r") as f:
        for k in ("correct", "concept"):
            if k in f:
                out[k] = f[k][:]
    return out


def _draw_panel(
    ax,
    embeddings: np.ndarray,
    correct: np.ndarray,
    title: str,
    method_label: str,
    max_points: int = 30000,
    elev: float = 22.0,
    azim: float = -60.0,
):
    n_samples, n_steps, _ = embeddings.shape
    total_points = n_samples * n_steps
    if total_points > max_points:
        stride = int(np.ceil(total_points / max_points))
        keep = np.arange(n_samples)[::stride]
    else:
        keep = np.arange(n_samples)

    emb = embeddings[keep, :, :3].reshape(-1, 3)
    step_flat = np.tile(np.arange(n_steps), len(keep))
    correct_flat = np.repeat(correct.astype(bool)[keep], n_steps)

    sc = None
    m = correct_flat
    if m.any():
        sc = ax.scatter(
            emb[m, 0], emb[m, 1], emb[m, 2],
            c=step_flat[m], cmap="viridis",
            vmin=0, vmax=n_steps - 1,
            marker="o", s=14, edgecolors="none",
            alpha=0.55, depthshade=True,
        )
    m = ~correct_flat
    if m.any():
        sc_w = ax.scatter(
            emb[m, 0], emb[m, 1], emb[m, 2],
            c=step_flat[m], cmap="viridis",
            vmin=0, vmax=n_steps - 1,
            marker="^", s=22, facecolors="none", linewidths=0.7,
            alpha=0.80, depthshade=True,
        )
        sc = sc_w if sc is None else sc

    # Centroid trajectories
    correct_bool = correct.astype(bool)
    for label, mask, col, ls in [
        ("Correct",   correct_bool,  "#118a4b", "-"),
        ("Incorrect", ~correct_bool, "#a02c1e", "--"),
    ]:
        if not mask.any():
            continue
        centroid = embeddings[mask, :, :3].mean(axis=0)
        ax.plot(
            centroid[:, 0], centroid[:, 1], centroid[:, 2],
            ls, color=col, linewidth=2.4, marker="D",
            markersize=6, markerfacecolor="white",
            markeredgewidth=1.6, alpha=0.95,
        )

    ax.set_xlabel(f"{method_label} 1", fontsize=9, labelpad=-2)
    ax.set_ylabel(f"{method_label} 2", fontsize=9, labelpad=-2)
    ax.set_zlabel(f"{method_label} 3", fontsize=9, labelpad=-2)
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(axis="both", which="major", labelsize=7)
    return sc


def _build_compare_figure(
    method: str,
    reduce_key: str,
    reduce_label: str,
    out_path: Path,
    trees: Dict[Tuple[str, str], Path],
):
    """method in {'phate','dmd'}; reduce_key in {'phate3d','dmd3d'}."""
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.18)

    sc_handle = None
    rows = [("coconut", "COCONUT"), ("codi", "CODI")]
    cols = [("initial", "Initial GSM8K"), ("simcot", "SimCoT")]
    for r, (mkey, mlabel) in enumerate(rows):
        for c, (skey, slabel) in enumerate(cols):
            run_dir = trees[(mkey, skey)]
            try:
                emb = _load_reduced(run_dir, reduce_key)
                meta = _load_meta(run_dir)
            except FileNotFoundError as exc:
                print(f"  [skip] {exc}")
                continue
            n_total = len(meta["correct"])
            n_corr = int(meta["correct"].astype(bool).sum())
            title = (
                f"{mlabel}  —  {slabel}\n"
                f"N = {n_total:,}  |  acc = {100*n_corr/max(n_total,1):.2f}%"
            )
            ax = fig.add_subplot(gs[r, c], projection="3d")
            sc = _draw_panel(ax, emb, meta["correct"], title, reduce_label)
            if sc is not None:
                sc_handle = sc

    # Shared colorbar across the figure
    if sc_handle is not None:
        cax = fig.add_axes([0.92, 0.20, 0.012, 0.60])
        cbar = fig.colorbar(sc_handle, cax=cax)
        cbar.set_label(
            "Latent CoT token index  t  (0 = first, T-1 = final)",
            fontsize=10,
        )
        n_steps = emb.shape[1]
        cbar.set_ticks(range(n_steps))
        cbar.set_ticklabels([f"t={t}" for t in range(n_steps)])

    # Shared correctness / centroid legend
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#666", markeredgecolor="#666",
               markersize=8, label="● Correct (filled, colour=t)"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="none", markeredgecolor="#666",
               markeredgewidth=1.4, markersize=9,
               label="△ Incorrect (open, colour=t)"),
        Line2D([0], [0], color="#118a4b", linewidth=3, linestyle="-",
               label="Correct centroid path"),
        Line2D([0], [0], color="#a02c1e", linewidth=3, linestyle="--",
               label="Incorrect centroid path"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=10,
        framealpha=0.9,
    )

    fig.suptitle(
        f"Latent Trajectories — {reduce_label} (3-D)  |  Initial GSM8K vs SimCoT",
        fontsize=14,
        y=0.99,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/compare_initial_vs_simcot"),
        help="Output directory for the comparison figures.",
    )
    args = parser.parse_args()

    trees = DEFAULT_TREES
    out = args.out

    print(f"\n=== Building 3-D PHATE comparison ===")
    _build_compare_figure(
        method="phate",
        reduce_key="phate3d",
        reduce_label="PHATE",
        out_path=out / "compare_phate_3d.png",
        trees=trees,
    )

    print(f"\n=== Building 3-D DMD comparison ===")
    _build_compare_figure(
        method="dmd",
        reduce_key="dmd3d",
        reduce_label="DMD",
        out_path=out / "compare_dmd_3d.png",
        trees=trees,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
