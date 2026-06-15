"""
Re-render every plot from a finished run's HDF5 artefacts.

Reads:
  <run>/latent_states/all_states.h5      — latent_thoughts, correct, concept
  <run>/reduced_states/*.h5               — pca / tsne / umap / dmd / phate
  <run>/reduced_states/dmd_spectrum.h5    — eigenvalues, growth rates
  <run>/stability_feats/stability.h5      — geometric features, local_lyapunov,
                                            perturbation_divergence, etc.
  <run>/dynamic_feats/features.h5         — legacy fallback for run trees made
                                            before the geometric features were
                                            merged into stability.h5

Writes every plot back into <run>/plots/.

No model loading, no inference — cheap (~60–90 s for N=8,792).

Usage:
    python scripts/replay/replot_from_cache.py <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Project-local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analysis.plotting import Plotting
from analysis.data_prep import GSM8K_CONCEPT_NAMES


def _load_h5(path: Path, keys: list[str] | None = None) -> dict:
    """Open an HDF5 file and return the requested datasets as numpy arrays."""
    out = {}
    with h5py.File(str(path), "r") as f:
        if keys is None:
            keys = list(f.keys())
        for k in keys:
            if k in f:
                out[k] = f[k][:]
    return out


def replot(run_dir: str | Path) -> None:
    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run}")

    print(f"\n{'=' * 70}\nReplotting: {run}\n{'=' * 70}")

    # ── Load every artefact ────────────────────────────────────────────────
    ls = _load_h5(run / "latent_states" / "all_states.h5")
    correct = ls["correct"].astype(bool)
    concept = ls["concept"]
    n_total = int(len(correct))
    print(f"  N samples: {n_total:,}  (accuracy: {correct.mean():.4f})")

    sfeats = _load_h5(run / "stability_feats" / "stability.h5")

    # Geometric features live in stability.h5 since the DynamicFeats merge;
    # older run trees still have them in dynamic_feats/features.h5.
    geometric_keys = ("step2step_change", "direction_consistency", "arc_length")
    if all(k in sfeats for k in geometric_keys):
        dfeats = {k: sfeats[k] for k in geometric_keys}
    else:
        dfeats = _load_h5(run / "dynamic_feats" / "features.h5")

    reductions: dict[str, np.ndarray] = {}
    for name in ("pca", "tsne", "umap", "dmd", "phate"):
        p = run / "reduced_states" / f"{name}_reduced.h5"
        if p.exists():
            r = _load_h5(p, ["embedding"])
            if "embedding" in r:
                reductions[name] = r["embedding"]

    dmd_spec_path = run / "reduced_states" / "dmd_spectrum.h5"
    dmd_spec = _load_h5(dmd_spec_path) if dmd_spec_path.exists() else None

    # ── Plotter ────────────────────────────────────────────────────────────
    # dpi=110 keeps every figure under 2000px on both axes (widest figsize
    # is 18" → 1980px).  Plenty of resolution for Slack/GDoc review and
    # fits under Anthropic's many-image request dimension cap.
    plotter = Plotting(output_dir=str(run), plot_format="png", dpi=110)

    # Trajectory plots
    for name, emb in reductions.items():
        try:
            plotter.make_trajectory_plot(emb, name, correct_mask=correct)
        except Exception as e:
            print(f"  ✗ trajectory_{name}: {e}")

    # Geometric features (aggregate, split by correctness)
    plotter.make_step_change_plot(
        dfeats["step2step_change"], correct_mask=correct)
    plotter.make_direction_consistency_plot(
        dfeats["direction_consistency"], correct_mask=correct)
    plotter.make_arc_length_plot(
        dfeats["arc_length"], correct_mask=correct,
        concept_labels=concept, concept_names=GSM8K_CONCEPT_NAMES,
    )

    # Stability plots
    if "local_lyapunov" in sfeats:
        plotter.make_lyapunov_plot(sfeats["local_lyapunov"], correct_mask=correct)

    if "distances_lag_1" in sfeats and "distances_lag_2" in sfeats:
        plotter.make_fixed_point_plot(
            sfeats["distances_lag_1"], sfeats["distances_lag_2"],
            correct_mask=correct,
        )

    if "perturbation_divergence" in sfeats:
        plotter.make_perturbation_divergence_plot(
            sfeats["perturbation_divergence"],
            divergence_std=sfeats.get("perturbation_divergence_std"),
            correct_mask=correct,
            relative_divergence=sfeats.get("perturbation_relative_divergence"),
        )

    # DMD spectrum (complex-plane scatter is per-sample — skip if huge)
    if dmd_spec is not None and n_total <= 200:
        try:
            plotter.make_dmd_spectrum_plot(
                dmd_spec["eigenvalues_real"],
                dmd_spec["eigenvalues_imag"],
                correct_mask=correct,
            )
        except Exception as e:
            print(f"  ✗ dmd_spectrum: {e}")
    elif dmd_spec is not None:
        # For N > 200 the per-sample scatter is unreadable — emit a
        # 2-panel aggregate view instead (histogram of |λ| and a density
        # scatter in the complex plane).  Delegate to a new helper.
        try:
            plotter.make_dmd_spectrum_aggregate_plot(dmd_spec, correct_mask=correct)
        except AttributeError:
            print("  (DMD aggregate plot helper not yet present — skipping)")
        except Exception as e:
            print(f"  ✗ dmd_spectrum aggregate: {e}")

    # Concept-wise trajectory plots (pooled, coloured by concept)
    for name, emb in reductions.items():
        try:
            plotter.make_concept_trajectory_plot(
                emb, name,
                concept_labels=concept,
                correct_mask=correct,
                concept_names=GSM8K_CONCEPT_NAMES,
            )
        except Exception as e:
            print(f"  ✗ concept_trajectory_{name}: {e}")

    # Per-concept trajectory plots (one figure per concept under plots/per_concept/)
    for name, emb in reductions.items():
        try:
            plotter.make_per_concept_trajectory_plots(
                emb, name,
                concept_labels=concept,
                concept_names=GSM8K_CONCEPT_NAMES,
                correct_mask=correct,
            )
        except Exception as e:
            print(f"  ✗ per_concept_trajectory_{name}: {e}")

    # Concept-wise metric plots
    concept_specs = [
        (dfeats["step2step_change"],       "step2step_change",
         r"$\|z_{t+1} - z_t\|$"),
        (dfeats["direction_consistency"],  "direction_consistency",
         r"$\cos(\Delta z_t, \Delta z_{t-1})$"),
        (dfeats["arc_length"],             "arc_length",
         r"Arc Length $\sum \|z_{t+1}-z_t\|$"),
    ]
    if "local_lyapunov" in sfeats:
        concept_specs.append((
            sfeats["local_lyapunov"], "lyapunov_sensitivity",
            r"$\log(\|\Delta z_{t+1}\| / \|\Delta z_t\|)$",
        ))
    if "distances_lag_1" in sfeats:
        concept_specs.append((
            sfeats["distances_lag_1"], "fixed_point_lag1",
            r"$\|z_{t+1} - z_t\|$ (lag=1)",
        ))

    for data, name, ylabel in concept_specs:
        try:
            plotter.make_concept_metric_plot(
                data, concept_labels=concept,
                correct_mask=correct,
                concept_names=GSM8K_CONCEPT_NAMES,
                metric_name=name, ylabel=ylabel,
            )
        except Exception as e:
            print(f"  ✗ concept_{name}: {e}")

    plotter.save_figs()
    print(f"  All plots saved to {run / 'plots'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs", nargs="+",
        help="One or more run directories under results/",
    )
    args = parser.parse_args()
    for d in args.run_dirs:
        replot(d)


if __name__ == "__main__":
    main()
