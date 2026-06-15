"""
Run the analysis pipeline starting from a cached
latent_states/all_states.h5 file rather than from a fresh inference run.

Equivalent to `python runner.py --paradigm X --method Y --run_dir <dir>`
but standalone, with reduction hyper-parameters exposed as CLI flags.
Read-only with respect to the cached HDF5; writes its outputs into the
same run directory:

    <run_dir>/latent_states/all_states.h5         (input, untouched)
    <run_dir>/reduced_states/{pca,tsne,umap,dmd,phate}_reduced.h5
    <run_dir>/reduced_states/dmd_spectrum.h5
    <run_dir>/stability_feats/stability.h5
    <run_dir>/plots/                               (combined + concept)
    <run_dir>/plots/per_concept/                   (one figure per bucket)

The active-perturbation test is a separate flow
(`runner.py --perturbation` or scripts/perturbation/add_perturbation.py)
because it requires re-running the model with noise. All post-hoc
trajectory features (step-to-step change, direction consistency, arc
length, fixed-point distances, Lyapunov sensitivity) are computed here
from the cached states.

Usage
-----
    python scripts/replay/analyze_from_cache.py results/vanilla_codi/<ts>

Or with explicit configuration:
    python scripts/replay/analyze_from_cache.py results/vanilla_codi/<ts> \\
        --tsne-perplexity 5.0 --umap-neighbors 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.dim_reduct import DimReduct
from analysis.stability import StabilityAnalysis
from analysis.plotting import Plotting
from analysis.data_prep import GSM8K_CONCEPT_NAMES


def _load_states(run_dir: Path) -> Dict[str, np.ndarray]:
    h5_path = run_dir / "latent_states" / "all_states.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Expected cached HDF5 at {h5_path}. Produce it first via "
            f"runner.py."
        )
    out: Dict[str, np.ndarray] = {}
    with h5py.File(str(h5_path), "r") as f:
        for k in ("latent_thoughts", "correct", "concept", "question"):
            if k in f:
                out[k] = f[k][:]
    if "latent_thoughts" not in out:
        raise RuntimeError(f"{h5_path} missing required dataset 'latent_thoughts'")
    if "correct" not in out:
        raise RuntimeError(f"{h5_path} missing required dataset 'correct'")
    if "concept" not in out:
        # Fall back: everything in the catch-all bucket so downstream
        # plotting still works (just produces one per_concept figure).
        out["concept"] = np.full(len(out["correct"]), 6, dtype=np.int32)
    return out


def _save_dmd_spectrum(dim_reduct: DimReduct, output_dir: Path) -> None:
    """Match runner.py:save_dmd_spectrum for cache compatibility."""
    spec = dim_reduct.get_dmd_spectral_summary()
    out = output_dir / "reduced_states" / "dmd_spectrum.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out), "w") as f:
        for k, v in spec.items():
            f.create_dataset(k, data=np.asarray(v))
    print(f"Saved DMD spectral summary to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path,
        help="Directory containing latent_states/all_states.h5 "
             "(e.g. results/simcot_codi_canonical/<ts>)",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=5.0)
    parser.add_argument("--umap-neighbors", type=int, default=5)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--phate-knn", type=int, default=5)
    parser.add_argument("--plot-format", default="png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    print(f"[analyze] run_dir = {run_dir}")

    # ─── Load cached states ─────────────────────────────────────────────────
    cached = _load_states(run_dir)
    states_np = cached["latent_thoughts"].astype(np.float32)
    correct_mask = cached["correct"].astype(bool)
    concept_labels = cached["concept"].astype(np.int32)
    n, T, d = states_np.shape
    print(f"[analyze] loaded {n} samples, {T} latent steps, {d}-dim states; "
          f"accuracy in cache = {correct_mask.mean()*100:.2f}%")

    states = torch.from_numpy(states_np)
    label_names = GSM8K_CONCEPT_NAMES

    # ─── Step 3: Dimensionality reduction ───────────────────────────────────
    print("\n[Step 3] Dimensionality reduction (PCA, t-SNE, UMAP, DMD, PHATE)...")
    dim_reduct = DimReduct(states)
    reductions: Dict[str, np.ndarray] = {}
    try:
        reductions["pca"] = dim_reduct.perform_pca(n_components=2)
    except Exception as e:
        print(f"  PCA failed: {e}")
        reductions["pca"] = None
    try:
        reductions["tsne"] = dim_reduct.perform_tsne(
            n_components=2, perplexity=args.tsne_perplexity, metric="euclidean",
        )
    except Exception as e:
        print(f"  t-SNE failed: {e}")
        reductions["tsne"] = None
    try:
        reductions["umap"] = dim_reduct.perform_umap(
            n_components=2, n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist, metric="euclidean",
        )
    except Exception as e:
        print(f"  UMAP failed: {e}")
        reductions["umap"] = None
    try:
        reductions["dmd"] = dim_reduct.perform_dmd(svd_rank=-1, exact=True)
    except Exception as e:
        print(f"  DMD failed: {e}")
        reductions["dmd"] = None
    try:
        reductions["phate"] = dim_reduct.perform_phate(
            n_components=2, knn=args.phate_knn, t="auto",
        )
    except Exception as e:
        print(f"  PHATE failed: {e}")
        reductions["phate"] = None

    dim_reduct.save_reduced_states(str(run_dir))
    if reductions.get("dmd") is not None:
        try:
            _save_dmd_spectrum(dim_reduct, run_dir)
        except Exception as e:
            print(f"  DMD spectrum save failed: {e}")

    # ─── Step 4: Trajectory features (geometric + stability, no perturbation) ──
    print("\n[Step 4] Trajectory features (geometric + Lyapunov + fixed-point)...")
    stab = StabilityAnalysis(states=states, model_wrapper=None, questions=None)
    stab.compute_all(include_perturbation=False)
    stab.summary()
    stab.save_features(str(run_dir))

    # ─── Step 5: All plots ──────────────────────────────────────────────────
    print("\n[Step 5] Generating plots...")
    plotter = Plotting(
        output_dir=str(run_dir), plot_format=args.plot_format, dpi=args.dpi,
    )
    step_changes = stab.features["step2step_change"]
    dir_consistency = stab.features["direction_consistency"]
    arc_lengths = stab.features["arc_length"]

    # Pooled trajectory plots per reduction
    for name, emb in reductions.items():
        if emb is not None:
            try:
                plotter.make_trajectory_plot(emb, name, correct_mask=correct_mask)
            except Exception as e:
                print(f"  trajectory_{name} failed: {e}")

    # Geometric feature plots
    plotter.make_step_change_plot(step_changes, correct_mask=correct_mask)
    plotter.make_direction_consistency_plot(dir_consistency, correct_mask=correct_mask)
    plotter.make_arc_length_plot(
        arc_lengths, correct_mask=correct_mask,
        concept_labels=concept_labels, concept_names=label_names,
    )

    # Stability plots
    if "local_lyapunov" in stab.features:
        plotter.make_lyapunov_plot(stab.features["local_lyapunov"], correct_mask=correct_mask)
    if "distances_lag_1" in stab.features and "distances_lag_2" in stab.features:
        plotter.make_fixed_point_plot(
            stab.features["distances_lag_1"], stab.features["distances_lag_2"],
            correct_mask=correct_mask,
        )
    # DMD spectrum plot
    if reductions.get("dmd") is not None:
        try:
            spec = dim_reduct.get_dmd_spectral_summary()
            if states.shape[0] > 200:
                plotter.make_dmd_spectrum_aggregate_plot(spec, correct_mask=correct_mask)
            else:
                plotter.make_dmd_spectrum_plot(
                    spec["eigenvalues_real"], spec["eigenvalues_imag"],
                    correct_mask=correct_mask,
                )
        except Exception as e:
            print(f"  dmd spectrum plot failed: {e}")

    # Concept-coloured pooled trajectory plots
    print("\n[Step 6b] Concept-coloured trajectory plots...")
    for name, emb in reductions.items():
        if emb is not None:
            try:
                plotter.make_concept_trajectory_plot(
                    emb, name,
                    concept_labels=concept_labels, correct_mask=correct_mask,
                    concept_names=label_names,
                )
            except Exception as e:
                print(f"  concept_trajectory_{name} failed: {e}")

    # Per-concept trajectory plots (separate figures)
    print("\n[Step 6b'] Per-concept trajectory plots (separate figures)...")
    for name, emb in reductions.items():
        if emb is not None:
            try:
                plotter.make_per_concept_trajectory_plots(
                    emb, name,
                    concept_labels=concept_labels, concept_names=label_names,
                    correct_mask=correct_mask,
                )
            except Exception as e:
                print(f"  per_concept_trajectory_{name} failed: {e}")

    # Concept-wise metric plots
    print("\n[Step 6c] Concept-wise metric plots...")
    concept_specs = [
        (step_changes, "step2step_change", r"$\|z_{t+1} - z_t\|$"),
        (dir_consistency, "direction_consistency", r"$\cos(\Delta z_t, \Delta z_{t-1})$"),
        (arc_lengths, "arc_length", r"Arc Length $\sum \|z_{t+1}-z_t\|$"),
    ]
    if "local_lyapunov" in stab.features:
        concept_specs.append((
            stab.features["local_lyapunov"], "lyapunov_sensitivity",
            r"$\log(\|\Delta z_{t+1}\| / \|\Delta z_t\|)$",
        ))
    if "distances_lag_1" in stab.features:
        concept_specs.append((
            stab.features["distances_lag_1"], "fixed_point_lag1",
            r"$\|z_{t+1} - z_t\|$ (lag=1)",
        ))
    for data, mname, ylabel in concept_specs:
        try:
            plotter.make_concept_metric_plot(
                data, concept_labels=concept_labels,
                correct_mask=correct_mask, concept_names=label_names,
                metric_name=mname, ylabel=ylabel,
            )
        except Exception as e:
            print(f"  concept_{mname} failed: {e}")

    plotter.save_figs()
    print(f"\n[done] All artefacts written under {run_dir}")
    print(f"       latent_states/  reduced_states/  "
          f"stability_feats/  plots/  plots/per_concept/")


if __name__ == "__main__":
    main()
