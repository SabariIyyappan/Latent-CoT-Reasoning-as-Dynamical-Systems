"""
Runner for Latent COT Dynamical Systems Analysis.

Single entry point for the paper's 2x2 paradigm x method grid. Three flows:

1. MAIN PIPELINE (default)
       load model -> run inference -> latent_states/all_states.h5
       -> dimensionality reduction + trajectory features + plots
   If --run_dir points at a directory whose trajectories H5 already
   exists, inference is skipped and the analysis runs on the cache.

2. ANALYSIS-ONLY (implicit)
       runner.py --paradigm X --method Y --run_dir results/X_Y/<ts>
   Trajectories available -> straight to analysis + plotting. No model
   or GPU needed.

3. PERTURBATION (separate flow, --perturbation)
       runner.py --paradigm X --method Y --perturbation --run_dir <dir>
   Takes the saved H5 trajectories as the clean baseline, re-runs the
   model with Gaussian noise on input embeddings, merges divergence
   metrics into stability_feats/stability.h5 and writes the plot.

Checkpoint bootstrap: for --paradigm simcot the upstream HuggingFace
release is translated automatically (simcot/bootstrap_<method>.py) when
the local checkpoint folder is missing, so runner.py is the only
command an evaluator needs.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import h5py
import yaml
import torch
import numpy as np

from analysis.wrappers import ModelWrapper
from analysis.data_prep import DataPrep, GSM8K_CONCEPT_NAMES
from analysis.dim_reduct import DimReduct
from analysis.stability import StabilityAnalysis
from analysis.plotting import Plotting


VANILLA_HF_REPOS = {
    "codi": "ModalityDance/latent-tts-codi",
    "coconut": "ModalityDance/latent-tts-coconut",
}


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_output_dir(config: dict, paradigm: str | None = None) -> str:
    """Create timestamped output directory for this experiment run.

    Naming convention for the cam-ready trunk:
        results/<paradigm>_<method>/<YYYYMMDD_HHMMSS>/
    e.g. results/simcot_codi/20260530_143012/

    When paradigm is None (legacy path) it falls back to the older
    results/<method>_<model>/<dataset>_<timestamp>/ scheme so existing
    config-only invocations keep working.
    """
    base = Path(config["output"]["base_dir"])
    method = config["method"]["name"]
    model = config["method"]["model_type"]
    dataset = config["dataset"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if paradigm is not None:
        output_dir = base / f"{paradigm}_{method}" / timestamp
    else:
        output_dir = base / f"{method}_{model}" / f"{dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Output directory: {output_dir}")
    return str(output_dir)


def ensure_checkpoint(config: dict, paradigm: str | None, method: str) -> None:
    """Make sure the method's checkpoint exists before loading the model.

    Sim-CoT: bootstrap the upstream HuggingFace release automatically
    (one-time translation into the layout ModelWrapper expects).
    Vanilla: print the exact download command and exit — the checkpoints
    are plain HF snapshots, no translation needed.
    """
    ckpt = Path(config["model"][method]["checkpoint"])
    if (ckpt / "config.json").is_file():
        return

    if paradigm == "simcot":
        bootstrap = Path("simcot") / f"bootstrap_{method}.py"
        print(f"Checkpoint missing at {ckpt} — running one-time bootstrap:")
        print(f"  {sys.executable} {bootstrap} --out {ckpt}")
        subprocess.run(
            [sys.executable, str(bootstrap), "--out", str(ckpt)],
            check=True,
        )
        if not (ckpt / "config.json").is_file():
            raise SystemExit(f"Bootstrap finished but {ckpt}/config.json still missing.")
    else:
        repo = VANILLA_HF_REPOS.get(method, "<hf-repo>")
        raise SystemExit(
            f"Checkpoint missing at {ckpt}. Download it first:\n"
            f"  huggingface-cli download {repo} --local-dir {ckpt}"
        )


def save_dmd_spectrum(dim_reduct: DimReduct, output_dir: str):
    """Persist DMD spectral data to HDF5."""
    spec = dim_reduct.get_dmd_spectral_summary()
    out = Path(output_dir) / "reduced_states" / "dmd_spectrum.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out), "w") as f:
        for k, v in spec.items():
            f.create_dataset(k, data=np.asarray(v))
    print(f"Saved DMD spectral summary to {out}")


# ─────────────────────────────────────────────────────────────────────
# Flow 1: inference — model load + dataset run + H5 trajectory cache
# ─────────────────────────────────────────────────────────────────────

def run_inference(config: dict, output_dir: str):
    """Load the model, run inference over the dataset, save latent states.

    Returns:
        (all_states, correct_mask, concept_labels, questions, label_names)
    """
    print("\n[Inference 1/2] Loading model...")
    method = config["method"]["name"]
    checkpoint = config["model"][method]["checkpoint"]

    wrapper = ModelWrapper(
        method=method,
        checkpoint=checkpoint,
        device=config["experiment"].get("device", "cpu"),
        latent_length=config["method"].get("latent_length", 6),
        max_new_tokens=config["method"].get("max_new_tokens", 512),
        do_sample=config["method"].get("do_sample", False),
    )
    print(f"Loaded {method} model from {checkpoint}")

    print("\n[Inference 2/2] Running inference on dataset...")
    data_prep = DataPrep(
        model_wrapper=wrapper,
        dataset_name=config["dataset"]["name"],
        split=config["dataset"].get("split", "test"),
        n_samples=config["dataset"].get("n_samples", 500),
        stratified=config["dataset"].get("stratified", True),
        seed=config["experiment"].get("seed", 42),
    )
    results = data_prep.execute_method()
    data_prep.save_latent_states(output_dir)

    label_names = data_prep.get_label_names()
    all_states = torch.stack([r["latent_thoughts"] for r in results], dim=0)
    correct_mask = np.array([r["correct"] for r in results])
    concept_labels = np.array([r["concept"] for r in results])
    questions = [r["question"] for r in results]

    print(f"\nLatent states shape: {all_states.shape}")
    print(f"Correct: {correct_mask.sum()}/{len(correct_mask)}")

    return all_states, correct_mask, concept_labels, questions, label_names


def load_cached_trajectories(run_dir: str):
    """Read trajectories from <run_dir>/latent_states/all_states.h5.

    Returns the same tuple as run_inference so the analysis flow is
    agnostic to whether the trajectories are fresh or cached.
    """
    h5_path = Path(run_dir) / "latent_states" / "all_states.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"No cached trajectories at {h5_path}. Run the main pipeline first."
        )

    with h5py.File(str(h5_path), "r") as f:
        states_np = f["latent_thoughts"][:].astype(np.float32)
        correct_mask = f["correct"][:].astype(bool)
        if "concept" in f:
            concept_labels = f["concept"][:].astype(np.int32)
        else:
            concept_labels = np.full(len(correct_mask), 6, dtype=np.int32)
        if "question" in f:
            questions = [
                q.decode("utf-8") if isinstance(q, bytes) else str(q)
                for q in f["question"][:]
            ]
        else:
            questions = []

    all_states = torch.from_numpy(states_np)
    print(f"Loaded cached trajectories from {h5_path}")
    print(f"  shape={tuple(all_states.shape)}, accuracy={correct_mask.mean()*100:.2f}%")
    return all_states, correct_mask, concept_labels, questions, GSM8K_CONCEPT_NAMES


# ─────────────────────────────────────────────────────────────────────
# Flow 2: analysis — reductions + trajectory features + plots
# ─────────────────────────────────────────────────────────────────────

def run_analysis(config: dict, output_dir: str, all_states, correct_mask,
                 concept_labels, label_names):
    """Dimensionality reduction, trajectory features, and every plot."""

    # ─── Dimensionality reduction ───────────────────────────────────
    print("\n[Analysis 1/3] Dimensionality reduction...")
    dim_reduct = DimReduct(all_states)

    dr_cfg = config["dim_reduction"]
    reductions = {}

    reduction_methods = [
        ("pca", dim_reduct.perform_pca, dr_cfg.get("pca", {})),
        ("tsne", dim_reduct.perform_tsne, dr_cfg.get("tsne", {})),
        ("umap", dim_reduct.perform_umap, dr_cfg.get("umap", {})),
        ("dmd", dim_reduct.perform_dmd, dr_cfg.get("dmd", {})),
        ("phate", dim_reduct.perform_phate, dr_cfg.get("phate", {})),
    ]

    for name, fn, kwargs in reduction_methods:
        try:
            reductions[name] = fn(**kwargs)
        except Exception as e:
            print(f"  {name.upper()} failed: {e}")
            reductions[name] = None

    dim_reduct.save_reduced_states(output_dir)

    if reductions["dmd"] is not None:
        try:
            save_dmd_spectrum(dim_reduct, output_dir)
        except Exception as e:
            print(f"  DMD spectrum save failed: {e}")

    # ─── Trajectory features (geometric + stability) ────────────────
    print("\n[Analysis 2/3] Trajectory features (geometric + stability)...")
    stab = StabilityAnalysis(states=all_states)
    stab.compute_all(include_perturbation=False)
    stab.summary()
    stab.save_features(output_dir)

    # ─── Plots ──────────────────────────────────────────────────────
    print("\n[Analysis 3/3] Generating plots...")
    plotter = Plotting(
        output_dir=output_dir,
        plot_format=config["output"].get("plot_format", "png"),
        dpi=config["output"].get("dpi", 300),
        dataset_name=config["dataset"]["name"],
    )

    # Trajectory plots for each reduction method
    for name, emb in reductions.items():
        if emb is not None:
            try:
                plotter.make_trajectory_plot(emb, name, correct_mask=correct_mask)
            except Exception as e:
                print(f"  trajectory_{name} plot failed: {e}")

    # Geometric feature plots
    step_changes = stab.features["step2step_change"]
    dir_consistency = stab.features["direction_consistency"]
    arc_lengths = stab.features["arc_length"]

    plotter.make_step_change_plot(step_changes, correct_mask=correct_mask)
    plotter.make_direction_consistency_plot(dir_consistency, correct_mask=correct_mask)
    plotter.make_arc_length_plot(
        arc_lengths,
        correct_mask=correct_mask,
        concept_labels=concept_labels,
        concept_names=label_names,
    )

    # Stability plots
    if "local_lyapunov" in stab.features:
        plotter.make_lyapunov_plot(stab.features["local_lyapunov"], correct_mask=correct_mask)

    if "distances_lag_1" in stab.features and "distances_lag_2" in stab.features:
        plotter.make_fixed_point_plot(
            stab.features["distances_lag_1"],
            stab.features["distances_lag_2"],
            correct_mask=correct_mask,
        )

    # DMD spectrum plot — aggregate view for large N (per-sample scatter
    # becomes an opaque cloud at N > ~200).
    if reductions["dmd"] is not None:
        try:
            spec = dim_reduct.get_dmd_spectral_summary()
            if all_states.shape[0] > 200:
                plotter.make_dmd_spectrum_aggregate_plot(
                    spec, correct_mask=correct_mask,
                )
            else:
                plotter.make_dmd_spectrum_plot(
                    spec["eigenvalues_real"],
                    spec["eigenvalues_imag"],
                    correct_mask=correct_mask,
                )
        except Exception as e:
            print(f"  dmd spectrum plot failed: {e}")

    # Concept-wise trajectory plots
    for name, emb in reductions.items():
        if emb is not None:
            try:
                plotter.make_concept_trajectory_plot(
                    emb, name,
                    concept_labels=concept_labels,
                    correct_mask=correct_mask,
                    concept_names=label_names,
                )
            except Exception as e:
                print(f"  concept_trajectory_{name} plot failed: {e}")

    # Per-concept trajectory plots in separate figures under plots/per_concept/
    for name, emb in reductions.items():
        if emb is not None:
            try:
                plotter.make_per_concept_trajectory_plots(
                    emb, name,
                    concept_labels=concept_labels,
                    concept_names=label_names,
                    correct_mask=correct_mask,
                )
            except Exception as e:
                print(f"  per_concept_trajectory_{name} plots failed: {e}")

    # Concept-wise metric plots
    concept_metric_specs = [
        (step_changes,    "step2step_change",        r"$\|z_{t+1} - z_t\|$"),
        (dir_consistency, "direction_consistency",   r"$\cos(\Delta z_t, \Delta z_{t-1})$"),
        (arc_lengths,     "arc_length",              r"Arc Length $\sum \|z_{t+1} - z_t\|$"),
    ]
    if "local_lyapunov" in stab.features:
        concept_metric_specs.append((
            stab.features["local_lyapunov"],
            "lyapunov_sensitivity",
            r"$\log(\|\Delta z_{t+1}\| / \|\Delta z_t\|)$",
        ))
    if "distances_lag_1" in stab.features:
        concept_metric_specs.append((
            stab.features["distances_lag_1"],
            "fixed_point_lag1",
            r"$\|z_{t+1} - z_t\|$ (lag=1)",
        ))

    for metric_data, metric_name, ylabel in concept_metric_specs:
        try:
            plotter.make_concept_metric_plot(
                metric_data,
                concept_labels=concept_labels,
                correct_mask=correct_mask,
                concept_names=label_names,
                metric_name=metric_name,
                ylabel=ylabel,
            )
        except Exception as e:
            print(f"  concept_{metric_name} plot failed: {e}")

    plotter.save_figs()


# ─────────────────────────────────────────────────────────────────────
# Flow 3: perturbation — separate flow over cached H5 trajectories
# ─────────────────────────────────────────────────────────────────────

def run_perturbation(config: dict, run_dir: str,
                     noise_std: float | None = None,
                     n_perturbations: int | None = None,
                     max_samples: int | None = None):
    """Perturbation stability as its own flow.

    Input is the cached latent_states/all_states.h5 of a finished run
    (the clean baseline). The model is re-loaded and re-run with
    Gaussian noise on the input embeddings; divergence metrics merge
    into the run's stability_feats/stability.h5 and the figure lands in
    its plots/ folder.

    For the full paper grid (sigma in {0.01, 0.1, 1.0} x both methods,
    N=2000 stratified) use scripts/perturbation/add_perturbation.py,
    which adds sigma-suffixed keys and filenames so runs coexist.
    """
    pert_cfg = config.get("perturbation", {})
    if noise_std is None:
        noise_std = float(pert_cfg.get("noise_std", 0.01))
    if n_perturbations is None:
        n_perturbations = int(pert_cfg.get("n_perturbations", 3))

    print("=" * 70)
    print("Perturbation flow (separate from main pipeline)")
    print(f"  run_dir          = {run_dir}")
    print(f"  noise_std        = {noise_std}")
    print(f"  n_perturbations  = {n_perturbations}")
    print("=" * 70)

    all_states, correct_mask, _, questions, _ = load_cached_trajectories(run_dir)
    if not questions:
        raise SystemExit(
            "Cached H5 has no 'question' dataset — perturbation needs the "
            "original questions to re-run the model."
        )
    if max_samples is not None and max_samples < len(questions):
        print(f"Capping at first {max_samples} of {len(questions)} samples")
        all_states = all_states[:max_samples]
        correct_mask = correct_mask[:max_samples]
        questions = questions[:max_samples]

    method = config["method"]["name"]
    checkpoint = config["model"][method]["checkpoint"]
    wrapper = ModelWrapper(
        method=method,
        checkpoint=checkpoint,
        device=config["experiment"].get("device", "cpu"),
        latent_length=config["method"].get("latent_length", 6),
        max_new_tokens=config["method"].get("max_new_tokens", 512),
        do_sample=config["method"].get("do_sample", False),
    )
    print(f"Loaded {method} model from {checkpoint}")

    stab = StabilityAnalysis(
        states=all_states, model_wrapper=wrapper, questions=questions,
    )
    stab.calc_perturbation_stability(
        noise_std=noise_std,
        n_perturbations=n_perturbations,
        seed=config["experiment"].get("seed", 42),
        clean_states=all_states,
    )

    # Merge into the run's stability.h5 without clobbering existing features
    h5_path = Path(run_dir) / "stability_feats" / "stability.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "a") as f:
        for name, arr in stab.features.items():
            if name in f:
                del f[name]
            f.create_dataset(name, data=np.asarray(arr))
    print(f"Merged perturbation features into {h5_path}")

    plotter = Plotting(
        output_dir=run_dir,
        plot_format=config["output"].get("plot_format", "png"),
        dpi=config["output"].get("dpi", 300),
        dataset_name=config["dataset"]["name"],
    )
    plotter.make_perturbation_divergence_plot(
        stab.features["perturbation_divergence"],
        divergence_std=stab.features.get("perturbation_divergence_std"),
        correct_mask=correct_mask,
        relative_divergence=stab.features.get("perturbation_relative_divergence"),
    )
    plotter.save_figs()

    print("\n" + "=" * 70)
    print(f"Perturbation flow complete. Outputs under: {run_dir}")
    print("=" * 70)
    return run_dir


# ─────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────

def run(config_path: str = "config.yaml", paradigm: str | None = None,
        method_override: str | None = None, n_samples_override: int | None = None,
        device_override: str | None = None, run_dir: str | None = None):
    """Main pipeline: inference (skipped when trajectories exist) + analysis.

    Args:
        config_path: YAML config (under vanilla/configs/ or simcot/configs/).
        paradigm: One of {"vanilla", "simcot"}. Names the output directory
            results/<paradigm>_<method>/<timestamp>/.
        method_override: Override config["method"]["name"].
        n_samples_override: Override config["dataset"]["n_samples"].
        device_override: Override config["experiment"]["device"] (e.g. "cpu").
        run_dir: Existing run directory. When its trajectories H5 exists,
            inference is skipped and analysis runs in place.
    """
    print("=" * 70)
    print("Latent COT Dynamical Systems Analysis")
    print(f"  paradigm = {paradigm or '(legacy, from config)'}")
    print(f"  config   = {config_path}")
    print("=" * 70)

    config = load_config(config_path)
    if method_override is not None:
        config["method"]["name"] = method_override
    if n_samples_override is not None:
        config.setdefault("dataset", {})["n_samples"] = n_samples_override
    if device_override is not None:
        config.setdefault("experiment", {})["device"] = device_override

    seed = config["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cached_h5 = (
        Path(run_dir) / "latent_states" / "all_states.h5" if run_dir else None
    )
    if cached_h5 is not None and cached_h5.is_file():
        # Trajectories available -> analysis + plotting only.
        output_dir = str(run_dir)
        (all_states, correct_mask, concept_labels,
         _questions, label_names) = load_cached_trajectories(output_dir)
    else:
        # Trajectories not available -> inference first.
        if run_dir is not None:
            print(f"No trajectories under {run_dir} — running inference fresh.")
        ensure_checkpoint(config, paradigm, config["method"]["name"])
        output_dir = setup_output_dir(config, paradigm=paradigm)
        (all_states, correct_mask, concept_labels,
         _questions, label_names) = run_inference(config, output_dir)

    run_analysis(config, output_dir, all_states, correct_mask,
                 concept_labels, label_names)

    # Record the output dir in a portable temp file so Colab display cells
    # can find the most recent run without relying on directory sorting.
    try:
        last_run_path = Path(tempfile.gettempdir()) / "last_run_dir.txt"
        last_run_path.write_text(str(output_dir))
    except OSError:
        pass  # non-fatal — Colab and CLI both work without it

    print("\n" + "=" * 70)
    print(f"Analysis complete! Results saved to: {output_dir}")
    print("=" * 70)

    return output_dir


def _default_config(paradigm: str, method: str) -> str:
    """Default YAML path for a (paradigm, method) cell."""
    return f"{paradigm}/configs/inference_{method}_gsm8k.yaml"


def _latest_run_dir(paradigm: str, method: str) -> str | None:
    """Most recent results/<paradigm>_<method>/<timestamp>/ with an H5."""
    base = Path("results") / f"{paradigm}_{method}"
    if not base.is_dir():
        return None
    candidates = sorted(
        (d for d in base.iterdir()
         if (d / "latent_states" / "all_states.h5").is_file()),
        key=lambda d: d.name,
    )
    return str(candidates[-1]) if candidates else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Latent COT Dynamical Systems Analysis — paradigm-aware runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Canonical workshop commands (one per cell of the paper's 2x2 "
            "paradigm x method grid):\n"
            "  python runner.py --paradigm vanilla --method codi\n"
            "  python runner.py --paradigm vanilla --method coconut\n"
            "  python runner.py --paradigm simcot  --method codi\n"
            "  python runner.py --paradigm simcot  --method coconut\n\n"
            "Each command writes results/<paradigm>_<method>/<timestamp>/.\n"
            "Sim-CoT checkpoints bootstrap automatically on first use.\n\n"
            "Re-run analysis on cached trajectories (no GPU needed):\n"
            "  python runner.py --paradigm vanilla --method codi \\\n"
            "      --run_dir results/vanilla_codi/<timestamp>\n\n"
            "Perturbation flow (separate from the main pipeline):\n"
            "  python runner.py --paradigm simcot --method codi \\\n"
            "      --perturbation --run_dir results/simcot_codi/<timestamp>\n"
        ),
    )
    parser.add_argument(
        "--paradigm", choices=["vanilla", "simcot"], default=None,
        help="Training paradigm: 'vanilla' (HF pretrained) or 'simcot' "
             "(internlm release, bootstrapped automatically). When given "
             "together with --method, --config defaults to "
             "<paradigm>/configs/inference_<method>_gsm8k.yaml.",
    )
    parser.add_argument(
        "--method", choices=["codi", "coconut"], default=None,
        help="Latent CoT method.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config. If omitted, derived from --paradigm and "
             "--method. If --paradigm and --method are also omitted, "
             "falls back to the legacy config.yaml in the repo root.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Override config's dataset.n_samples (useful for smoke tests).",
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default=None,
        help="Override config's experiment.device. Use 'cpu' on machines "
             "without a GPU (e.g. evaluator laptops running smoke tests).",
    )
    parser.add_argument(
        "--run_dir", default=None,
        help="Existing run directory. Main pipeline: skip inference when its "
             "latent_states/all_states.h5 exists. Perturbation flow: the run "
             "whose trajectories serve as the clean baseline (defaults to "
             "the latest run for the paradigm/method).",
    )
    parser.add_argument(
        "--perturbation", action="store_true",
        help="Run the perturbation flow instead of the main pipeline. "
             "Consumes cached H5 trajectories; re-runs the model with noisy "
             "input embeddings; merges metrics into stability.h5.",
    )
    parser.add_argument(
        "--noise_std", type=float, default=None,
        help="Perturbation flow: Gaussian noise sigma on input embeddings "
             "(default from config's perturbation block, else 0.01).",
    )
    parser.add_argument(
        "--n_perturbations", type=int, default=None,
        help="Perturbation flow: noisy re-runs per question (default 3).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Perturbation flow: cap the number of questions (it re-runs "
             "the model n_perturbations times per question).",
    )
    args = parser.parse_args()

    # Resolve config path
    if args.config is None:
        if args.paradigm and args.method:
            args.config = _default_config(args.paradigm, args.method)
        else:
            args.config = "config.yaml"

    if args.perturbation:
        if not (args.paradigm and args.method):
            parser.error("--perturbation requires --paradigm and --method")
        config = load_config(args.config)
        if args.method is not None:
            config["method"]["name"] = args.method
        if args.device is not None:
            config.setdefault("experiment", {})["device"] = args.device
        run_dir = args.run_dir or _latest_run_dir(args.paradigm, args.method)
        if run_dir is None:
            parser.error(
                f"No cached run found under results/{args.paradigm}_{args.method}/ "
                f"— run the main pipeline first or pass --run_dir."
            )
        run_perturbation(
            config, run_dir,
            noise_std=args.noise_std,
            n_perturbations=args.n_perturbations,
            max_samples=args.max_samples,
        )
    else:
        run(
            config_path=args.config,
            paradigm=args.paradigm,
            method_override=args.method,
            n_samples_override=args.n_samples,
            device_override=args.device,
            run_dir=args.run_dir,
        )
