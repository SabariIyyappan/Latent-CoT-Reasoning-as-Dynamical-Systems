"""
Visualization for latent thought trajectories and dynamical features.

Plot conventions (per review feedback, rev. 2):
- Trajectory plots: scatter with viridis colorbar for latent CoT token index,
  filled circle = correct / open triangle = incorrect, class-centroid overlay.
- Metric plots: 2 averaged curves — correct (green) vs incorrect (red) —
  with std bands clipped at 0 (physical quantities cannot be negative),
  axis labels name the quantity explicitly and the total sample count.
- Arc length: histogram + boxplot + concept strip (sample-index spike chart
  was unreadable at N > ~200 and has been dropped).

All plots operate on outputs produced by data_prep / stability.
The same plot functions are called by ``runner.py`` during a fresh run and by
``scripts/replay/replot_from_cache.py`` when re-rendering from saved HDF5
artefacts.
"""

from pathlib import Path
from typing import Optional, List, Dict
import colorsys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ── Shared styling helpers ───────────────────────────────────────────────────

_NOTE_COLOR = "#555555"

def _annotate_n(ax, n_total: int, loc: str = "upper right") -> None:
    """Annotate total sample count on a metric plot."""
    ax.text(
        0.98 if "right" in loc else 0.02,
        0.02 if "lower" in loc else 0.98,
        f"N = {n_total:,} samples (aggregated)",
        transform=ax.transAxes,
        fontsize=9,
        color=_NOTE_COLOR,
        ha="right" if "right" in loc else "left",
        va="bottom" if "lower" in loc else "top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
    )


def _concept_footnote(fig, msg: str = None) -> None:
    """Tiny grey footnote under a concept-wise plot."""
    fig.text(
        0.5, -0.01,
        msg or ("Concept labels = keyword classifier over question text "
                "(see analysis/data_prep.py  |  docs/concept_bucketing.md). "
                "GSM8K has no native concept labels."),
        ha="center", va="top",
        fontsize=8, color=_NOTE_COLOR, style="italic",
    )


def _safe_fill_between(ax, x, mean, std, *, color, alpha=0.2, clip_at_zero=False):
    """fill_between that optionally clips the lower bound at zero (for
    non-negative physical quantities like distances)."""
    lower = mean - std
    if clip_at_zero:
        lower = np.maximum(lower, 0.0)
    ax.fill_between(x, lower, mean + std, color=color, alpha=alpha)


def _split_correct_incorrect(data: np.ndarray, correct_mask: np.ndarray):
    """Split data into correct/incorrect groups and compute means."""
    correct_idx = np.where(correct_mask)[0]
    wrong_idx = np.where(~correct_mask)[0]
    correct_data = data[correct_idx] if len(correct_idx) > 0 else None
    wrong_data = data[wrong_idx] if len(wrong_idx) > 0 else None
    correct_mean = np.nanmean(correct_data, axis=0) if correct_data is not None else None
    wrong_mean = np.nanmean(wrong_data, axis=0) if wrong_data is not None else None
    correct_std = np.nanstd(correct_data, axis=0) if correct_data is not None else None
    wrong_std = np.nanstd(wrong_data, axis=0) if wrong_data is not None else None
    return {
        "correct": {"data": correct_data, "mean": correct_mean, "std": correct_std, "n": len(correct_idx)},
        "wrong": {"data": wrong_data, "mean": wrong_mean, "std": wrong_std, "n": len(wrong_idx)},
    }


class Plotting:
    """
    Generates and saves trajectory and feature plots.

    Trajectory plots use scatter with viridis colorbar for latent step index.
    Metric plots show 2 averaged curves (correct vs incorrect) with std bands.
    """

    def __init__(self, output_dir: str, plot_format: str = "png", dpi: int = 300,
                 dataset_name: str = "gsm8k"):
        self.output_dir = Path(output_dir) / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_format = plot_format
        self.dpi = dpi
        self.dataset_name = dataset_name

    def _concept_footnote_msg(self) -> str:
        """Return the correct footnote for concept-wise plots based on the active dataset."""
        if self.dataset_name == "math":
            return (
                "Concept labels = subject field from EleutherAI/hendrycks_math dataset metadata "
                "(Algebra, Counting & Probability, Geometry, Intermediate Algebra, "
                "Number Theory, Prealgebra, Precalculus)."
            )
        return (
            "Concept labels = keyword classifier over question text "
            "(see analysis/data_prep.py  |  docs/concept_bucketing.md). "
            "GSM8K has no native concept labels."
        )

    def make_trajectory_plot(
        self,
        embeddings: np.ndarray,
        method_name: str,
        correct_mask: Optional[np.ndarray] = None,
        sample_labels: Optional[List[str]] = None,
        max_points: int = 50000,
        subtitle: Optional[str] = None,
        filename_suffix: str = "",
        output_subdir: Optional[str] = None,
    ) -> str:
        """
        2-D scatter of every latent state, coloured by CoT token index.

        Semantics of the visual encodings:
          color (viridis)  = latent CoT token index  t ∈ {0 .. T-1}
                             (0 = first token emitted by the latent chain,
                              T-1 = final token just before the decode head)
          marker           = correctness of the final answer
                             filled ●  = correct
                             hollow △  = incorrect
          dashed line      = centroid trajectory per class (mean point per t,
                             connected), overlaid so the "average path"
                             required by the review is unmissable

        Args:
            embeddings:     [N, T, 2]
            correct_mask:   [N] bool or None
            max_points:     random-subsample if N*T > this, purely for render
                            speed.  The centroid overlay still uses every
                            sample in the input arrays.
        """
        n_samples, n_steps, _ = embeddings.shape
        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))

        # Optionally down-sample for render speed (claim "n=8792" on plot
        # stays accurate because the centroids use all data).
        total_points = n_samples * n_steps
        if total_points > max_points:
            stride = int(np.ceil(total_points / max_points))
            keep = np.arange(n_samples)[::stride]
        else:
            keep = np.arange(n_samples)

        # Flatten for two scatter() calls (one per correctness class)
        emb_flat = embeddings[keep].reshape(-1, 2)
        step_flat = np.tile(np.arange(n_steps), len(keep))

        if correct_mask is not None:
            correct_flat = np.repeat(correct_mask[keep], n_steps)
        else:
            correct_flat = np.ones(len(emb_flat), dtype=bool)

        # (1) correct — filled circles
        m = correct_flat
        if m.any():
            sc = ax.scatter(
                emb_flat[m, 0], emb_flat[m, 1],
                c=step_flat[m], cmap="viridis",
                vmin=0, vmax=n_steps - 1,
                marker="o", s=24, edgecolors="none",
                alpha=0.55, zorder=2,
            )
        # (2) incorrect — open triangles with coloured edge (no fill)
        m = ~correct_flat
        if m.any():
            sc_w = ax.scatter(
                emb_flat[m, 0], emb_flat[m, 1],
                c=step_flat[m], cmap="viridis",
                vmin=0, vmax=n_steps - 1,
                marker="^", s=38, facecolors="none", linewidths=0.9,
                alpha=0.80, zorder=3,
            )
            sc = sc_w if not correct_flat.any() else sc

        # (3) Centroid trajectory per class — the "average path" required
        if correct_mask is not None:
            for label, mask_idx, col, ls in [
                ("Correct",   correct_mask,  "#118a4b", "-"),
                ("Incorrect", ~correct_mask, "#a02c1e", "--"),
            ]:
                if not mask_idx.any():
                    continue
                centroid = embeddings[mask_idx].mean(axis=0)   # [T, 2]
                ax.plot(
                    centroid[:, 0], centroid[:, 1],
                    ls, color=col, linewidth=3.0, marker="D",
                    markersize=9, markerfacecolor="white",
                    markeredgewidth=2.0, alpha=0.95, zorder=6,
                    label=f"{label} centroid (n={int(mask_idx.sum()):,})",
                )
                # Step index annotation at each centroid knot
                for t in range(n_steps):
                    ax.annotate(
                        f"{t}",
                        (centroid[t, 0], centroid[t, 1]),
                        textcoords="offset points",
                        xytext=(6, 6), fontsize=9,
                        color=col, fontweight="bold", zorder=7,
                    )

        ax.set_xlabel(f"{method_name.upper()} Dim 1", fontsize=12)
        ax.set_ylabel(f"{method_name.upper()} Dim 2", fontsize=12)
        title = (
            f"Latent Thought Trajectories — {method_name.upper()}"
            f"  |  N = {n_samples:,} samples × T = {n_steps} tokens"
        )
        if subtitle:
            title = f"{title}\n{subtitle}"
        ax.set_title(title, fontsize=13)

        # Colorbar with explicit semantic label
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(
            "Latent CoT token index  t  (0 = first, T-1 = final)",
            fontsize=11,
        )
        cbar.set_ticks(range(n_steps))
        cbar.set_ticklabels([f"t={t}" for t in range(n_steps)])

        # Class legend
        if correct_mask is not None:
            correctness_handles = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#666", markeredgecolor="#666",
                       markersize=9, label="● Correct (filled, colour=t)"),
                Line2D([0], [0], marker="^", color="w",
                       markerfacecolor="none", markeredgecolor="#666",
                       markeredgewidth=1.4, markersize=10,
                       label="△ Incorrect (open, colour=t)"),
                Line2D([0], [0], color="#118a4b", linewidth=3, linestyle="-",
                       label="Correct centroid path"),
                Line2D([0], [0], color="#a02c1e", linewidth=3, linestyle="--",
                       label="Incorrect centroid path"),
            ]
            ax.legend(handles=correctness_handles, loc="best", fontsize=10,
                      framealpha=0.9)

        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        return self._save_fig(
            fig,
            f"trajectory_{method_name}{filename_suffix}",
            subdir=output_subdir,
        )

    # ─── 3-D trajectory plot (DMD / PHATE only) ──────────────────────────────

    def make_3d_trajectory_plot(
        self,
        embeddings: np.ndarray,
        method_name: str,
        correct_mask: Optional[np.ndarray] = None,
        max_points: int = 50000,
        subtitle: Optional[str] = None,
        filename_suffix: str = "",
        output_subdir: Optional[str] = None,
        elev: float = 22.0,
        azim: float = -60.0,
    ) -> str:
        """3-D scatter mirror of ``make_trajectory_plot``.

        Visual encodings match the 2-D version exactly:
          color (viridis) = latent CoT token index t in {0..T-1}
          marker          = correctness  (filled o = correct, hollow ^ = incorrect)
          centroid path   = mean trajectory per class, in 3-D, with step labels

        Args:
            embeddings: [N, T, 3]
            elev, azim: matplotlib 3D view angles for the saved figure
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)

        n_samples, n_steps, dim = embeddings.shape
        if dim < 3:
            raise ValueError(
                f"make_3d_trajectory_plot needs embeddings with 3 components "
                f"on the last axis, got {dim}"
            )
        emb3 = embeddings[..., :3]

        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")

        # Down-sample if needed
        total_points = n_samples * n_steps
        if total_points > max_points:
            stride = int(np.ceil(total_points / max_points))
            keep = np.arange(n_samples)[::stride]
        else:
            keep = np.arange(n_samples)

        emb_flat = emb3[keep].reshape(-1, 3)
        step_flat = np.tile(np.arange(n_steps), len(keep))

        if correct_mask is not None:
            correct_flat = np.repeat(correct_mask[keep], n_steps)
        else:
            correct_flat = np.ones(len(emb_flat), dtype=bool)

        sc = None
        m = correct_flat
        if m.any():
            sc = ax.scatter(
                emb_flat[m, 0], emb_flat[m, 1], emb_flat[m, 2],
                c=step_flat[m], cmap="viridis",
                vmin=0, vmax=n_steps - 1,
                marker="o", s=18, edgecolors="none",
                alpha=0.55, depthshade=True,
            )
        m = ~correct_flat
        if m.any():
            sc_w = ax.scatter(
                emb_flat[m, 0], emb_flat[m, 1], emb_flat[m, 2],
                c=step_flat[m], cmap="viridis",
                vmin=0, vmax=n_steps - 1,
                marker="^", s=28, facecolors="none", linewidths=0.9,
                alpha=0.80, depthshade=True,
            )
            sc = sc_w if sc is None else sc

        # Centroid trajectories per class — drawn through 3-D
        if correct_mask is not None:
            for label, mask_idx, col, ls in [
                ("Correct",   correct_mask,  "#118a4b", "-"),
                ("Incorrect", ~correct_mask, "#a02c1e", "--"),
            ]:
                if not mask_idx.any():
                    continue
                centroid = emb3[mask_idx].mean(axis=0)  # [T, 3]
                ax.plot(
                    centroid[:, 0], centroid[:, 1], centroid[:, 2],
                    ls, color=col, linewidth=3.0, marker="D",
                    markersize=8, markerfacecolor="white",
                    markeredgewidth=2.0, alpha=0.95,
                    label=f"{label} centroid (n={int(mask_idx.sum()):,})",
                )
                for t in range(n_steps):
                    ax.text(
                        centroid[t, 0], centroid[t, 1], centroid[t, 2],
                        f"{t}", fontsize=8, color=col, fontweight="bold",
                    )

        ax.set_xlabel(f"{method_name.upper()} Dim 1", fontsize=11)
        ax.set_ylabel(f"{method_name.upper()} Dim 2", fontsize=11)
        ax.set_zlabel(f"{method_name.upper()} Dim 3", fontsize=11)
        title = (
            f"Latent Thought Trajectories — {method_name.upper()} (3-D)"
            f"  |  N = {n_samples:,} samples × T = {n_steps} tokens"
        )
        if subtitle:
            title = f"{title}\n{subtitle}"
        ax.set_title(title, fontsize=12)
        ax.view_init(elev=elev, azim=azim)

        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, pad=0.10, shrink=0.7)
            cbar.set_label(
                "Latent CoT token index  t  (0 = first, T-1 = final)",
                fontsize=10,
            )
            cbar.set_ticks(range(n_steps))
            cbar.set_ticklabels([f"t={t}" for t in range(n_steps)])

        if correct_mask is not None:
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
            ax.legend(handles=handles, loc="upper left", fontsize=9,
                      framealpha=0.9)

        plt.tight_layout()
        return self._save_fig(
            fig,
            f"trajectory_{method_name}_3d{filename_suffix}",
            subdir=output_subdir,
        )

    # ─── Per-concept wrappers ────────────────────────────────────────────────

    def make_per_concept_trajectory_plots(
        self,
        embeddings: np.ndarray,
        method_name: str,
        concept_labels: np.ndarray,
        concept_names: List[str],
        correct_mask: Optional[np.ndarray] = None,
        min_samples: int = 5,
        output_subdir: str = "per_concept",
    ) -> List[str]:
        """Emit one trajectory figure per concept bucket using the same
        viridis-by-CoT-step semantics as make_trajectory_plot. Combined
        figures are still produced by the caller; this wrapper adds
        per-concept views alongside.

        Concepts with fewer than ``min_samples`` samples are skipped because
        the centroid overlay and step colorbar become uninformative below
        roughly five samples. Output filenames look like
        ``trajectory_<method>_concept<cid>_<slug>.<ext>`` and land under
        ``<plots_dir>/<output_subdir>/``.

        Returns the list of saved file paths (skipped concepts contribute no
        path).
        """
        saved: List[str] = []
        n_concepts = len(concept_names) if concept_names else (
            int(concept_labels.max()) + 1 if len(concept_labels) > 0 else 0
        )
        for cid in range(n_concepts):
            mask = concept_labels == cid
            n = int(mask.sum())
            if n < min_samples:
                name = concept_names[cid] if concept_names else f"concept{cid}"
                print(
                    f"  per-concept {cid} ({name}): only {n} samples, skipping"
                )
                continue
            emb_c = embeddings[mask]
            cm_c = correct_mask[mask] if correct_mask is not None else None
            cname = concept_names[cid] if concept_names else f"concept{cid}"
            slug = (
                cname.lower()
                .replace(" & ", "_and_")
                .replace(" ", "_")
                .replace("/", "_")
                .replace("&", "and")
            )
            n_correct = int(cm_c.sum()) if cm_c is not None else 0
            sub = (
                f"Concept: {cname}  |  N = {n:,}  |  "
                f"correct = {n_correct} ({(n_correct/max(n,1))*100:.1f}%)"
            )
            path = self.make_trajectory_plot(
                emb_c,
                method_name,
                correct_mask=cm_c,
                subtitle=sub,
                filename_suffix=f"_concept{cid}_{slug}",
                output_subdir=output_subdir,
            )
            saved.append(path)
        return saved

    def make_per_concept_3d_trajectory_plots(
        self,
        embeddings: np.ndarray,
        method_name: str,
        concept_labels: np.ndarray,
        concept_names: List[str],
        correct_mask: Optional[np.ndarray] = None,
        min_samples: int = 5,
        output_subdir: str = "per_concept",
    ) -> List[str]:
        """3-D counterpart to ``make_per_concept_trajectory_plots``.

        Concepts with fewer than ``min_samples`` samples are skipped. Output
        filenames look like ``trajectory_<method>_3d_concept<cid>_<slug>.<ext>``.
        """
        saved: List[str] = []
        n_concepts = len(concept_names) if concept_names else (
            int(concept_labels.max()) + 1 if len(concept_labels) > 0 else 0
        )
        for cid in range(n_concepts):
            mask = concept_labels == cid
            n = int(mask.sum())
            if n < min_samples:
                cname = concept_names[cid] if concept_names else f"concept{cid}"
                print(
                    f"  per-concept-3d {cid} ({cname}): only {n} samples, skipping"
                )
                continue
            emb_c = embeddings[mask]
            cm_c = correct_mask[mask] if correct_mask is not None else None
            cname = concept_names[cid] if concept_names else f"concept{cid}"
            slug = (
                cname.lower()
                .replace(" & ", "_and_")
                .replace(" ", "_")
                .replace("/", "_")
                .replace("&", "and")
            )
            n_correct = int(cm_c.sum()) if cm_c is not None else 0
            sub = (
                f"Concept: {cname}  |  N = {n:,}  |  "
                f"correct = {n_correct} ({(n_correct/max(n,1))*100:.1f}%)"
            )
            path = self.make_3d_trajectory_plot(
                emb_c,
                method_name,
                correct_mask=cm_c,
                subtitle=sub,
                filename_suffix=f"_concept{cid}_{slug}",
                output_subdir=output_subdir,
            )
            saved.append(path)
        return saved

    # ─── Helper for metric plots with correct/incorrect averaged curves ──────

    def _plot_metric_with_avg(
        self,
        ax,
        data: np.ndarray,
        steps: np.ndarray,
        correct_mask: Optional[np.ndarray],
        ylabel: str,
        title: str,
        show_individual: bool = False,
        clip_at_zero: bool = False,
    ):
        """
        Plot metric data with 2 averaged curves (correct vs incorrect).

        Args:
            data: [n_samples, n_values] metric data
            steps: [n_values] x-axis values
            correct_mask: [n_samples] bool
            clip_at_zero: clip the lower edge of the std band at 0, for
                non-negative physical quantities (distances, divergences).
                This prevents the visual illusion of a "wider" band when
                mean - std would drop into physically impossible negatives.
        """
        n_total = int(data.shape[0])

        if correct_mask is not None:
            groups = _split_correct_incorrect(data, correct_mask)

            # Faded individual lines in background (disabled by default)
            if show_individual:
                for i in range(data.shape[0]):
                    color = "#2ecc71" if correct_mask[i] else "#e74c3c"
                    ax.plot(steps, data[i], color=color, alpha=0.15, linewidth=0.8)

            # Averaged correct curve
            if groups["correct"]["mean"] is not None:
                m = groups["correct"]["mean"]
                s = groups["correct"]["std"]
                ax.plot(steps, m, "o-", color="#2ecc71", linewidth=2.5, markersize=7,
                        label=f"Correct (n={groups['correct']['n']:,})", zorder=5)
                _safe_fill_between(ax, steps, m, s, color="#2ecc71",
                                   alpha=0.22, clip_at_zero=clip_at_zero)

            # Averaged incorrect curve
            if groups["wrong"]["mean"] is not None:
                m = groups["wrong"]["mean"]
                s = groups["wrong"]["std"]
                ax.plot(steps, m, "x--", color="#e74c3c", linewidth=2.5, markersize=7,
                        label=f"Incorrect (n={groups['wrong']['n']:,})", zorder=5)
                _safe_fill_between(ax, steps, m, s, color="#e74c3c",
                                   alpha=0.22, clip_at_zero=clip_at_zero)
        else:
            # No correctness info — just plot mean
            ax.plot(steps, data.mean(axis=0), "k-s", linewidth=2.5, markersize=8,
                    label="Mean", zorder=5)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        _annotate_n(ax, n_total, loc="lower right")

    # ─── Metric plots ───────────────────────────────────────────────────────

    def make_step_change_plot(
        self,
        step_changes: np.ndarray,
        correct_mask: Optional[np.ndarray] = None,
    ) -> str:
        """Step-to-step change: 2 averaged curves (correct vs incorrect).

        X-axis = latent transition index k ∈ {1..T-1}, where k = t→t+1.
        ``step_changes`` shape: [N_samples, T-1].  N is annotated on-plot.
        """
        n_samples, n_transitions = step_changes.shape
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        steps = np.arange(1, n_transitions + 1)

        self._plot_metric_with_avg(
            ax, step_changes, steps, correct_mask,
            ylabel=r"$\|z_{t+1} - z_t\|_2$",
            title="Step-to-Step Change Size — latent-chain velocity",
            clip_at_zero=True,
        )
        ax.set_xlabel(
            f"Latent transition index  k  (k = t→t+1, k ∈ 1..{n_transitions})",
            fontsize=12,
        )
        ax.set_xticks(steps)

        plt.tight_layout()
        return self._save_fig(fig, "step2step_change")

    def make_direction_consistency_plot(
        self,
        dir_consistency: np.ndarray,
        correct_mask: Optional[np.ndarray] = None,
    ) -> str:
        """Direction consistency: cos(Δz_t, Δz_{t-1}) — persistence of motion.

        X-axis = t ∈ {2..T-1}; needs two adjacent Δs, so starts at 2.
        cos = +1: motion persists; 0: orthogonal pivot; −1: reversal/oscillation.
        """
        n_samples, n_pairs = dir_consistency.shape
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        steps = np.arange(2, n_pairs + 2)

        self._plot_metric_with_avg(
            ax, dir_consistency, steps, correct_mask,
            ylabel=r"$\cos(\Delta z_t, \Delta z_{t-1})$",
            title="Direction Consistency — persistence of motion"
                  "\n(+1: same direction  |  0: orthogonal pivot  |  −1: reversal)",
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(
            f"Latent step t  (t ∈ 2..{n_pairs + 1}; needs two adjacent Δs)",
            fontsize=12,
        )
        ax.set_xticks(steps)
        ax.set_ylim(-1.1, 1.1)

        plt.tight_layout()
        return self._save_fig(fig, "direction_consistency")

    def make_arc_length_plot(
        self,
        arc_lengths: np.ndarray,
        correct_mask: Optional[np.ndarray] = None,
        concept_labels: Optional[np.ndarray] = None,
        concept_names: Optional[List[str]] = None,
    ) -> str:
        """
        Arc length = Σ_t ‖z_{t+1} − z_t‖₂ (total path traversed in latent space).

        Three panels replace the old per-sample spike chart (unreadable at
        N>~200 — the spikes overlap into an opaque wall):

          (a) Histogram of arc length, correct vs incorrect overlay
          (b) Box-and-whisker by correctness (median, IQR, whiskers, outliers)
          (c) Per-concept strip of arc length, correctness-coloured
        """
        n_samples = int(arc_lengths.shape[0])

        has_concepts = concept_labels is not None and len(concept_labels) == n_samples
        ncols = 3 if has_concepts else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.5))
        ax_hist, ax_box = axes[0], axes[1]
        ax_strip = axes[2] if has_concepts else None

        # ── (a) Histogram ──────────────────────────────────────────────────
        if correct_mask is not None:
            bins = np.linspace(float(arc_lengths.min()), float(arc_lengths.max()), 60)
            ax_hist.hist(arc_lengths[correct_mask],  bins=bins, alpha=0.55,
                         color="#2ecc71", label=f"Correct (n={int(correct_mask.sum()):,})")
            ax_hist.hist(arc_lengths[~correct_mask], bins=bins, alpha=0.55,
                         color="#e74c3c", label=f"Incorrect (n={int((~correct_mask).sum()):,})")
            # annotate means
            mc = float(arc_lengths[correct_mask].mean())
            mi = float(arc_lengths[~correct_mask].mean())
            ax_hist.axvline(mc, color="#118a4b", linestyle="--", linewidth=1.5,
                            label=f"mean(correct)   = {mc:.1f}")
            ax_hist.axvline(mi, color="#a02c1e", linestyle="--", linewidth=1.5,
                            label=f"mean(incorrect) = {mi:.1f}")
        else:
            ax_hist.hist(arc_lengths, bins=60, color="#3498db", alpha=0.8)

        ax_hist.set_xlabel(r"Arc Length  $\sum_t \|z_{t+1}-z_t\|_2$", fontsize=11)
        ax_hist.set_ylabel("Sample count", fontsize=11)
        ax_hist.set_title("(a) Arc-length distribution", fontsize=12)
        ax_hist.legend(fontsize=10, loc="upper left")
        ax_hist.grid(True, alpha=0.3)
        _annotate_n(ax_hist, n_samples, loc="upper right")

        # ── (b) Box plot ──────────────────────────────────────────────────
        if correct_mask is not None:
            bp_data = [arc_lengths[correct_mask], arc_lengths[~correct_mask]]
            bp_labels = [f"Correct\n(n={int(correct_mask.sum()):,})",
                         f"Incorrect\n(n={int((~correct_mask).sum()):,})"]
            bp = ax_box.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                                widths=0.55, showfliers=True,
                                flierprops=dict(marker=".", markersize=2,
                                                markerfacecolor="#888",
                                                markeredgecolor="#888", alpha=0.5))
            for patch, color in zip(bp["boxes"], ["#2ecc71", "#e74c3c"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(2)
        else:
            ax_box.boxplot([arc_lengths], labels=["All samples"])

        ax_box.set_ylabel(r"Arc Length  $\sum_t \|z_{t+1}-z_t\|_2$", fontsize=11)
        ax_box.set_title("(b) Arc length by correctness", fontsize=12)
        ax_box.grid(True, axis="y", alpha=0.3)

        # ── (c) Per-concept strip ─────────────────────────────────────────
        if has_concepts:
            n_concepts = 7
            names = (concept_names
                     if concept_names is not None
                     else [f"C{i}" for i in range(n_concepts)])
            rng = np.random.default_rng(0)
            for cid in range(n_concepts):
                mcon = concept_labels == cid
                if not mcon.any():
                    continue
                xs = cid + rng.uniform(-0.32, 0.32, size=int(mcon.sum()))
                ys = arc_lengths[mcon]
                if correct_mask is not None:
                    ok = correct_mask[mcon]
                    ax_strip.scatter(xs[ok],  ys[ok],  s=3, color="#2ecc71", alpha=0.35)
                    ax_strip.scatter(xs[~ok], ys[~ok], s=3, color="#e74c3c", alpha=0.35)
                else:
                    ax_strip.scatter(xs, ys, s=3, color="#3498db", alpha=0.4)
                ax_strip.errorbar(
                    cid, float(ys.mean()), yerr=float(ys.std()),
                    fmt="o", color="black", markerfacecolor="white",
                    markersize=8, elinewidth=1.5, zorder=5,
                )
            ax_strip.set_xticks(range(n_concepts))
            ax_strip.set_xticklabels([n[:14] for n in names], rotation=35,
                                     ha="right", fontsize=9)
            ax_strip.set_ylabel(r"Arc Length", fontsize=11)
            ax_strip.set_title("(c) Per-concept (strip + mean±1σ)", fontsize=12)
            ax_strip.grid(True, axis="y", alpha=0.3)

        fig.suptitle(
            "Path Complexity (Arc Length) — total latent-chain distance",
            fontsize=14, y=1.02,
        )
        plt.tight_layout()
        return self._save_fig(fig, "arc_length")

    def make_lyapunov_plot(
        self,
        local_lyapunov: np.ndarray,
        correct_mask: Optional[np.ndarray] = None,
    ) -> str:
        """Local Lyapunov: log(‖Δz_{t+1}‖/‖Δz_t‖), diverging vs converging.

        X-axis = t ∈ {2..T-1} (needs adjacent Δ magnitudes).
        """
        n_samples, n_pairs = local_lyapunov.shape
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        steps = np.arange(2, n_pairs + 2)

        self._plot_metric_with_avg(
            ax, local_lyapunov, steps, correct_mask,
            ylabel=r"$\log_{10}(\|\Delta z_{t+1}\| / \|\Delta z_t\|)$",
            title="Local Lyapunov Sensitivity — trajectory stability per step"
                  "\n(> 0: diverging  |  < 0: converging  |  0: neutral)",
        )
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.6)
        ax.set_xlabel(
            f"Latent step t  (t ∈ 2..{n_pairs + 1}; ratio uses ‖Δz_t‖ and ‖Δz_{{t+1}}‖)",
            fontsize=12,
        )
        ax.set_xticks(steps)

        plt.tight_layout()
        return self._save_fig(fig, "lyapunov_sensitivity")

    def make_fixed_point_plot(
        self,
        distances_lag1: np.ndarray,
        distances_lag2: np.ndarray,
        correct_mask: Optional[np.ndarray] = None,
    ) -> str:
        """Fixed-point analysis: ‖z_{t+k} − z_t‖ at lag k ∈ {1, 2}.

        If the chain converges to a fixed point the curves decrease; flat
        curves imply ongoing motion / oscillation.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, dist, lag in [(axes[0], distances_lag1, 1), (axes[1], distances_lag2, 2)]:
            n_samples, n_steps = dist.shape
            steps = np.arange(n_steps)

            self._plot_metric_with_avg(
                ax, dist, steps, correct_mask,
                ylabel=fr"$\|z_{{t+{lag}}} - z_t\|_2$",
                title=f"Lag = {lag}  (compares z_t and z_{{t+{lag}}})",
                clip_at_zero=True,
            )
            ax.set_xlabel(
                f"Latent step t  (t ∈ 0..{n_steps - 1})",
                fontsize=12,
            )
            ax.set_xticks(steps)

        plt.suptitle(
            "Fixed-Point Analysis — distance at lag 1 vs lag 2"
            "\n(monotonic decrease → approaching fixed point; flat → orbiting)",
            fontsize=13,
        )
        plt.tight_layout()
        return self._save_fig(fig, "fixed_point_analysis")

    def make_perturbation_divergence_plot(
        self,
        divergence: np.ndarray,
        divergence_std: Optional[np.ndarray] = None,
        correct_mask: Optional[np.ndarray] = None,
        relative_divergence: Optional[np.ndarray] = None,
        noise_std: float = 0.01,
        n_perturbations: int = 3,
        filename_suffix: str = "",
    ) -> str:
        """Perturbation stability diagnostic (three panels).

        (a) Absolute divergence    ‖z_t^clean − z_t^perturbed‖₂, std band
            clipped at 0 (distances cannot be negative — removes the visual
            illusion that correct has wider variance than incorrect).
        (b) Log-y absolute divergence — makes the incorrect-higher ordering
            visible at every step instead of just the last.
        (c) Relative divergence    ‖Δ‖ / ‖z_t^clean‖ — scale-free version.

        Args:
            noise_std, n_perturbations: shown in the figure title for context.
            filename_suffix: appended to the saved filename so multiple runs
                at different sigmas don't overwrite each other (e.g. pass
                "_sigma1.0" to write "perturbation_divergence_sigma1.0.png").
        """
        n_samples, n_steps = divergence.shape
        steps = np.arange(n_steps)
        has_rel = relative_divergence is not None
        ncols = 3 if has_rel else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.5))

        # (a) Absolute, linear-y, clipped band
        self._plot_metric_with_avg(
            axes[0], divergence, steps, correct_mask,
            ylabel=r"$\|z_t^{perturbed} - z_t^{clean}\|_2$",
            title="(a) Absolute divergence  (linear y, σ-band clipped at 0)",
            clip_at_zero=True,
        )
        axes[0].set_xlabel(f"Latent step t  (t ∈ 0..{n_steps - 1})", fontsize=11)
        axes[0].set_xticks(steps)

        # (b) Absolute, log-y — same data, reveals ordering
        self._plot_metric_with_avg(
            axes[1], np.clip(divergence, 1e-6, None), steps, correct_mask,
            ylabel=r"$\|z_t^{perturbed} - z_t^{clean}\|_2$  (log scale)",
            title="(b) Absolute divergence — log y",
            clip_at_zero=True,
        )
        axes[1].set_yscale("log")
        axes[1].set_xlabel(f"Latent step t  (t ∈ 0..{n_steps - 1})", fontsize=11)
        axes[1].set_xticks(steps)

        # (c) Relative divergence (if supplied)
        if has_rel:
            self._plot_metric_with_avg(
                axes[2], relative_divergence, steps, correct_mask,
                ylabel=r"$\|z_t^{perturbed} - z_t^{clean}\|_2 / \|z_t^{clean}\|_2$",
                title="(c) Relative divergence — scale-free",
                clip_at_zero=True,
            )
            axes[2].set_xlabel(f"Latent step t  (t ∈ 0..{n_steps - 1})", fontsize=11)
            axes[2].set_xticks(steps)

        fig.suptitle(
            "Perturbation Stability — sensitivity to Gaussian noise in initial latent"
            f"\n(sigma = {noise_std}, n_perturbations = {n_perturbations};  "
            "growing = sensitive, flat = robust)",
            fontsize=13, y=1.04,
        )
        plt.tight_layout()
        return self._save_fig(fig, f"perturbation_divergence{filename_suffix}")

    def make_dmd_spectrum_aggregate_plot(
        self,
        dmd_spec: Dict[str, np.ndarray],
        correct_mask: Optional[np.ndarray] = None,
    ) -> str:
        """
        Aggregate DMD spectrum view for large N — the per-sample complex-plane
        scatter becomes a solid cloud at N > ~200.

        Three panels:
          (a) 2-D histogram (density) of eigenvalues in the complex plane,
              with the unit circle overlaid.
          (b) Histogram of max |λ| per sample, correct vs incorrect.
          (c) Stacked bar of n_unstable_modes (|λ|>1 count per sample),
              correct vs incorrect.
        """
        e_real = dmd_spec["eigenvalues_real"]   # [N, K]
        e_imag = dmd_spec["eigenvalues_imag"]   # [N, K]
        max_abs = dmd_spec["max_abs_eig"]       # [N]
        n_unst = dmd_spec["n_unstable_modes"].astype(int)  # [N]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # (a) density in complex plane
        re_flat = e_real.ravel()
        im_flat = e_imag.ravel()
        valid = ~np.isnan(re_flat) & ~np.isnan(im_flat)
        h = axes[0].hist2d(
            re_flat[valid], im_flat[valid],
            bins=120, cmap="magma", cmin=1,
        )
        theta = np.linspace(0, 2 * np.pi, 200)
        axes[0].plot(np.cos(theta), np.sin(theta), "w--", linewidth=1.5,
                     alpha=0.9, label="Unit circle (|λ|=1)")
        axes[0].axhline(0, color="w", linewidth=0.5, alpha=0.5)
        axes[0].axvline(0, color="w", linewidth=0.5, alpha=0.5)
        axes[0].set_xlabel(r"Re($\lambda$)", fontsize=12)
        axes[0].set_ylabel(r"Im($\lambda$)", fontsize=12)
        axes[0].set_title(
            f"(a) Eigenvalue density  (all N × K eigenvalues)"
            f"\nN = {e_real.shape[0]:,}, K = {e_real.shape[1]}",
            fontsize=12,
        )
        axes[0].set_aspect("equal")
        axes[0].legend(loc="upper left", fontsize=9, framealpha=0.85)
        fig.colorbar(h[3], ax=axes[0], fraction=0.045, pad=0.02,
                     label="density (count)")

        # (b) Max |λ| histogram
        if correct_mask is not None:
            bins = np.linspace(0, max(1.6, float(max_abs.max())), 80)
            axes[1].hist(max_abs[correct_mask],  bins=bins, alpha=0.6,
                         color="#2ecc71", label=f"Correct (n={int(correct_mask.sum()):,})")
            axes[1].hist(max_abs[~correct_mask], bins=bins, alpha=0.6,
                         color="#e74c3c", label=f"Incorrect (n={int((~correct_mask).sum()):,})")
        else:
            axes[1].hist(max_abs, bins=80, color="#3498db", alpha=0.8)
        axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1.5,
                        label="|λ| = 1 (stability boundary)")
        axes[1].set_xlabel(r"max $|\lambda|$  per sample", fontsize=11)
        axes[1].set_ylabel("Sample count", fontsize=11)
        axes[1].set_title("(b) Spectral radius distribution", fontsize=12)
        axes[1].legend(fontsize=10, loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # (c) Count of unstable modes
        max_k = int(n_unst.max()) + 1
        bins_u = np.arange(-0.5, max_k + 0.5, 1.0)
        if correct_mask is not None:
            axes[2].hist(n_unst[correct_mask], bins=bins_u, alpha=0.6,
                         color="#2ecc71",
                         label=f"Correct (n={int(correct_mask.sum()):,})")
            axes[2].hist(n_unst[~correct_mask], bins=bins_u, alpha=0.6,
                         color="#e74c3c",
                         label=f"Incorrect (n={int((~correct_mask).sum()):,})")
        else:
            axes[2].hist(n_unst, bins=bins_u, color="#3498db", alpha=0.8)
        axes[2].set_xticks(range(max_k))
        axes[2].set_xlabel(r"# modes with $|\lambda| > 1$  per sample", fontsize=11)
        axes[2].set_ylabel("Sample count", fontsize=11)
        axes[2].set_title("(c) Unstable-mode count", fontsize=12)
        axes[2].legend(fontsize=10, loc="upper right")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            "DMD Spectrum — aggregate view (density instead of scatter)",
            fontsize=14, y=1.02,
        )
        plt.tight_layout()
        return self._save_fig(fig, "dmd_spectrum")

    def make_dmd_spectrum_plot(
        self,
        eigenvalues_real: np.ndarray,
        eigenvalues_imag: np.ndarray,
        correct_mask: Optional[np.ndarray] = None,
    ) -> str:
        """DMD eigenvalues in complex plane with unit circle."""
        n_samples = eigenvalues_real.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.6, label="Unit circle")

        # Plot correct and wrong samples with different markers
        for i in range(n_samples):
            is_correct = correct_mask is None or correct_mask[i]
            marker = "o" if is_correct else "X"
            color = "#2ecc71" if is_correct else "#e74c3c"
            label = f"Sample {i} ({'correct' if is_correct else 'wrong'})"

            valid = ~np.isnan(eigenvalues_real[i])
            ax.scatter(
                eigenvalues_real[i][valid],
                eigenvalues_imag[i][valid],
                c=color, marker=marker, s=120,
                edgecolors="black", linewidths=0.8,
                label=label, alpha=0.85, zorder=3,
            )

        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5)
        ax.set_xlabel(r"Re($\lambda$)", fontsize=12)
        ax.set_ylabel(r"Im($\lambda$)", fontsize=12)
        ax.set_title("DMD Eigenvalues  (inside unit circle = stable)", fontsize=13)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save_fig(fig, "dmd_spectrum")

    # ─── Concept-wise helpers ────────────────────────────────────────────────

    @staticmethod
    def _lighten_color(rgb, amount: float = 0.45):
        """Return a lighter version of an RGB tuple by shifting lightness."""
        h, l, s = colorsys.rgb_to_hls(*rgb[:3])
        return colorsys.hls_to_rgb(h, min(1.0, l + amount * (1.0 - l)), s)

    # ─── Concept-wise trajectory plot ────────────────────────────────────────

    def make_concept_trajectory_plot(
        self,
        embeddings: np.ndarray,
        method_name: str,
        concept_labels: np.ndarray,
        correct_mask: np.ndarray,
        concept_names: Optional[List[str]] = None,
    ) -> str:
        """
        2D scatter of latent trajectories coloured by math concept.

        Each sample's steps are connected by a faint line in the concept colour.
        Correct samples: filled circles at full opacity.
        Incorrect samples: open circles (light fill, coloured edge) at reduced opacity.
        7 concept colours come from tab10; dark/light shade encodes correctness.

        Args:
            embeddings:     [n_samples, n_steps, 2]
            method_name:    reduction name, e.g. "tsne"
            concept_labels: [n_samples] int in [0, n_concepts)
            correct_mask:   [n_samples] bool
            concept_names:  optional list of label strings (length = n_concepts)
        """
        n_samples, n_steps, _ = embeddings.shape
        n_concepts = int(concept_labels.max()) + 1 if len(concept_labels) > 0 else 7
        cmap = plt.get_cmap("tab10")
        base_colors = np.asarray([cmap(i) for i in range(n_concepts)])  # [C, 4]

        fig, ax = plt.subplots(1, 1, figsize=(11, 9))

        # Vectorised scatter: flatten (N, T, 2) -> (N*T, 2) and build per-point
        # colour / alpha / marker arrays.  Previous per-sample plt.plot loop was
        # O(N) python calls ⇒ 4 min per figure at N = 8,792.  This takes ~3 s.
        flat_xy   = embeddings.reshape(-1, 2)                               # [N*T, 2]
        flat_cid  = np.repeat(concept_labels.astype(int), n_steps)          # [N*T]
        flat_ok   = np.repeat(correct_mask.astype(bool), n_steps)           # [N*T]
        flat_rgba = base_colors[flat_cid].copy()                            # [N*T, 4]
        # Correctness encoding: correct = solid, incorrect = lightened + alpha
        if (~flat_ok).any():
            # Lighten RGB, keep alpha separate
            dark = flat_rgba[~flat_ok, :3]
            h_l_s = np.array([colorsys.rgb_to_hls(*r) for r in dark])
            h_l_s[:, 1] = np.minimum(1.0, h_l_s[:, 1] + 0.45 * (1.0 - h_l_s[:, 1]))
            light = np.array([colorsys.hls_to_rgb(*hls) for hls in h_l_s])
            flat_rgba[~flat_ok, :3] = light

        # alpha channel: correct = 0.80, incorrect = 0.55
        flat_rgba[:, 3] = np.where(flat_ok, 0.80, 0.55)

        # Single scatter call for correct (filled) and one for incorrect (open)
        m = flat_ok
        if m.any():
            ax.scatter(flat_xy[m, 0], flat_xy[m, 1],
                       c=flat_rgba[m],
                       edgecolors=(0, 0, 0, 0.6), linewidths=0.4,
                       marker="o", s=45, zorder=2)
        m = ~flat_ok
        if m.any():
            # Use dark edge (original concept colour) to convey "incorrect"
            dark_edges = base_colors[flat_cid[m]]
            ax.scatter(flat_xy[m, 0], flat_xy[m, 1],
                       facecolors=flat_rgba[m],
                       edgecolors=dark_edges, linewidths=1.0,
                       marker="o", s=45, zorder=3)

        # ── Legend ─────────────────────────────────────────────────────────
        concept_patches = [
            mpatches.Patch(
                facecolor=base_colors[cid],
                edgecolor="black",
                linewidth=0.5,
                label=concept_names[cid] if concept_names else f"Concept {cid}",
            )
            for cid in range(n_concepts)
        ]
        correctness_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#555555", markeredgecolor="black",
                   markeredgewidth=0.5, markersize=9, label="Correct (filled)"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#dddddd", markeredgecolor="#555555",
                   markeredgewidth=1.0, markersize=9, label="Incorrect (open)"),
        ]
        legend1 = ax.legend(
            handles=concept_patches,
            title="Math Concept", title_fontsize=10,
            fontsize=9, loc="upper left",
            framealpha=0.85,
        )
        ax.add_artist(legend1)
        ax.legend(
            handles=correctness_handles,
            title="Correctness", title_fontsize=10,
            fontsize=9, loc="lower right",
            framealpha=0.85,
        )

        ax.set_xlabel(f"{method_name.upper()} Dim 1", fontsize=12)
        ax.set_ylabel(f"{method_name.upper()} Dim 2", fontsize=12)
        ax.set_title(
            f"Latent Trajectories — {method_name.upper()}  "
            f"|  7 Math Concepts  |  N = {n_samples:,}",
            fontsize=14,
        )
        ax.grid(True, alpha=0.2)
        _concept_footnote(fig, self._concept_footnote_msg())
        plt.tight_layout()
        return self._save_fig(fig, f"trajectory_concept_{method_name}")

    # ─── Concept-wise metric plot ─────────────────────────────────────────────

    def make_concept_metric_plot(
        self,
        metric_data: np.ndarray,
        concept_labels: np.ndarray,
        correct_mask: np.ndarray,
        concept_names: Optional[List[str]] = None,
        metric_name: str = "metric",
        ylabel: str = "",
    ) -> str:
        """
        4×2 subplot grid — one panel per math concept.

        Each panel shows 2 averaged curves (correct=green, incorrect=red) with
        std-deviation bands, identical to the global metric plots but scoped to
        each concept's samples.  1-D metrics (arc length) use a bar chart.

        Args:
            metric_data:    [n_samples, T]  (curve) or [n_samples] (scalar/bar)
            concept_labels: [n_samples] int
            correct_mask:   [n_samples] bool
            concept_names:  list of 7 strings
            metric_name:    used for the output filename
            ylabel:         y-axis label string (supports LaTeX math)
        """
        n_concepts = 7
        is_scalar = metric_data.ndim == 1  # arc-length style: one value per sample

        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        axes_flat = axes.flatten()

        for cid in range(n_concepts):
            ax = axes_flat[cid]
            name = concept_names[cid] if concept_names else f"Concept {cid}"

            mask_concept = concept_labels == cid
            n_total = int(mask_concept.sum())

            if n_total == 0:
                ax.set_title(f"{name}\n(no samples)", fontsize=11)
                ax.axis("off")
                continue

            data_c = metric_data[mask_concept]
            ok_mask = correct_mask[mask_concept]
            n_ok = int(ok_mask.sum())
            n_wrong = n_total - n_ok

            if is_scalar:
                # Bar chart: two bars (correct mean, incorrect mean) with std
                categories, values, errors, colors = [], [], [], []
                if n_ok > 0:
                    categories.append(f"Correct\n(n={n_ok})")
                    values.append(float(np.mean(data_c[ok_mask])))
                    errors.append(float(np.std(data_c[ok_mask])))
                    colors.append("#2ecc71")
                if n_wrong > 0:
                    categories.append(f"Incorrect\n(n={n_wrong})")
                    values.append(float(np.mean(data_c[~ok_mask])))
                    errors.append(float(np.std(data_c[~ok_mask])))
                    colors.append("#e74c3c")
                ax.bar(
                    range(len(categories)), values,
                    yerr=errors, capsize=5,
                    color=colors, edgecolor="black", linewidth=0.8,
                    error_kw={"elinewidth": 1.5},
                )
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, fontsize=9)
            else:
                # Curve plot: mean ± std for correct and incorrect groups
                n_steps = data_c.shape[1]
                steps = np.arange(1, n_steps + 1)

                if n_ok > 0:
                    m = np.nanmean(data_c[ok_mask], axis=0)
                    s = np.nanstd(data_c[ok_mask], axis=0)
                    ax.plot(steps, m, "o-", color="#2ecc71", linewidth=2.0,
                            markersize=5, label=f"Correct (n={n_ok})", zorder=4)
                    ax.fill_between(steps, m - s, m + s, color="#2ecc71", alpha=0.20)

                if n_wrong > 0:
                    m = np.nanmean(data_c[~ok_mask], axis=0)
                    s = np.nanstd(data_c[~ok_mask], axis=0)
                    ax.plot(steps, m, "x--", color="#e74c3c", linewidth=2.0,
                            markersize=5, label=f"Incorrect (n={n_wrong})", zorder=4)
                    ax.fill_between(steps, m - s, m + s, color="#e74c3c", alpha=0.20)

                ax.set_xlabel("Step", fontsize=9)
                ax.legend(fontsize=8, loc="best")

            ax.set_title(f"{name}  (n={n_total})", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.25)

        # Hide the unused 8th panel (4×2 grid, 7 concepts)
        axes_flat[7].set_visible(False)

        fig.suptitle(
            f"Concept-wise: {metric_name.replace('_', ' ').title()}"
            f"  |  N_total = {int(len(concept_labels)):,} samples",
            fontsize=15, y=1.01,
        )
        _concept_footnote(fig, self._concept_footnote_msg())
        plt.tight_layout()
        return self._save_fig(fig, f"concept_{metric_name}")

    # ─── Difficulty-wise trajectory plot (MATH benchmark) ────────────────────

    def make_difficulty_trajectory_plot(
        self,
        embeddings: np.ndarray,
        method_name: str,
        difficulty_labels: np.ndarray,
        correct_mask: np.ndarray,
        difficulty_names: Optional[List[str]] = None,
    ) -> str:
        """
        2D scatter of latent trajectories coloured by MATH difficulty level (Level 1–5).

        Level 1 (easiest) = green, Level 5 (hardest) = red via RdYlGn colormap.
        Correct samples: filled circles. Incorrect: lighter fill, coloured edge.
        """
        n_samples, n_steps, _ = embeddings.shape
        n_levels = 5
        cmap = plt.get_cmap("RdYlGn")
        level_colors = [cmap(0.85 - i * 0.18) for i in range(n_levels)]  # L1=green → L5=red

        fig, ax = plt.subplots(1, 1, figsize=(11, 9))

        for i in range(n_samples):
            traj = embeddings[i]
            lvl = int(difficulty_labels[i])
            if lvl < 0 or lvl >= n_levels:
                continue
            is_correct = bool(correct_mask[i])
            base = level_colors[lvl]

            if is_correct:
                face, edge, alpha, lw_edge = base, (0, 0, 0, 0.6), 0.80, 0.4
            else:
                face = self._lighten_color(base, amount=0.45)
                edge, alpha, lw_edge = base, 0.55, 1.0

            ax.plot(traj[:, 0], traj[:, 1], color=base, alpha=0.10, linewidth=0.6, zorder=1)
            ax.scatter(
                traj[:, 0], traj[:, 1],
                color=[face] * n_steps,
                edgecolors=[edge] * n_steps,
                linewidths=lw_edge,
                marker="o", s=55, alpha=alpha, zorder=2,
            )

        names = difficulty_names or [f"Level {i + 1}" for i in range(n_levels)]
        diff_patches = [
            mpatches.Patch(facecolor=level_colors[i], edgecolor="black",
                           linewidth=0.5, label=names[i])
            for i in range(n_levels)
        ]
        correctness_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                   markeredgecolor="black", markeredgewidth=0.5, markersize=9,
                   label="Correct (filled)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#dddddd",
                   markeredgecolor="#555555", markeredgewidth=1.0, markersize=9,
                   label="Incorrect (open)"),
        ]
        legend1 = ax.legend(handles=diff_patches, title="Difficulty", title_fontsize=10,
                            fontsize=9, loc="upper left", framealpha=0.85)
        ax.add_artist(legend1)
        ax.legend(handles=correctness_handles, title="Correctness", title_fontsize=10,
                  fontsize=9, loc="lower right", framealpha=0.85)

        ax.set_xlabel(f"{method_name.upper()} Dim 1", fontsize=12)
        ax.set_ylabel(f"{method_name.upper()} Dim 2", fontsize=12)
        ax.set_title(
            f"Latent Trajectories — {method_name.upper()}  |  Difficulty Level",
            fontsize=14,
        )
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        return self._save_fig(fig, f"trajectory_difficulty_{method_name}")

    # ─── Difficulty-wise metric plot (MATH benchmark) ────────────────────────

    def make_difficulty_metric_plot(
        self,
        metric_data: np.ndarray,
        difficulty_labels: np.ndarray,
        correct_mask: np.ndarray,
        difficulty_names: Optional[List[str]] = None,
        metric_name: str = "metric",
        ylabel: str = "",
    ) -> str:
        """
        3×2 subplot grid — one panel per difficulty level (Level 1–5).

        Mirrors make_concept_metric_plot but split by difficulty instead of subject.
        1-D metrics (arc length) use bar charts; multi-step metrics use mean±std curves.
        The 6th panel (bottom-right) is hidden.
        """
        n_levels = 5
        is_scalar = metric_data.ndim == 1

        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        axes_flat = axes.flatten()
        names = difficulty_names or [f"Level {i + 1}" for i in range(n_levels)]

        for lvl in range(n_levels):
            ax = axes_flat[lvl]
            name = names[lvl]

            mask_lvl = difficulty_labels == lvl
            n_total = int(mask_lvl.sum())

            if n_total == 0:
                ax.set_title(f"{name}\n(no samples)", fontsize=11)
                ax.axis("off")
                continue

            data_l = metric_data[mask_lvl]
            ok_mask = correct_mask[mask_lvl]
            n_ok = int(ok_mask.sum())
            n_wrong = n_total - n_ok

            if is_scalar:
                categories, values, errors, colors = [], [], [], []
                if n_ok > 0:
                    categories.append(f"Correct\n(n={n_ok})")
                    values.append(float(np.mean(data_l[ok_mask])))
                    errors.append(float(np.std(data_l[ok_mask])))
                    colors.append("#2ecc71")
                if n_wrong > 0:
                    categories.append(f"Incorrect\n(n={n_wrong})")
                    values.append(float(np.mean(data_l[~ok_mask])))
                    errors.append(float(np.std(data_l[~ok_mask])))
                    colors.append("#e74c3c")
                ax.bar(
                    range(len(categories)), values,
                    yerr=errors, capsize=5,
                    color=colors, edgecolor="black", linewidth=0.8,
                    error_kw={"elinewidth": 1.5},
                )
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, fontsize=9)
            else:
                n_steps = data_l.shape[1]
                steps = np.arange(1, n_steps + 1)

                if n_ok > 0:
                    m = np.nanmean(data_l[ok_mask], axis=0)
                    s = np.nanstd(data_l[ok_mask], axis=0)
                    ax.plot(steps, m, "o-", color="#2ecc71", linewidth=2.0,
                            markersize=5, label=f"Correct (n={n_ok})", zorder=4)
                    ax.fill_between(steps, m - s, m + s, color="#2ecc71", alpha=0.20)

                if n_wrong > 0:
                    m = np.nanmean(data_l[~ok_mask], axis=0)
                    s = np.nanstd(data_l[~ok_mask], axis=0)
                    ax.plot(steps, m, "x--", color="#e74c3c", linewidth=2.0,
                            markersize=5, label=f"Incorrect (n={n_wrong})", zorder=4)
                    ax.fill_between(steps, m - s, m + s, color="#e74c3c", alpha=0.20)

                ax.set_xlabel("Step", fontsize=9)
                ax.legend(fontsize=8, loc="best")

            ax.set_title(f"{name}  (n={n_total})", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.25)

        # Hide the unused 6th panel (3×2 grid, 5 levels)
        axes_flat[5].set_visible(False)

        fig.suptitle(
            f"Difficulty-wise: {metric_name.replace('_', ' ').title()}"
            f"  |  N_total = {int(len(difficulty_labels)):,} samples",
            fontsize=15, y=1.01,
        )
        plt.tight_layout()
        return self._save_fig(fig, f"difficulty_{metric_name}")

    # ─── _save_fig with subdir support (used by per-concept variants) ────────

    def _save_fig(
        self,
        fig: plt.Figure,
        name: str,
        subdir: Optional[str] = None,
    ) -> str:
        """Save figure and close, with a retry for transient Windows file locks.

        When subdir is provided the figure is written to
        self.output_dir/<subdir>/<name>.<plot_format> and the subdirectory is
        created on demand. This is used by per-concept plot variants which
        emit one file per concept bucket and would clutter the main plots
        directory if written there.
        """
        out_dir = self.output_dir if subdir is None else self.output_dir / subdir
        if subdir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"{name}.{self.plot_format}"
        # On Windows, a lingering handle from antivirus scan / Explorer preview
        # can cause OSError(22) on first write — retry once after a short wait.
        import time, os
        for attempt in range(3):
            try:
                # Remove stale file if present (sidesteps some filesystem quirks)
                if filepath.exists():
                    try:
                        os.remove(str(filepath))
                    except OSError:
                        pass
                fig.savefig(str(filepath), dpi=self.dpi, bbox_inches="tight")
                break
            except OSError as e:
                if attempt == 2:
                    plt.close(fig)
                    raise
                time.sleep(0.6)
        plt.close(fig)
        print(f"Saved plot: {filepath}")
        return str(filepath)

    def save_figs(self):
        """Placeholder - individual plot methods already save."""
        print(f"All plots saved to {self.output_dir}")
