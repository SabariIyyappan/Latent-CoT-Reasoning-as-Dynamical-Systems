"""
Trajectory features and stability analysis for latent CoT trajectories.

Single module for every per-trajectory metric in the paper. Treats each
sequence of latent states as a discrete-time trajectory in state space
and computes:

Geometric trajectory features
- Step-to-step change size: ||z_{t+1} - z_t||
- Direction consistency: cos(z_{t+1} - z_t, z_t - z_{t-1})
- Arc length / path complexity: sum(||z_{t+1} - z_t||)

Dynamical-systems stability tests
- Fixed-point analysis: local contraction/expansion ||z_k - z_t|| for k = t +/- 1, t +/- 2
- Lyapunov-style sensitivity: neighbor log-divergence rate across parallel trajectories
- Perturbation stability: re-run model with noisy input embeddings, measure divergence

The perturbation test is the only feature that needs the model loaded; it
runs as its own flow (`runner.py --perturbation`) against a cached
latent_states/all_states.h5. Everything else is post-hoc trajectory math.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch

from .wrappers import ModelWrapper

# Import the upstream noise utility so we use the same noise injection
# mechanism the models were designed for (see src/UPSTREAM.md)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.utils import add_noise as latent_add_noise


class StabilityAnalysis:
    """
    Computes geometric trajectory features and stability metrics for
    latent trajectories.

    Takes either:
    1. Pre-computed states (tensor) for post-hoc analysis
    2. A ModelWrapper + question list for active perturbation experiments
    """

    def __init__(
        self,
        states: Optional[torch.Tensor] = None,
        model_wrapper: Optional[ModelWrapper] = None,
        questions: Optional[List[str]] = None,
    ):
        """
        Args:
            states: Tensor [n_samples, n_steps, dim] - pre-computed latent trajectories
            model_wrapper: For active perturbation (re-running model with noise)
            questions: Input questions (needed if model_wrapper is used)
        """
        self.states = states.float() if states is not None else None
        self.wrapper = model_wrapper
        self.questions = questions
        if states is not None:
            self.n_samples, self.n_steps, self.dim = states.shape
        else:
            self.n_samples = self.n_steps = self.dim = 0

        self.features: Dict[str, np.ndarray] = {}

    # ─────────────────────────────────────────────────────────────
    # 1. Geometric trajectory features (post-hoc)
    # ─────────────────────────────────────────────────────────────

    def calc_step2step_change(self) -> np.ndarray:
        """
        Compute step-to-step change size: ||z_{t+1} - z_t||

        Measures the magnitude of change between consecutive latent states.
        Large jumps may indicate phase transitions in reasoning.

        Returns:
            Array of shape [n_samples, n_steps - 1] with L2 norms of
            consecutive state differences.
        """
        diffs = self.states[:, 1:, :] - self.states[:, :-1, :]  # [n, T-1, d]
        norms = torch.norm(diffs, dim=-1).numpy()               # [n, T-1]

        self.features["step2step_change"] = norms
        return norms

    def calc_direction_consistency(self) -> np.ndarray:
        """
        Compute direction consistency: cos(z_{t+1} - z_t, z_t - z_{t-1})

        Measures how consistently the trajectory moves in the same direction.
        High values (close to 1) indicate smooth, consistent reasoning paths.
        Low/negative values indicate direction reversals or erratic paths.

        Returns:
            Array of shape [n_samples, n_steps - 2] with cosine similarities
            between consecutive displacement vectors.
        """
        diffs = self.states[:, 1:, :] - self.states[:, :-1, :]  # [n, T-1, d]

        v1 = diffs[:, :-1, :]  # z_t - z_{t-1},     [n, T-2, d]
        v2 = diffs[:, 1:, :]   # z_{t+1} - z_t,     [n, T-2, d]

        dot = torch.sum(v1 * v2, dim=-1)                    # [n, T-2]
        norm1 = torch.norm(v1, dim=-1).clamp(min=1e-8)
        norm2 = torch.norm(v2, dim=-1).clamp(min=1e-8)
        cosine = (dot / (norm1 * norm2)).numpy()

        self.features["direction_consistency"] = cosine
        return cosine

    def calc_arc_length(self) -> np.ndarray:
        """
        Compute arc length / path complexity: sum(||z_{t+1} - z_t||)

        Total path length traversed by the latent trajectory.
        Longer paths may indicate more complex reasoning processes.

        Returns:
            Array of shape [n_samples] with total arc length per trajectory.
        """
        if "step2step_change" not in self.features:
            self.calc_step2step_change()

        arc_lengths = self.features["step2step_change"].sum(axis=-1)  # [n]
        self.features["arc_length"] = arc_lengths
        return arc_lengths

    # ─────────────────────────────────────────────────────────────
    # 2. Fixed-point / local contraction analysis (post-hoc)
    # ─────────────────────────────────────────────────────────────

    def calc_fixed_point_distances(self, max_lag: int = 2) -> Dict[str, np.ndarray]:
        """
        Fixed point analysis: compute ||z_k - z_t|| for k = t +/- 1, t +/- 2, ...

        A fixed point is a state z* where f(z*) = z*. Near a fixed point, the
        trajectory slows down: distances to neighbors become small.

        Returns a dict with:
            - distances_lag_{l}: [n_samples, n_steps] array of ||z_{t+l} - z_t||
              for l in [-max_lag, max_lag] (l=0 is zero, skipped)
            - contraction_ratio: [n_samples, n_steps-2] ratio ||z_{t+2}-z_{t+1}||/||z_{t+1}-z_t||
              values < 1 indicate contraction (stable), > 1 indicate expansion
        """
        n, T, _ = self.states.shape
        result = {}

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                continue
            dist = torch.full((n, T), float("nan"))
            abs_lag = abs(lag)
            if lag > 0:
                # ||z_{t+lag} - z_t|| valid for t in [0, T-lag)
                diff = self.states[:, lag:, :] - self.states[:, : T - lag, :]
                dist[:, : T - lag] = torch.norm(diff, dim=-1)
            else:
                # ||z_{t-abs_lag} - z_t|| valid for t in [abs_lag, T)
                diff = self.states[:, : T - abs_lag, :] - self.states[:, abs_lag:, :]
                dist[:, abs_lag:] = torch.norm(diff, dim=-1)
            result[f"distances_lag_{lag}"] = dist.numpy()

        # Contraction ratio: how much each step shrinks or expands the step size
        steps = torch.norm(self.states[:, 1:, :] - self.states[:, :-1, :], dim=-1)  # [n, T-1]
        # ratio = step[t+1] / step[t]
        ratio = steps[:, 1:] / steps[:, :-1].clamp(min=1e-10)  # [n, T-2]
        result["contraction_ratio"] = ratio.numpy()

        # Is the trajectory contracting on average?
        result["mean_log_contraction"] = np.log(ratio.numpy() + 1e-10).mean(axis=-1)  # [n]

        self.features.update(result)
        return result

    # ─────────────────────────────────────────────────────────────
    # 3. Lyapunov-style sensitivity (post-hoc local estimate)
    # ─────────────────────────────────────────────────────────────

    def calc_local_lyapunov(self) -> np.ndarray:
        """
        Neighbor log-divergence rate as Lyapunov surrogate.

        At each step t, for each sample i, finds the nearest other sample j*
        in state space and measures how quickly that pair separates over one step:

            lambda_nbr(t, i) = log( ||z_{t+1}^{j*} - z_{t+1}^i||
                                    / ||z_t^{j*} - z_t^i|| )

        Positive = nearby trajectories diverging (sensitive dynamics).
        Negative = nearby trajectories converging (stable dynamics).

        This is a proper Lyapunov surrogate: it directly asks whether a small
        perturbation to the current state (represented by the nearest neighbor
        across samples) grows or decays in one step — without requiring model
        re-runs or analytic Jacobians.

        Returns:
            Array of shape [n_samples, n_steps - 1] with per-step neighbor
            divergence rates.

        Stores:
            features["local_lyapunov"] — [n, T-1] per-sample per-step rates
            features["lyapunov_mean"]  — [n] mean rate per sample
        """
        n, T, d = self.states.shape
        states_np = self.states.numpy()

        neighbor_rates = np.full((n, T - 1), np.nan)  # [n, T-1]

        for t in range(T - 1):
            Z_t  = states_np[:, t,     :]  # [n, d]
            Z_t1 = states_np[:, t + 1, :]  # [n, d]

            # Pairwise squared distances at step t via dot-product trick
            sq = (Z_t ** 2).sum(axis=1)                            # [n]
            D2 = sq[:, None] + sq[None, :] - 2.0 * (Z_t @ Z_t.T) # [n, n]
            D2 = np.maximum(D2, 0.0)                               # numerical floor
            np.fill_diagonal(D2, np.inf)                           # exclude self-pairs

            j_star = np.argmin(D2, axis=1)                         # [n]
            d_t    = np.sqrt(D2[np.arange(n), j_star])            # [n]
            d_t1   = np.linalg.norm(Z_t1 - Z_t1[j_star], axis=1) # [n]

            neighbor_rates[:, t] = np.log(
                np.maximum(d_t1, 1e-10) / np.maximum(d_t, 1e-10)
            )

        neighbor_mean = np.nanmean(neighbor_rates, axis=1)  # [n]

        self.features["local_lyapunov"] = neighbor_rates
        self.features["lyapunov_mean"]  = neighbor_mean
        return neighbor_rates

    # ─────────────────────────────────────────────────────────────
    # 4. Active perturbation stability (requires model; separate flow)
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def calc_perturbation_stability(
        self,
        noise_std: float = 0.01,
        n_perturbations: int = 3,
        seed: int = 42,
        clean_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Perturbation stability test: re-run the model with noisy input embeddings
        and measure how much the latent trajectory changes.

        For each question, runs the model n_perturbations times with different
        Gaussian noise added to the input embeddings. Measures trajectory
        divergence as the average L2 distance between clean and perturbed
        trajectories at each latent step.

        Args:
            clean_states: Optional [n_samples, n_steps, d] tensor of pre-computed
                clean trajectories. When provided, avoids re-running the model for
                the clean baseline (saves n_samples × 1 forward passes).
                If None, the clean trajectory is re-inferred for each question.

        Returns dict with:
            - divergence: [n_samples, n_steps] mean L2 distance from clean trajectory
            - divergence_std: [n_samples, n_steps] std across perturbations
            - relative_divergence: [n_samples, n_steps] divergence / ||z_t||
              (scale-invariant measure)
        """
        if self.wrapper is None or self.questions is None:
            raise ValueError("Perturbation stability requires model_wrapper and questions")

        model = self.wrapper.model
        tokenizer = self.wrapper.tokenizer

        divergences = []
        for q_idx, question in enumerate(self.questions):
            # Clean trajectory (reference) — use pre-computed states if available
            # to avoid re-running n_samples extra forward passes.
            if clean_states is not None:
                clean_traj = clean_states[q_idx].cpu()  # [T, d]
            else:
                clean_out = self.wrapper.run_inference(question)
                clean_traj = self.wrapper.extract_latent_thoughts(clean_out)  # [T, d]
                del clean_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Perturbed trajectories
            perturbed_trajs = []
            formatted = self.wrapper.format_input(question)
            inputs = tokenizer(formatted, return_tensors="pt").to(self.wrapper.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            base_embeds = model.get_input_embeddings()(input_ids)  # [1, L, d]

            for p in range(n_perturbations):
                torch.manual_seed(seed + p)
                # Use LatentTTS's add_noise() for consistency with the model's
                # own stochastic sampling mechanism (same Gaussian approach but
                # supports optional masking for targeted perturbation)
                noisy_embeds = latent_add_noise(base_embeds.clone(), std=noise_std)

                # The LatentGenerationMixin.generate() expects both input_ids
                # (for detecting latent_id tokens) and inputs_embeds (the actual
                # values used). Since our prompt has no <|latent|> tokens yet
                # (only <|start-latent|>), the model will use inputs_embeds
                # directly without any latent replacement.
                try:
                    out = model.generate(
                        input_ids=input_ids,
                        inputs_embeds=noisy_embeds,
                        attention_mask=attention_mask,
                        generation_config=self.wrapper.generation_config,
                        num_return_sequences=1,
                        use_cache=True,
                    )
                    pt = getattr(out, "latent_thoughts", None)
                    if pt is None:
                        del out
                        continue
                    if pt.dim() == 3:
                        pt = pt[0]
                    perturbed_trajs.append(pt.cpu())
                    del out
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"  perturbation {p} failed for Q{q_idx}: {type(e).__name__}: {e}")

            if not perturbed_trajs:
                # No successful perturbations
                divergences.append({
                    "mean": np.full(clean_traj.shape[0], np.nan),
                    "std": np.full(clean_traj.shape[0], np.nan),
                    "rel": np.full(clean_traj.shape[0], np.nan),
                })
                continue

            perturbed = torch.stack(perturbed_trajs, dim=0)  # [P, T, d]
            # Align shapes if needed
            min_T = min(clean_traj.shape[0], perturbed.shape[1])
            clean_traj = clean_traj[:min_T]
            perturbed = perturbed[:, :min_T]

            # Divergence at each step: mean L2 distance from clean
            diff = perturbed - clean_traj.unsqueeze(0)  # [P, T, d]
            dist = torch.norm(diff, dim=-1)  # [P, T]

            mean_div = dist.mean(dim=0).numpy()  # [T]
            std_div = dist.std(dim=0).numpy()
            clean_norm = torch.norm(clean_traj, dim=-1).numpy()  # [T]
            rel_div = mean_div / (clean_norm + 1e-10)

            divergences.append({"mean": mean_div, "std": std_div, "rel": rel_div})

        # All trajectories should have the same length (fixed latent_length).
        # If any sample failed entirely, it will have NaN arrays of the same
        # length as the clean trajectory (set in the failure branch above).
        # Verify uniform length before stacking.
        lengths = set(len(d["mean"]) for d in divergences)
        assert len(lengths) == 1, (
            f"Expected uniform trajectory lengths but got {lengths}. "
            f"This can happen if latent_length varies across samples "
            f"(e.g. with max_latent_length mode). Consider using fixed "
            f"latent_length for perturbation analysis."
        )

        div_mean = np.stack([d["mean"] for d in divergences])
        div_std = np.stack([d["std"] for d in divergences])
        div_rel = np.stack([d["rel"] for d in divergences])

        result = {
            "perturbation_divergence": div_mean,
            "perturbation_divergence_std": div_std,
            "perturbation_relative_divergence": div_rel,
            "perturbation_noise_std": np.array([noise_std]),
            "perturbation_n_perturbations": np.array([n_perturbations]),
        }

        self.features.update(result)
        return result

    # ─────────────────────────────────────────────────────────────
    # 5. Aggregate summary
    # ─────────────────────────────────────────────────────────────

    def compute_all(
        self,
        include_perturbation: bool = False,
        perturbation_kwargs: Optional[Dict] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute all trajectory features.

        Geometric features (step-to-step change, direction consistency,
        arc length) plus fixed-point distances and Lyapunov sensitivity.

        Args:
            include_perturbation: Whether to run the active perturbation test
                (requires model_wrapper and questions). The cam-ready pipeline
                keeps perturbation as a separate flow; this stays False there.
            perturbation_kwargs:  Forwarded to calc_perturbation_stability.
        """
        if self.states is not None:
            self.calc_step2step_change()
            self.calc_direction_consistency()
            self.calc_arc_length()
            self.calc_fixed_point_distances(max_lag=2)
            self.calc_local_lyapunov()

        if include_perturbation:
            kwargs = perturbation_kwargs or {}
            self.calc_perturbation_stability(**kwargs)

        return self.features

    def save_features(self, output_dir: str):
        """Save all computed features to HDF5 (stability_feats/stability.h5)."""
        out = Path(output_dir) / "stability_feats"
        out.mkdir(parents=True, exist_ok=True)

        filepath = out / "stability.h5"
        with h5py.File(str(filepath), "w") as f:
            for name, arr in self.features.items():
                f.create_dataset(name, data=np.asarray(arr))

        print(f"Saved trajectory features to {filepath}")
        return str(filepath)

    @staticmethod
    def load_features(filepath: str) -> Dict[str, np.ndarray]:
        features = {}
        with h5py.File(filepath, "r") as f:
            for key in f.keys():
                features[key] = f[key][:]
        return features

    def summary(self) -> str:
        """Human-readable summary of computed features."""
        lines = [f"Trajectory Feature Summary ({self.n_samples} samples, {self.n_steps} steps)"]
        lines.append("=" * 60)

        if "step2step_change" in self.features:
            s = self.features["step2step_change"]
            lines.append("Step-to-step change ||z_{t+1} - z_t||:")
            lines.append(f"  Mean: {np.nanmean(s):.4f}, Std: {np.nanstd(s):.4f}")
            lines.append(f"  Min:  {np.nanmin(s):.4f}, Max: {np.nanmax(s):.4f}")

        if "direction_consistency" in self.features:
            d = self.features["direction_consistency"]
            lines.append("Direction consistency cos(delta_t, delta_{t-1}):")
            lines.append(f"  Mean: {np.nanmean(d):.4f}, Std: {np.nanstd(d):.4f}")
            lines.append(f"  Min:  {np.nanmin(d):.4f}, Max: {np.nanmax(d):.4f}")

        if "arc_length" in self.features:
            a = self.features["arc_length"]
            lines.append("Arc length (total path):")
            lines.append(f"  Mean: {np.nanmean(a):.4f}, Std: {np.nanstd(a):.4f}")
            lines.append(f"  Min:  {np.nanmin(a):.4f}, Max: {np.nanmax(a):.4f}")

        if "local_lyapunov" in self.features:
            lyap = self.features["local_lyapunov"]
            mean_lyap = self.features["lyapunov_mean"]
            lines.append("Lyapunov surrogate (neighbor log-divergence rate):")
            lines.append(f"  Mean across all samples & steps: {np.nanmean(lyap):.4f}")
            n_stable = int(np.sum(mean_lyap < 0))
            lines.append(f"  Converging (mean < 0): {n_stable}/{len(mean_lyap)} samples")
            n_diverging = len(mean_lyap) - n_stable
            lines.append(f"  Diverging  (mean > 0): {n_diverging}/{len(mean_lyap)} samples")

        if "contraction_ratio" in self.features:
            ratio = self.features["contraction_ratio"]
            lines.append("Contraction ratio (step[t+1]/step[t]):")
            lines.append(f"  Mean: {np.nanmean(ratio):.4f}")

        if "perturbation_divergence" in self.features:
            pd = self.features["perturbation_divergence"]
            lines.append("Perturbation stability (L2 divergence from clean):")
            lines.append(f"  Mean divergence per step (avg across samples): {np.nanmean(pd, axis=0)}")

        text = "\n".join(lines)
        print(text)
        return text
