"""
Dimensionality reduction for latent thought trajectories.

Supports PCA, t-SNE, UMAP, Dynamic Mode Decomposition (DMD via PyDMD),
and PHATE for projecting high-dimensional latent states into visualizable spaces.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import h5py
import numpy as np
import torch


class DimReduct:
    """
    Dimensionality reduction on latent thought states.

    Takes raw latent trajectories [n_samples, n_steps, hidden_dim] and
    produces reduced representations for visualization and analysis.
    """

    def __init__(self, states: torch.Tensor):
        """
        Args:
            states: Tensor of shape [n_samples, n_steps, hidden_dim]
        """
        self.states = states
        self.n_samples, self.n_steps, self.hidden_dim = states.shape
        # Flatten for methods that expect 2D input: [n_samples * n_steps, hidden_dim]
        self.states_flat = states.reshape(-1, self.hidden_dim).numpy()
        self.reduced: Dict[str, np.ndarray] = {}
        self.pca_explained_variance_ratio: Optional[np.ndarray] = None

    def perform_pca(self, n_components: int = 2) -> np.ndarray:
        """
        Perform PCA dimensionality reduction.

        Linear projection onto directions of maximum variance. Fast, deterministic,
        and interpretable. Good baseline for comparison against nonlinear methods.

        Returns:
            Array of shape [n_samples, n_steps, n_components]
        """
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components, random_state=42)
        embedding = pca.fit_transform(self.states_flat)
        result = embedding.reshape(self.n_samples, self.n_steps, n_components)
        self.reduced["pca"] = result

        self.pca_explained_variance_ratio = pca.explained_variance_ratio_
        print(f"PCA: {self.states_flat.shape} -> {embedding.shape} -> reshaped {result.shape}")
        print(f"  explained variance ratio: {pca.explained_variance_ratio_}")
        return result

    def perform_tsne(
        self,
        n_components: int = 2,
        perplexity: float = 5.0,
        metric: str = "euclidean",
    ) -> np.ndarray:
        """
        Perform t-SNE dimensionality reduction.

        Preserves local neighborhood structure. Perplexity auto-adjusts
        to be < (n_points - 1) / 3 for small datasets.

        Returns:
            Array of shape [n_samples, n_steps, n_components]
        """
        from sklearn.manifold import TSNE

        n_points = self.states_flat.shape[0]
        effective_perplexity = min(perplexity, max(2.0, (n_points - 1) / 3.0))
        if effective_perplexity != perplexity:
            print(f"t-SNE: perplexity {perplexity} -> {effective_perplexity:.2f} (n_points={n_points})")

        tsne = TSNE(
            n_components=n_components,
            perplexity=effective_perplexity,
            metric=metric,
            random_state=42,
            init="pca",
        )
        embedding = tsne.fit_transform(self.states_flat)
        result = embedding.reshape(self.n_samples, self.n_steps, n_components)
        self.reduced["tsne"] = result
        print(f"t-SNE: {self.states_flat.shape} -> {embedding.shape} -> reshaped {result.shape}")
        return result

    def perform_umap(
        self,
        n_components: int = 2,
        n_neighbors: int = 5,
        min_dist: float = 0.1,
        metric: str = "euclidean",
    ) -> np.ndarray:
        """
        Perform UMAP dimensionality reduction.

        Returns:
            Array of shape [n_samples, n_steps, n_components]
        """
        import umap

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
        )
        embedding = reducer.fit_transform(self.states_flat)
        result = embedding.reshape(self.n_samples, self.n_steps, n_components)
        self.reduced["umap"] = result
        print(f"UMAP: {self.states_flat.shape} -> {embedding.shape} -> reshaped {result.shape}")
        return result

    def perform_dmd(
        self,
        svd_rank: int = -1,
        exact: bool = True,
        n_components: int = 2,
    ) -> np.ndarray:
        """
        Perform Dynamic Mode Decomposition via a single batched GPU computation.

        DMD treats the latent trajectory as a discrete-time dynamical system
        z_{t+1} = A z_t and finds spatial modes (eigenvectors of A) and
        their temporal dynamics (eigenvalues of A).

        All N trajectories are decomposed in one batched call to
        torch.linalg.svd + torch.linalg.eig instead of N serial per-sample
        calls, giving ~100x speedup at large N on GPU.

        Eigenvalue magnitude interpretation:
            |lambda| < 1 -> mode decays (stable / contracting)
            |lambda| = 1 -> mode is neutral (periodic / circular)
            |lambda| > 1 -> mode grows (unstable / expanding)

        Also populates self.dmd_eigenvalues [n_samples, n_modes] and
        self.dmd_growth_rates [n_samples, n_modes] for downstream analysis.

        Args:
            n_components: Number of DMD modes to project the trajectory onto
                for visualisation. 2 (default) reproduces the original 2-D
                projection; pass 3 to obtain a 3-D projection (used by the
                3-D trajectory plots requested for the SimCoT vs initial
                comparison).

        Returns:
            Array of shape [n_samples, n_steps, n_components] (trajectory
            projected onto first n_components DMD modes).
        """
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N, T, D = self.states.shape

        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")

        # Snapshot matrices: X1 = z_{0..T-2}, X2 = z_{1..T-1}
        # DMD convention: columns = snapshots, rows = features → [N, D, T-1]
        X1 = self.states[:, :-1, :].permute(0, 2, 1).to(_device).float()
        X2 = self.states[:, 1:, :].permute(0, 2, 1).to(_device).float()
        X_full = self.states.permute(0, 2, 1).to(_device).float()   # [N, D, T]

        # Thin SVD of X1 → U [N, D, r], s [N, r], Vh [N, r, T-1]
        # With full_matrices=False, r = min(D, T-1) = T-1 = 5 for 6-step trajectories.
        U, s, Vh = torch.linalg.svd(X1, full_matrices=False)

        if svd_rank > 0:
            r = min(svd_rank, s.shape[-1])
            U, s, Vh = U[:, :, :r], s[:, :r], Vh[:, :r, :]

        # Guard against degenerate (constant) trajectories with zero singular values.
        s_safe = s.clamp(min=1e-10)
        V = Vh.transpose(-1, -2)   # [N, T-1, r]

        # Reduced operator  A_tilde = U^T X2 V Σ^{-1}  →  [N, r, r]
        # Dividing each column k of (U^T X2 V) by s_safe[k]:
        #   unsqueeze(-2) broadcasts s_safe [N, r] → [N, 1, r]
        A_tilde = (
            torch.bmm(torch.bmm(U.transpose(-1, -2), X2), V)
            / s_safe.unsqueeze(-2)
        )

        # Eigendecompose A_tilde; returns complex eigenvalues and right eigenvectors
        eigvals, W = torch.linalg.eig(A_tilde)   # [N, r] complex, [N, r, r] complex

        # Exact DMD modes  Φ = X2 V Σ^{-1} W  →  [N, D, r] complex
        modes = torch.bmm(
            (X2 @ V / s_safe.unsqueeze(-2)).to(torch.complex64),
            W,
        )

        # Project all T snapshots onto first n_components DMD modes
        #   Re(Φ[:, :n_components]^H  X_full)  →  [N, n_components, T]  →  [N, T, n_components]
        # When fewer than n_components modes exist (rare; only with svd_rank
        # constraints on very short trajectories), zero-pad to keep the
        # output shape uniform across the batch.
        n_modes = modes.shape[-1]
        if n_modes >= n_components:
            modes_k = modes[:, :, :n_components]
        else:
            modes_k = torch.cat(
                [modes,
                 torch.zeros(N, D, n_components - n_modes,
                             dtype=torch.complex64, device=_device)],
                dim=-1,
            )
        projected = torch.real(
            torch.bmm(modes_k.conj().transpose(-1, -2), X_full.to(torch.complex64))
        )                                              # [N, n_components, T]
        result = projected.permute(0, 2, 1).cpu().numpy()   # [N, T, n_components]


        # Eigenvalues and derived spectral quantities → CPU numpy
        # All N samples have the same r modes (uniform shape; no NaN padding needed).
        eigvals_np = eigvals.cpu().numpy()             # [N, r] complex128
        with np.errstate(divide="ignore", invalid="ignore"):
            growth_rates = np.log(np.abs(eigvals_np))  # log|λ| — growth/decay rate
            frequencies  = np.angle(eigvals_np)         # arg(λ) — rotation frequency

        # Store under "dmd" for n_components==2 (back-compat with downstream
        # consumers that read `self.reduced["dmd"]`); under "dmd{N}d" for
        # higher dimensions used by the 3-D trajectory plots.
        key = "dmd" if n_components == 2 else f"dmd{n_components}d"
        self.reduced[key]        = result
        self.dmd_eigenvalues     = eigvals_np
        self.dmd_growth_rates    = growth_rates
        self.dmd_frequencies     = frequencies

        max_abs_eigs = np.nanmax(np.abs(eigvals_np), axis=-1)   # [N]
        n_stable     = int(np.sum(max_abs_eigs < 1.0))
        print(f"DMD: batched decomposition ({_device}), output shape {result.shape}")
        print(f"  Eigenvalues shape: {eigvals_np.shape}")
        print(f"  Max |eig| — mean {max_abs_eigs.mean():.3f} / "
              f"min {max_abs_eigs.min():.3f} / max {max_abs_eigs.max():.3f}")
        print(f"  Stable (all |eig| < 1): {n_stable}/{N} samples")
        return result

    def get_dmd_spectral_summary(self) -> Dict[str, np.ndarray]:
        """
        Return DMD spectral analysis summary for downstream use.

        Must be called after perform_dmd(). Returns a dict with:
            - eigenvalues: [n_samples, n_modes] complex
            - growth_rates: [n_samples, n_modes] log|eig| (stability indicator)
            - frequencies: [n_samples, n_modes] arg(eig) (rotation frequency)
            - max_abs_eig: [n_samples] spectral radius per sample
            - n_unstable_modes: [n_samples] count of modes with |eig| > 1
        """
        if not hasattr(self, "dmd_eigenvalues"):
            raise RuntimeError("Must call perform_dmd() before get_dmd_spectral_summary()")

        eigs = self.dmd_eigenvalues
        abs_eigs = np.abs(eigs)
        return {
            "eigenvalues_real": eigs.real,
            "eigenvalues_imag": eigs.imag,
            "growth_rates": self.dmd_growth_rates,
            "frequencies": self.dmd_frequencies,
            "abs_eigenvalues": abs_eigs,
            "max_abs_eig": np.nanmax(abs_eigs, axis=-1),
            "n_unstable_modes": np.nansum(abs_eigs > 1.0, axis=-1).astype(np.int32),
        }

    def perform_phate(
        self,
        n_components: int = 2,
        knn: int = 5,
        t: str = "auto",
    ) -> np.ndarray:
        """
        Perform PHATE dimensionality reduction.

        PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding)
        preserves both local and global structure, ideal for trajectory data.

        Returns:
            Array of shape [n_samples, n_steps, n_components]
        """
        import phate

        phate_op = phate.PHATE(
            n_components=n_components,
            knn=knn,
            t=t,
            random_state=42,
            verbose=False,
        )
        embedding = phate_op.fit_transform(self.states_flat)
        result = embedding.reshape(self.n_samples, self.n_steps, n_components)
        self.reduced["phate"] = result
        print(f"PHATE: {self.states_flat.shape} -> {embedding.shape} -> reshaped {result.shape}")
        return result

    def save_reduced_states(self, output_dir: str):
        """Save all reduced state representations to HDF5."""
        out = Path(output_dir) / "reduced_states"
        out.mkdir(parents=True, exist_ok=True)

        for method_name, data in self.reduced.items():
            filepath = out / f"{method_name}_reduced.h5"
            with h5py.File(str(filepath), "w") as f:
                f.create_dataset("embedding", data=data)
                f.attrs["method"] = method_name
                f.attrs["n_samples"] = self.n_samples
                f.attrs["n_steps"] = self.n_steps
                f.attrs["original_dim"] = self.hidden_dim
            print(f"Saved {method_name} reduced states to {filepath}")

        return str(out)

    @staticmethod
    def load_reduced_states(filepath: str) -> np.ndarray:
        """Load reduced states from HDF5."""
        with h5py.File(filepath, "r") as f:
            return f["embedding"][:]
