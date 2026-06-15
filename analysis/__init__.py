"""
Latent CoT Dynamical Systems Analysis — shared library.

This package is paradigm-agnostic: both vanilla CoT and Sim-CoT runs flow
through the same modules. Top-level imports are stable and form the
documented public surface used by the rest of the cam-ready trunk.

Submodules:
    data_prep    — GSM8K loading + concept-bucket classification.
    wrappers     — ModelWrapper, MODEL_REGISTRY (single source of truth
                   for per-method special tokens, prompt templates, and
                   checkpoint loading).
    dim_reduct   — PCA / t-SNE / UMAP / PHATE / DMD with consistent API.
    stability    — Trajectory features (step-to-step change, direction
                   consistency, arc length), fixed-point distances,
                   Lyapunov sensitivity, and active perturbation.
    plotting     — All plot functions; figure paths land under
                   <run_dir>/plots/ alongside the cached HDF5 features.

Usage:
    from analysis import ModelWrapper, DimReduct, StabilityAnalysis, Plotting
    from analysis import DataPrep
"""

from .wrappers import ModelWrapper, MODEL_REGISTRY
from .data_prep import (
    DataPrep,
    GSM8K_CONCEPT_NAMES,
    CONCEPT_NAMES,
)
from .dim_reduct import DimReduct
from .stability import StabilityAnalysis
from .plotting import Plotting

__all__ = [
    "ModelWrapper",
    "MODEL_REGISTRY",
    "DataPrep",
    "GSM8K_CONCEPT_NAMES",
    "CONCEPT_NAMES",
    "DimReduct",
    "StabilityAnalysis",
    "Plotting",
]
