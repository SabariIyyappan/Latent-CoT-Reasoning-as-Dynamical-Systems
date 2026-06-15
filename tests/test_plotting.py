"""Tests for analysis.plotting shared helpers."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from analysis.plotting import _safe_fill_between


def _min_y_in_collections(ax):
    """Lowest y-coordinate across every fill patch on the axes."""
    ys = []
    for coll in ax.collections:
        for path in coll.get_paths():
            ys.extend(path.vertices[:, 1].tolist())
    return min(ys) if ys else float("inf")


def test_std_band_nonnegative():
    # Jerome's review item 3: perturbation plot's green "correct" band
    # was being drawn below zero by matplotlib's fill_between because
    # mean - std went negative.  With clip_at_zero=True the rendered
    # lower bound must never dip below 0.
    fig, ax = plt.subplots()
    x = np.arange(4)
    mean = np.array([0.30, 0.25, 0.40, 0.35])
    std = np.array([0.50, 0.60, 0.55, 0.45])  # mean - std would be negative
    _safe_fill_between(ax, x, mean, std, color="green", clip_at_zero=True)

    lo = _min_y_in_collections(ax)
    assert lo >= -1e-9, f"clipped band rendered below zero: min_y={lo}"
    plt.close(fig)


def test_std_band_unclipped_renders_negative():
    # Sanity — when clip_at_zero=False the function MUST render the
    # negative lower bound (otherwise the clip test above is vacuous).
    fig, ax = plt.subplots()
    x = np.arange(2)
    mean = np.array([0.30, 0.30])
    std = np.array([0.50, 0.50])
    _safe_fill_between(ax, x, mean, std, color="green", clip_at_zero=False)

    lo = _min_y_in_collections(ax)
    assert lo < 0, f"unclipped band should dip below zero; got min_y={lo}"
    plt.close(fig)
