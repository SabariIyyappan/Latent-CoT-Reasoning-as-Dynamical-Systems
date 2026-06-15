# `scripts/` — analysis workflows

Post-inference analysis scripts, grouped by what they produce in the
paper.

| Subfolder | Purpose | Paper reference |
|---|---|---|
| `perturbation/` | Paper-grid perturbation runs (σ ∈ {0.01, 0.1, 1.0}) + the consolidated 3-σ grid plot | §5.1, hyperparameters A.1.2 |
| `trajectory/` | 3-D (k = 3) DMD / PHATE trajectory plots + initial-vs-SimCoT comparison | §4.3.1, A.1.1 |
| `replay/` | Re-run the analysis (or just re-render plots) from a cached `latent_states/all_states.h5` — no GPU needed | Figures 2–9 |

## Which script produces which figure

| Paper section | Figure | Producer |
|---|---|---|
| §1        | Fig 1 — Pipeline overview                   | schematic (not pipeline-generated) |
| §5.1      | Fig 2 — Step-to-step change                 | main pipeline (`runner.py`) |
| §5.1      | Fig 3 — Lyapunov sensitivity                | main pipeline |
| §5.1      | Fig 4 — Direction consistency               | main pipeline |
| §5.2      | Fig 5 — DMD trajectory                      | main pipeline |
| §5.2      | Fig 6 — PHATE trajectory                    | main pipeline |
| §5.2      | Fig 7 — UMAP trajectory                     | main pipeline |
| A.2.1     | Fig 8 — PCA projections                     | main pipeline |
| A.2.2     | Fig 9 — t-SNE projections                   | main pipeline |

"Main pipeline" = one `runner.py --paradigm <p> --method <m>` run per
cell; the replay scripts below regenerate the same figures from the
cached HDF5 without re-running inference.

Pipeline outputs without a numbered paper figure: arc length (Eq. 3,
discussed in §5.1), fixed-point distances (§4.3.2), concept-wise metric
breakdowns, perturbation divergence (§5.1 / A.1.2), and the 3-D
trajectory views (§4.3.1).

## Typical usage patterns

### Re-run analysis from cached trajectories (no GPU)

```bash
# Full analysis recompute (reductions + features + plots):
python scripts/replay/analyze_from_cache.py results/vanilla_codi/<run>/
# … equivalent to:
python runner.py --paradigm vanilla --method codi --run_dir results/vanilla_codi/<run>/

# Plots only (reads the already-saved feature HDF5s):
python scripts/replay/replot_from_cache.py results/vanilla_codi/<run>/
```

### Perturbation stability — quick single run

The perturbation flow lives in `runner.py` and consumes a finished
run's H5 trajectories as the clean baseline:

```bash
python runner.py --paradigm simcot --method codi --perturbation \
    --run_dir results/simcot_codi/<run>/ --noise_std 0.01
```

### Perturbation stability — full σ grid (§5.1)

```bash
# σ ∈ {0.01, 0.1, 1.0} × method ∈ {codi, coconut} — six runs
for sigma in 0.01 0.1 1.0; do
  for method in codi coconut; do
    python scripts/perturbation/add_perturbation.py \
      --method $method \
      --checkpoint checkpoints/simcot_${method}_gpt2 \
      --out_dir results/simcot_${method}_literal \
      --n_samples 2000 --noise_std $sigma \
      --filename_suffix _sigma${sigma} \
      --stability_key_suffix _sigma${sigma}
  done
done

# Then produce the consolidated 3-σ grids
python scripts/perturbation/plot_perturbation_relative_grid.py
```

`add_perturbation.py` writes σ-suffixed HDF5 keys and PNG filenames so
multiple σ runs coexist in one result tree. The Colab notebook under
`notebooks/` wraps this grid for hosted GPUs.

### 3-D trajectory plots

```bash
python scripts/trajectory/add_3d_plots.py results/vanilla_codi/<run>/     # vanilla cells
python scripts/trajectory/add_3d_plots.py results/simcot_codi/<run>/      # Sim-CoT cells

# Side-by-side comparison
python scripts/trajectory/compare_initial_vs_simcot_3d.py
```
