#!/usr/bin/env bash
# main_pipeline.sh
# ----------------------------------------------------------------------
# Set up the environment and run the MAIN analysis pipeline for one cell:
# H5 latent prep (inference) + analysis (reductions, trajectory features,
# all figures). Five clear steps, top to bottom.
#
# Run one cell (defaults to vanilla + codi):
#     bash pipeline_scripts/main_pipeline.sh
#
# Pick a different cell with the PARADIGM / METHOD variables:
#     PARADIGM=simcot METHOD=coconut bash pipeline_scripts/main_pipeline.sh
#
# Quick smoke run (few samples, CPU) before committing to a full run:
#     N_SAMPLES=5 DEVICE=cpu bash pipeline_scripts/main_pipeline.sh
# ----------------------------------------------------------------------
set -euo pipefail

# Project root (this script lives in pipeline_scripts/, one level down).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── 1. Initialize the virtual environment ─────────────────────────────
if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  python -m venv "${REPO_ROOT}/.venv"
fi
# Activate it (Windows Git Bash uses Scripts/, Linux/macOS use bin/).
if [[ -f "${REPO_ROOT}/.venv/Scripts/activate" ]]; then
  source "${REPO_ROOT}/.venv/Scripts/activate"
else
  source "${REPO_ROOT}/.venv/bin/activate"
fi

# ── 2. Install the requirements ───────────────────────────────────────
pip install -r "${REPO_ROOT}/requirements.txt"

# ── 3. Move into the project folder ───────────────────────────────────
cd "${REPO_ROOT}"

# ── 4. Choose the method and paradigm to run ──────────────────────────
# Method   : codi | coconut
# Paradigm : vanilla | simcot
#
# The four available combinations (set PARADIGM and METHOD to pick one):
#     PARADIGM=vanilla METHOD=codi
#     PARADIGM=vanilla METHOD=coconut
#     PARADIGM=simcot  METHOD=codi
#     PARADIGM=simcot  METHOD=coconut
PARADIGM="${PARADIGM:-vanilla}"
METHOD="${METHOD:-codi}"
# Optional smoke-test knobs — leave N_SAMPLES empty for the full dataset.
N_SAMPLES="${N_SAMPLES:-}"
DEVICE="${DEVICE:-}"

# ── 5. Run the main analysis pipeline ─────────────────────────────────
# Writes results/<paradigm>_<method>/<timestamp>/ with the latent-state
# HDF5, all reductions, trajectory features, and figures for this cell.
EXTRA=()
[[ -n "${N_SAMPLES}" ]] && EXTRA+=(--n_samples "${N_SAMPLES}")
[[ -n "${DEVICE}" ]] && EXTRA+=(--device "${DEVICE}")

echo "[main_pipeline] paradigm=${PARADIGM}  method=${METHOD}  ${EXTRA[*]:-}"
python runner.py --paradigm "${PARADIGM}" --method "${METHOD}" ${EXTRA[@]+"${EXTRA[@]}"}
