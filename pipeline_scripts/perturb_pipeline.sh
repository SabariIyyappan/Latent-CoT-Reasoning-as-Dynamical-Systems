#!/usr/bin/env bash
# perturb_pipeline.sh
# ----------------------------------------------------------------------
# Set up the environment and run the PERTURBATION analysis for one cell.
# Reuses the cell's cached H5 latent representations as the clean
# baseline (preparing them first if none exist), then re-runs the model
# with Gaussian noise on the input embeddings and writes the divergence
# outputs. Same five-step layout as main_pipeline.sh.
#
# Run one cell (defaults to vanilla + codi):
#     bash pipeline_scripts/perturb_pipeline.sh
#
# Pick a different cell / noise level:
#     PARADIGM=simcot METHOD=codi NOISE_STD=0.1 bash pipeline_scripts/perturb_pipeline.sh
#
# Quick smoke run (cap samples, CPU) before committing to a full run:
#     MAX_SAMPLES=3 DEVICE=cpu bash pipeline_scripts/perturb_pipeline.sh
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

# ── 4. Choose the method, paradigm, and noise level to run ────────────
# Method   : codi | coconut
# Paradigm : vanilla | simcot
# Noise    : Gaussian sigma on the input embeddings (paper uses 0.01)
#
# The four available combinations (set PARADIGM and METHOD to pick one):
#     PARADIGM=vanilla METHOD=codi
#     PARADIGM=vanilla METHOD=coconut
#     PARADIGM=simcot  METHOD=codi
#     PARADIGM=simcot  METHOD=coconut
PARADIGM="${PARADIGM:-vanilla}"
METHOD="${METHOD:-codi}"
NOISE_STD="${NOISE_STD:-0.01}"
# Optional smoke-test knobs — leave empty for the full run.
DEVICE="${DEVICE:-}"          # cpu | cuda
MAX_SAMPLES="${MAX_SAMPLES:-}" # cap the perturbed questions
N_SAMPLES="${N_SAMPLES:-}"    # cap the prep run if the H5 has to be built first

# ── 5. Run the perturbation pipeline ──────────────────────────────────
# Find the newest cached run for this cell; if none exists, prepare the
# H5 latent representations first (inference + analysis), then perturb.
latest_run_with_h5() {
  ls -d "results/${PARADIGM}_${METHOD}"/*/ 2>/dev/null \
    | while read -r d; do
        [[ -f "${d}latent_states/all_states.h5" ]] && echo "${d%/}"
      done \
    | sort | tail -n1
}

RUN_DIR="$(latest_run_with_h5)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "[perturb_pipeline] no cached H5 for ${PARADIGM}_${METHOD} — preparing it first"
  PREP=()
  [[ -n "${N_SAMPLES}" ]] && PREP+=(--n_samples "${N_SAMPLES}")
  [[ -n "${DEVICE}" ]] && PREP+=(--device "${DEVICE}")
  python runner.py --paradigm "${PARADIGM}" --method "${METHOD}" ${PREP[@]+"${PREP[@]}"}
  RUN_DIR="$(latest_run_with_h5)"
fi

EXTRA=()
[[ -n "${MAX_SAMPLES}" ]] && EXTRA+=(--max_samples "${MAX_SAMPLES}")
[[ -n "${DEVICE}" ]] && EXTRA+=(--device "${DEVICE}")

echo "[perturb_pipeline] paradigm=${PARADIGM}  method=${METHOD}  noise_std=${NOISE_STD}  ${EXTRA[*]:-}"
echo "[perturb_pipeline] run_dir=${RUN_DIR}"
python runner.py --paradigm "${PARADIGM}" --method "${METHOD}" \
    --perturbation --run_dir "${RUN_DIR}" --noise_std "${NOISE_STD}" ${EXTRA[@]+"${EXTRA[@]}"}
