"""
Generic perturbation_divergence driver — works against any LatentGenerationMixin
checkpoint, at any noise_std, and writes sigma-suffixed outputs so multiple
runs coexist in the same result tree.

Why this exists alongside add_perturbation_simcot.py
----------------------------------------------------
add_perturbation_simcot.py was wired specifically for the SimCoT bootstrap
checkpoints (sigma=0.01, output names without suffix). This generalised
sibling adds:

  --noise_std        any sigma value
  --filename_suffix  appended to PNG name so e.g. sigma=1.0 lands at
                     plots/perturbation_divergence_sigma1.0.png and does
                     not collide with the existing sigma=0.01 figure
  --stability_key_suffix
                     appended to keys merged into stability.h5 so e.g.
                     "perturbation_divergence_sigma1.0" coexists with the
                     prior "perturbation_divergence" datasets

The clean baseline is always re-inferred via the supplied checkpoint so the
divergence reflects that exact model's response to noise (no path-A vs
path-C model drift). Stratified subsample defaults to 500 across the 7
GSM8K concept buckets — runs in ~20-40 min per call on CPU, viable for
the full 4-config matrix (2 methods × 2 sigmas).

Usage
-----
    python scripts/add_perturbation.py \\
        --method coconut --checkpoint checkpoints/coconut \\
        --out_dir results/coconut_gpt2/<run_dir> \\
        --n_samples 500 --noise_std 1.0 \\
        --filename_suffix _sigma1.0 \\
        --stability_key_suffix _sigma1.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import json
import math
import re
import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.wrappers import ModelWrapper
from analysis.stability import StabilityAnalysis
from analysis.plotting import Plotting
from analysis.data_prep import GSM8K_CONCEPT_NAMES, _CONCEPT_KEYWORDS


def _classify_concept(question: str) -> int:
    """Lift of analysis.data_prep._classify_concept_text — keep here so this
    script is self-contained when the HDF5 doesn't ship a 'concept' field."""
    t = question.lower()
    for cid, kws in enumerate(_CONCEPT_KEYWORDS):
        for kw in kws:
            if kw in t:
                return cid
    return 6


def _parse_gold(answer_text: str):
    """Pull the numeric ground-truth answer out of a GSM8K answer string."""
    if "####" in answer_text:
        ans = answer_text.split("####")[-1].strip()
    else:
        ans = answer_text.strip()
    ans = ans.replace(",", "")
    try:
        return float(ans)
    except (ValueError, TypeError):
        return None


def _extract_predicted_number(text: str):
    text = text.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except (ValueError, TypeError):
        return None


def _stratified_indices(
    concept: np.ndarray,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Pick n_samples indices balanced across the concept buckets present.

    Each bucket gets floor(n / k) plus one extra for the first (n % k)
    buckets, drawn at random without replacement. If a bucket has fewer
    samples than its quota, all of its samples are kept and the deficit is
    redistributed to other buckets.
    """
    rng = np.random.default_rng(seed)
    buckets = sorted(np.unique(concept).tolist())
    k = len(buckets)
    base = n_samples // k
    extra = n_samples % k
    chosen: List[int] = []
    leftover_quota = 0
    for i, b in enumerate(buckets):
        quota = base + (1 if i < extra else 0) + leftover_quota
        leftover_quota = 0
        idx = np.where(concept == b)[0]
        if len(idx) <= quota:
            chosen.extend(idx.tolist())
            leftover_quota = quota - len(idx)
        else:
            pick = rng.choice(idx, size=quota, replace=False)
            chosen.extend(pick.tolist())
    chosen = sorted(chosen)[:n_samples]
    return np.asarray(chosen, dtype=np.int64)


def _load_meta_from_hdf5(run_dir: Path) -> Tuple[List[str], np.ndarray, np.ndarray, list]:
    """Load (questions, correct, concept, gold) from the run's all_states.h5.

    Older initial-run HDF5s may not store the question text — in that case
    the caller falls back to _load_meta_from_json.
    """
    h5p = run_dir / "latent_states" / "all_states.h5"
    if not h5p.is_file():
        raise FileNotFoundError(f"Missing {h5p}")
    with h5py.File(str(h5p), "r") as f:
        if "question" not in f:
            raise KeyError(f"{h5p} has no 'question' field; use --questions_json")
        questions = [q.decode() if isinstance(q, bytes) else str(q)
                     for q in f["question"][:]]
        correct = f["correct"][:].astype(bool)
        concept = f["concept"][:].astype(np.int32)
    return questions, correct, concept, [None] * len(questions)


def _load_meta_from_json(json_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, list]:
    """Load (questions, [empty correct], concept, gold) from the GSM8K JSON
    used by the literal extract path. Correctness will be computed on the
    fly during inference (clean answer parsed and compared to gold).
    """
    with json_path.open(encoding="utf-8") as f:
        records = json.load(f)
    questions = [r["question"] for r in records]
    concept = np.array([_classify_concept(q) for q in questions], dtype=np.int32)
    gold = [_parse_gold(r["answer"]) for r in records]
    correct = np.zeros(len(questions), dtype=bool)  # filled later from clean run
    return questions, correct, concept, gold


def _merge_into_stability_h5(
    out_dir: Path, result: dict, key_suffix: str = "",
) -> None:
    """Add the perturbation_* arrays to the existing stability.h5 in place
    without touching pre-existing datasets. When key_suffix is non-empty
    each key is renamed (e.g. perturbation_divergence_sigma1.0) so that
    several sigma runs coexist."""
    p = out_dir / "stability_feats" / "stability.h5"
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(p), "a") as f:
        for k, v in result.items():
            target = f"{k}{key_suffix}"
            if target in f:
                del f[target]
            f.create_dataset(target, data=v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True,
                        choices=["coconut", "codi", "codi_math_distilled",
                                 "coconut_math_finetuned"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="results/<method>_gpt2/<run_dir>/")
    parser.add_argument("--questions_json", type=Path, default=None,
                        help="Optional path to data/gsm8k_for_literal/"
                             "gsm8k_train_test.json. When provided, questions "
                             "are loaded from JSON rather than from <out_dir>/"
                             "latent_states/all_states.h5 (used when the HDF5 "
                             "doesn't store question text). Correctness is "
                             "computed on the fly from clean inference.")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Stratified subsample size (default 500). Pass 0 "
                             "to use the full dataset.")
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--n_perturbations", type=int, default=3)
    parser.add_argument("--filename_suffix", default="",
                        help="Appended to plot filename (e.g. _sigma1.0)")
    parser.add_argument("--stability_key_suffix", default="",
                        help="Appended to stability.h5 dataset keys "
                             "(e.g. _sigma1.0)")
    parser.add_argument("--device", default=None,
                        help="cpu | cuda | auto. Default: cuda if available.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[setup] device = {device}")

    # ── Load metadata + stratify ─────────────────────────────────────────
    if args.questions_json is not None:
        print(f"[data] reading questions from {args.questions_json}")
        questions, correct, concept, gold = _load_meta_from_json(args.questions_json)
        recompute_correct = True
    else:
        try:
            print(f"[data] reading {args.out_dir}/latent_states/all_states.h5")
            questions, correct, concept, gold = _load_meta_from_hdf5(args.out_dir)
            recompute_correct = False
        except (FileNotFoundError, KeyError) as e:
            print(f"[data] HDF5 unusable ({e}); falling back to default JSON")
            default_json = PROJECT_ROOT / "data" / "gsm8k_for_literal" / "gsm8k_train_test.json"
            questions, correct, concept, gold = _load_meta_from_json(default_json)
            recompute_correct = True
    n_total = len(questions)
    print(f"[data] total samples = {n_total}; recompute_correct={recompute_correct}")

    if args.n_samples <= 0 or args.n_samples >= n_total:
        idx = np.arange(n_total, dtype=np.int64)
        print(f"[data] using all {n_total} samples")
    else:
        idx = _stratified_indices(concept, args.n_samples, seed=args.seed)
        print(f"[data] stratified subsample size = {len(idx)}")
        per_concept = {GSM8K_CONCEPT_NAMES[c]: int((concept[idx] == c).sum())
                       for c in sorted(np.unique(concept[idx]).tolist())
                       if c < len(GSM8K_CONCEPT_NAMES)}
        print(f"[data] per-concept counts: {per_concept}")

    sub_questions = [questions[i] for i in idx]
    sub_correct = correct[idx].copy()
    sub_gold = [gold[i] for i in idx]

    # ── Load checkpoint ──────────────────────────────────────────────────
    print(f"[setup] loading {args.method} from {args.checkpoint}...")
    t0 = time.time()
    wrapper = ModelWrapper(
        method=args.method,
        checkpoint=str(args.checkpoint),
        device=device,
    )
    print(f"[setup] loaded in {time.time()-t0:.1f}s")

    # ── Optional: re-compute correctness on the subsample via clean run ──
    if recompute_correct:
        print("[recompute] running clean inference on subsample to get correctness...")
        t0 = time.time()
        for i, q in enumerate(sub_questions):
            try:
                out = wrapper.run_inference(q)
                tok = wrapper.tokenizer
                # Decode generated tokens; the wrapper output contains the
                # full sequence including the prompt. Strip prompt by length.
                seq = out.sequences[0] if hasattr(out, "sequences") else out
                if torch.is_tensor(seq):
                    decoded = tok.decode(seq.tolist(), skip_special_tokens=True)
                else:
                    decoded = str(seq)
                pred = _extract_predicted_number(decoded)
                g = sub_gold[i]
                if g is not None and pred is not None and not math.isinf(pred):
                    sub_correct[i] = abs(pred - g) < 1e-3
            except Exception as exc:
                print(f"  [recompute] q{i} failed: {type(exc).__name__}: {exc}")
            if (i + 1) % 50 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"  [recompute] {i+1}/{len(sub_questions)} done  "
                      f"acc={100*sub_correct[:i+1].mean():.1f}%  rate={rate:.2f}/s")
        print(f"[recompute] done in {(time.time()-t0)/60:.1f} min  "
              f"final acc={100*sub_correct.mean():.2f}%")

    # ── Run perturbation ─────────────────────────────────────────────────
    sa = StabilityAnalysis(model_wrapper=wrapper, questions=sub_questions)
    print(
        f"[perturb] running on N={len(sub_questions)} questions  "
        f"noise_std={args.noise_std}  n_perturbations={args.n_perturbations}..."
    )
    t0 = time.time()
    result = sa.calc_perturbation_stability(
        noise_std=args.noise_std,
        n_perturbations=args.n_perturbations,
        seed=args.seed,
        # clean_states intentionally None: re-infer clean baseline using
        # the supplied checkpoint so divergence reflects pure noise
        # response of THIS model.
    )
    elapsed = time.time() - t0
    print(
        f"[perturb] done in {elapsed/60:.1f} min  "
        f"({elapsed/max(len(sub_questions),1):.2f} s / question)"
    )

    div = result["perturbation_divergence"]
    div_std = result["perturbation_divergence_std"]
    rel = result["perturbation_relative_divergence"]
    n_eff = int(np.isfinite(div).any(axis=1).sum())
    print(
        f"[perturb] divergence shape={div.shape}  "
        f"finite-rows={n_eff}/{len(sub_questions)}  "
        f"step-means={np.nanmean(div, axis=0)}"
    )

    # ── Persist into stability.h5 ────────────────────────────────────────
    _merge_into_stability_h5(args.out_dir, result, key_suffix=args.stability_key_suffix)
    # Also write the index map AND the recomputed correctness mask (suffixed)
    # so downstream re-plot scripts can recover the correct/incorrect split
    # without re-running clean inference.
    with h5py.File(str(args.out_dir / "stability_feats" / "stability.h5"), "a") as f:
        idx_key = f"perturbation_subsample_index{args.stability_key_suffix}"
        if idx_key in f:
            del f[idx_key]
        f.create_dataset(idx_key, data=idx)
        mask_key = f"perturbation_correct_mask{args.stability_key_suffix}"
        if mask_key in f:
            del f[mask_key]
        f.create_dataset(mask_key, data=sub_correct)
    print(f"[save] merged perturbation_*{args.stability_key_suffix} "
          f"into {args.out_dir}/stability_feats/stability.h5")

    # ── Plot ─────────────────────────────────────────────────────────────
    plotter = Plotting(str(args.out_dir))
    plotter.make_perturbation_divergence_plot(
        divergence=div,
        divergence_std=div_std,
        correct_mask=sub_correct,
        relative_divergence=rel,
        noise_std=args.noise_std,
        n_perturbations=args.n_perturbations,
        filename_suffix=args.filename_suffix,
    )
    print(f"[plot] saved {args.out_dir}/plots/"
          f"perturbation_divergence{args.filename_suffix}.png")


if __name__ == "__main__":
    main()
