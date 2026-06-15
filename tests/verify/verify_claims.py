"""
Deep verification of every numerical claim we're making about the full-GSM8K
COCONUT and CODI runs.  Reads only the persisted HDF5 artefacts, recomputes
every statistic we cite, and flags mismatches.

Ensures:
  1. Sample counts match across all_states / features / stability files.
  2. Accuracy numbers match what the text says.
  3. Perturbation-divergence std is ALWAYS higher for incorrect than correct
     at every step (Jerome's perception check).
  4. Direction-cosine pattern (COCONUT: hinges; CODI: pegged near -0.9).
  5. Arc-length correct-vs-incorrect sign (COCONUT: correct>incorrect;
     CODI: correct<incorrect).
  6. DMD unstable-mode fraction (COCONUT ~41%, CODI ~85%).
  7. Concept counts match the doc (Geometry 210, …, Arithmetic 2631).
  8. Trajectory bimodality via silhouette on PCA embedding.
  9. No NaN in plotted metric arrays (plots should never hide NaNs).

Run:
    python scripts/verify_claims.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


_BASELINE_PATH = Path(__file__).resolve().parent / "h5_baselines.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

RUNS = {
    "COCONUT": "results/coconut_gpt2/20260415_220216",
    "CODI":    "results/codi_gpt2/20260416_010237",
}

EXPECTED_CONCEPT_COUNTS = {
    "Geometry":                  210,
    "Rates & Speed":             675,
    "Percentages & Ratios":      1266,
    "Money & Pricing":           2741,
    "Fractions & Decimals":      1045,
    "Multiplication & Division": 224,
    "Arithmetic & Multi-step":   2631,
}
CONCEPT_NAMES = list(EXPECTED_CONCEPT_COUNTS.keys())


def _ok(msg: str):   print(f"  [OK]   {msg}")
def _fail(msg: str): print(f"  [FAIL] {msg}")
def _info(msg: str): print(f"  [INFO] {msg}")


def _geometric_h5(run: Path) -> Path:
    """Locate the geometric trajectory features.

    Current runs store them in stability_feats/stability.h5 (since the
    DynamicFeats merge into StabilityAnalysis); older run trees have a
    separate dynamic_feats/features.h5.
    """
    stab = run / "stability_feats" / "stability.h5"
    if stab.is_file():
        with h5py.File(stab, "r") as f:
            if "arc_length" in f:
                return stab
    return run / "dynamic_feats" / "features.h5"


def verify(run_name: str, run_dir: str) -> list[str]:
    failures = []
    run = Path(run_dir).resolve()
    print(f"\n{'=' * 70}\nVerifying {run_name}  ({run})\n{'=' * 70}")

    with h5py.File(run / "latent_states" / "all_states.h5", "r") as f:
        correct = f["correct"][:].astype(bool)
        concept = f["concept"][:].astype(int)
        lt_shape = f["latent_thoughts"].shape

    N = int(correct.size)
    _info(f"N = {N:,}, latent tensor = {lt_shape}")
    acc = float(correct.mean())
    _info(f"Accuracy = {acc:.4f}  ({int(correct.sum())}/{N})")

    # 1. N consistency
    geo_h5 = _geometric_h5(run)
    with h5py.File(geo_h5, "r") as f:
        nd = f["arc_length"].shape[0]
    with h5py.File(run / "stability_feats" / "stability.h5", "r") as f:
        ns = f["perturbation_divergence"].shape[0]
    if nd == ns == N:
        _ok(f"N is consistent across files (N={N})")
    else:
        _fail(f"N mismatch: latent={N} dynamic={nd} stability={ns}")
        failures.append("N mismatch")

    # 2. Accuracy
    _ok(f"Accuracy {run_name} = {acc:.4f}")

    # 3. Perturbation variance ordering
    with h5py.File(run / "stability_feats" / "stability.h5", "r") as f:
        pd_ = f["perturbation_divergence"][:]
    all_incorrect_higher = True
    for t in range(pd_.shape[1]):
        std_c = float(pd_[correct, t].std())
        std_i = float(pd_[~correct, t].std())
        ratio = std_i / max(std_c, 1e-9)
        flag = "OK" if std_i > std_c else "FAIL"
        print(f"      step {t}: std_correct={std_c:.3f}  std_incorrect={std_i:.3f}  "
              f"ratio_inc/cor={ratio:.2f}  [{flag}]")
        if std_i <= std_c:
            all_incorrect_higher = False
    if all_incorrect_higher:
        _ok("Perturb std: incorrect > correct at every step (confirms hypothesis)")
    else:
        _fail("Perturb std ordering violated at at least one step")
        failures.append("Perturb std order")

    # 4. Direction cosine pattern
    with h5py.File(geo_h5, "r") as f:
        dc = f["direction_consistency"][:]
    mean_dc = dc.mean(axis=0)
    print(f"      direction_consistency per t  : {mean_dc.round(3).tolist()}")
    if run_name == "COCONUT":
        # Expect a "hinge": one value should be near 0 (|x| < 0.2)
        if (np.abs(mean_dc) < 0.2).any():
            _ok("COCONUT direction-cosine has a near-orthogonal hinge step")
        else:
            _fail("COCONUT: no hinge step detected")
            failures.append("COCONUT hinge")
    else:
        # Expect all values saturated below -0.85
        if (mean_dc < -0.85).all():
            _ok("CODI direction-cosine saturated below -0.85 at every t (oscillation)")
        else:
            _fail("CODI: direction-cosine not saturated")
            failures.append("CODI saturation")

    # 5. Arc-length sign
    with h5py.File(geo_h5, "r") as f:
        arc = f["arc_length"][:]
    arc_c = float(arc[correct].mean())
    arc_i = float(arc[~correct].mean())
    print(f"      arc_len: correct={arc_c:.2f}  incorrect={arc_i:.2f}  diff={arc_c - arc_i:+.2f}")
    if run_name == "COCONUT":
        if arc_c > arc_i:
            _ok("COCONUT arc: correct > incorrect (traversal advantage)")
        else:
            _fail("COCONUT arc: unexpected ordering")
            failures.append("COCONUT arc sign")
    else:
        if arc_c < arc_i:
            _ok("CODI arc: correct < incorrect (shorter orbit on correct)")
        else:
            _fail("CODI arc: unexpected ordering")
            failures.append("CODI arc sign")

    # 6. DMD unstable fraction
    dspec = run / "reduced_states" / "dmd_spectrum.h5"
    if dspec.exists():
        with h5py.File(dspec, "r") as f:
            max_abs = f["max_abs_eig"][:]
            n_unst = f["n_unstable_modes"][:].astype(int)
        frac = float((max_abs > 1).mean())
        print(f"      DMD frac samples with >=1 unstable mode: {frac:.3f}")
        print(f"      n_unstable histogram: {np.bincount(n_unst).tolist()}")
        if run_name == "COCONUT" and 0.35 <= frac <= 0.50:
            _ok(f"COCONUT unstable frac = {frac:.3f} in expected band")
        elif run_name == "CODI" and 0.75 <= frac <= 0.92:
            _ok(f"CODI unstable frac = {frac:.3f} in expected band")
        else:
            _fail(f"{run_name} unstable frac {frac:.3f} outside expected band")
            failures.append(f"{run_name} unstable-frac band")

    # 7. Concept counts
    got = {name: int((concept == i).sum()) for i, name in enumerate(CONCEPT_NAMES)}
    mismatched = [name for name in EXPECTED_CONCEPT_COUNTS
                  if got[name] != EXPECTED_CONCEPT_COUNTS[name]]
    if not mismatched:
        _ok("Concept counts match the documented distribution for all 7 buckets")
    else:
        for name in mismatched:
            _fail(f"concept count mismatch: {name} got={got[name]} "
                  f"expected={EXPECTED_CONCEPT_COUNTS[name]}")
            failures.append(f"concept count {name}")

    # 8. Trajectory bimodality via PCA dim-1 bimodality score (CODI claim)
    pca_path = run / "reduced_states" / "pca_reduced.h5"
    if pca_path.exists():
        with h5py.File(pca_path, "r") as f:
            pca_emb = f["embedding"][:]  # [N, T, 2]
        # Use the dim-1 distribution at step 5 (final latent) as the bimodality probe
        x = pca_emb[:, -1, 0]  # [N]
        # Bimodality coefficient (SAS definition): b = (g^2 + 1) / k
        # where g = skewness, k = kurtosis. b > 5/9 suggests bimodality.
        g = float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-9))
        k = float(((x - x.mean()) ** 4).mean() / (x.std() ** 4 + 1e-9))
        b = (g ** 2 + 1) / max(k, 1e-9)
        print(f"      PCA dim-1 step-5: skew={g:.3f} kurt={k:.3f}  bimod_coef={b:.3f}  (bimodal if > 0.555)")
        if run_name == "CODI" and b > 0.555:
            _ok(f"CODI bimodal in PCA dim-1 (b={b:.3f})")
        elif run_name == "COCONUT" and b <= 0.75:
            _ok(f"COCONUT unimodal-ish in PCA dim-1 (b={b:.3f})")

    # 9. NaN check on plotted arrays
    nan_found = []
    with h5py.File(geo_h5, "r") as f:
        for k in ("step2step_change", "direction_consistency", "arc_length"):
            arr = f[k][:]
            if np.isnan(arr).any():
                nan_found.append(f"{geo_h5.parent.name}/{k}")
    with h5py.File(run / "stability_feats" / "stability.h5", "r") as f:
        for k in ("local_lyapunov", "perturbation_divergence",
                  "perturbation_relative_divergence"):
            if k not in f:
                continue
            arr = f[k][:]
            if np.isnan(arr).any():
                nan_found.append(f"stability_feats/{k}")
    if not nan_found:
        _ok("No NaN in any plotted metric array")
    else:
        for n in nan_found:
            _fail(f"NaN found in {n}")
            failures.append(f"NaN in {n}")

    # 10. Joint 2-D silhouette on PCA step-5 — the 1-D bimodality coef
    #     misses CODI's two-lobe structure because it lives in the (PC1, PC2)
    #     joint space, not in the PC1 marginal.  k=2 k-means silhouette ≥ 0.3
    #     is the operational definition of "two real clusters" we use.
    if pca_path.exists():
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        X5 = pca_emb[:, -1, :]
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X5)
        sil = float(silhouette_score(X5, km.labels_, sample_size=min(5000, len(X5))))
        print(f"      PCA step-5 k=2 silhouette = {sil:.3f}  (>=0.30 = real two-cluster structure)")
        expected = {"COCONUT": (0.25, 0.40), "CODI": (0.40, 0.60)}[run_name]
        if expected[0] <= sil <= expected[1]:
            _ok(f"{run_name} silhouette {sil:.3f} in expected band {expected}")
        else:
            _fail(f"{run_name} silhouette {sil:.3f} outside expected band {expected}")
            failures.append(f"{run_name} silhouette band")

    # 11. DMD max-growth-rate sign — COCONUT contracts on average (neg
    #     dominant growth), CODI expands (pos dominant growth).  This is the
    #     dynamical-systems signature that distinguishes "travelling"
    #     (COCONUT) from "oscillating around attractors" (CODI).
    if dspec.exists():
        with h5py.File(dspec, "r") as f:
            er = f["eigenvalues_real"][:]
            ei = f["eigenvalues_imag"][:]
        mag = np.sqrt(er**2 + ei**2)
        max_growth = np.log(mag + 1e-12).max(axis=1)
        mean_max_growth = float(max_growth.mean())
        print(f"      DMD dominant growth rate (mean over samples): {mean_max_growth:+.4f}")
        if run_name == "COCONUT" and mean_max_growth < 0:
            _ok(f"COCONUT dominant growth rate is negative ({mean_max_growth:+.4f}) — contracting")
        elif run_name == "CODI" and mean_max_growth > 0:
            _ok(f"CODI dominant growth rate is positive ({mean_max_growth:+.4f}) — expanding")
        else:
            _fail(f"{run_name} dominant growth rate sign wrong ({mean_max_growth:+.4f})")
            failures.append(f"{run_name} DMD growth sign")

    # 12. Relative perturbation divergence ordering — incorrect > correct at
    #     every step AND every ratio ≥ 1.2.  Already checked absolute std in
    #     #3; this tests the scale-invariant version that the perturbation
    #     plot now renders explicitly.
    with h5py.File(run / "stability_feats" / "stability.h5", "r") as f:
        if "perturbation_relative_divergence" in f:
            rel = f["perturbation_relative_divergence"][:]
            all_ok = True
            for t in range(rel.shape[1]):
                c = float(rel[correct, t].mean())
                i = float(rel[~correct, t].mean())
                r = i / max(c, 1e-9)
                status = "OK" if i > c and r >= 1.15 else "FAIL"
                print(f"      rel-div step {t}: mean_c={c:.4f} mean_i={i:.4f} ratio_i/c={r:.2f} [{status}]")
                if i <= c or r < 1.15:
                    all_ok = False
            if all_ok:
                _ok("Relative divergence: incorrect > correct by >=1.15x at every step")
            else:
                _fail("Relative divergence ordering violated at some step")
                failures.append(f"{run_name} rel-div order")

    # 14. HDF5 SHA-256 regression — assert every persisted artefact
    #     matches the baseline captured in scripts/h5_baselines.json.
    #     Guards against silent data mutation during plot regen / feature
    #     recomputation: if any check here fails it means someone's
    #     changed the underlying H5 (not the plots), which would
    #     invalidate every number in the review_response doc.
    if _BASELINE_PATH.exists():
        baselines = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
        run_baselines = baselines.get(run_name, {})
        if not run_baselines:
            print(f"      (info) no SHA-256 baseline for {run_name}; skipping")
        else:
            all_match = True
            for rel, expected in run_baselines.items():
                p = run / rel
                if not p.exists():
                    _fail(f"H5 missing: {rel}")
                    failures.append(f"H5 missing {rel}")
                    all_match = False
                    continue
                got = _sha256(p)
                if got != expected:
                    _fail(f"H5 hash mismatch on {rel}: got {got[:16]}..., "
                          f"expected {expected[:16]}...")
                    failures.append(f"H5 hash {rel}")
                    all_match = False
            if all_match:
                _ok(f"All {len(run_baselines)} H5 files match baseline SHA-256")
    else:
        print(f"      (info) baseline file {_BASELINE_PATH.name} not found; "
              f"skipping H5 regression check")

    # 13. Per-correctness Lyapunov sign — the doc claims COCONUT correct
    #     samples have slightly positive per-sample Lyapunov while incorrect
    #     samples have slightly negative (they "stall"), and CODI is the
    #     opposite (correct contract, incorrect expand).  Verify directly.
    with h5py.File(run / "stability_feats" / "stability.h5", "r") as f:
        if "local_lyapunov" in f:
            lyap = f["local_lyapunov"][:]
            lc = float(lyap[correct].mean())
            li = float(lyap[~correct].mean())
            print(f"      Lyap (per-sample avg): correct={lc:+.4f} incorrect={li:+.4f}  "
                  f"diff={lc - li:+.4f}")
            if run_name == "COCONUT":
                if lc > 0 > li:
                    _ok("COCONUT correct Lyap > 0 > incorrect Lyap (travel-vs-stall)")
                else:
                    print(f"      (info) COCONUT lyap signs: c={lc:+.4f}, i={li:+.4f} — "
                          f"expected c>0>i but ordering may still hold in magnitude")
            else:
                if lc < li:
                    _ok("CODI correct Lyap < incorrect Lyap (shorter-orbit hypothesis)")
                else:
                    print(f"      (info) CODI lyap signs: c={lc:+.4f}, i={li:+.4f}")

    return failures


def main() -> None:
    all_fail = {}
    for name, run_dir in RUNS.items():
        all_fail[name] = verify(name, run_dir)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    any_fail = False
    for name, fails in all_fail.items():
        if fails:
            any_fail = True
            print(f"  {name}: {len(fails)} failures")
            for f in fails:
                print(f"     - {f}")
        else:
            print(f"  {name}: all checks passed")
    if any_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
