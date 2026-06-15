"""
Data preparation and method execution.

Loads GSM8K, MATH (EleutherAI/hendrycks_math), or MATH-500
(HuggingFaceH4/MATH-500) dataset, runs latent reasoning models, and saves
extracted latent states in HDF5 format.

Supports stratified sampling across concept buckets so that
downstream concept-wise analyses have balanced group sizes.

Labels are the 7 keyword-classified concept buckets defined by
``_CONCEPT_KEYWORDS`` / ``GSM8K_CONCEPT_NAMES`` (GSM8K), or the 7
subject fields from ``MATH_CONCEPT_NAMES`` (MATH).
"""

import re
from pathlib import Path
from typing import Optional, Dict, List, Any

import h5py
import numpy as np
import torch
import datasets

from .wrappers import ModelWrapper, PromptTooLongError


# ── GSM8K concept taxonomy ────────────────────────────────────────────────────

GSM8K_CONCEPT_NAMES: List[str] = [
    "Geometry",
    "Rates & Speed",
    "Percentages & Ratios",
    "Money & Pricing",
    "Fractions & Decimals",
    "Multiplication & Division",
    "Arithmetic & Multi-step",
]

# Backward-compat alias (imported by runner.py and older code paths).
CONCEPT_NAMES = GSM8K_CONCEPT_NAMES

# Priority-ordered keyword lists (first match wins).
# More specific categories are checked before generic ones to prevent
# e.g. a geometry problem that mentions "cost" being mis-labelled as Money.
_CONCEPT_KEYWORDS: List[List[str]] = [
    # 0 — Geometry
    ["area", "perimeter", "volume", "rectangle", "triangle", "circle",
     "square", "radius", "diameter", "polygon", "parallelogram", "trapezoid",
     "circumference", "hypotenuse", "diagonal"],
    # 1 — Rates & Speed
    ["speed", "mph", "per hour", "per minute", "per second", "distance",
     "miles", "kilometers", "km", "faster", "slower", "velocity",
     "traveled", "travelling", "travels"],
    # 2 — Percentages & Ratios
    ["percent", "%", "ratio", "proportion", "discount", "tax",
     "markup", "interest rate", "out of 100", "probability"],
    # 3 — Money & Pricing
    ["dollar", "cent", "cost", "price", "pay", "paid", "buy", "bought",
     "sell", "sold", "earn", "earned", "profit", "wage", "salary",
     "charge", "fee", "budget", "spend", "spent", "affordable", "expensive",
     "cheap", "refund", "receipt", "purchase"],
    # 4 — Fractions & Decimals
    ["fraction", "decimal", "half", "halves", "third", "quarter",
     "1/2", "1/3", "1/4", "2/3", "3/4", "numerator", "denominator"],
    # 5 — Multiplication & Division
    ["multiply", "multiplied", "times as many", "product of",
     "divide", "divided", "divisible", "quotient", "factor of"],
    # 6 — Arithmetic & Multi-step  (catch-all — no keywords needed)
]


# ── MATH-500 concept taxonomy ─────────────────────────────────────────────────

MATH500_CONCEPT_NAMES: List[str] = [
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
]

_MATH500_SUBJECT_TO_ID: Dict[str, int] = {
    "Algebra": 0,
    "Counting & Probability": 1,
    "Geometry": 2,
    "Intermediate Algebra": 3,
    "Number Theory": 4,
    "Prealgebra": 5,
    "Precalculus": 6,
}

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _extract_boxed_answer(solution: str) -> Optional[float]:
    """
    Extract the final \\boxed{...} answer from a MATH solution string and
    try to parse it as a float.  Returns None if extraction or parsing fails.
    """
    matches = _BOXED_RE.findall(solution or "")
    if not matches:
        return None
    raw = matches[-1].strip()
    # Strip common LaTeX formatting before float-parsing
    raw = (
        raw.replace("\\,", "").replace("\\!", "")
           .replace(",", "").replace(" ", "")
           .replace("\\frac", "").replace("{", "").replace("}", "")
    )
    try:
        return float(raw)
    except ValueError:
        return None


def classify_concept(question: str) -> int:
    """
    Classify a GSM8K question into one of 7 math concept buckets.

    Uses priority-ordered keyword matching — the first bucket whose
    keyword list contains a substring of the lowercased question wins.
    If nothing matches, returns 6 (Arithmetic & Multi-step, catch-all).

    Priority order (0 → 5, then catch-all 6):
        0 Geometry
        1 Rates & Speed
        2 Percentages & Ratios
        3 Money & Pricing
        4 Fractions & Decimals
        5 Multiplication & Division
        6 Arithmetic & Multi-step  (no keywords; catch-all)

    The priority is intentional: a geometry problem that also mentions
    cost should be labelled Geometry, not Money.

    Args:
        question: Raw question string from GSM8K.

    Returns:
        Integer concept ID in [0, 6].

    Worked example::

        >>> classify_concept("A rectangle costs $5 per square foot. "
        ...                  "What is the total cost for a 12 sq-ft room?")
        0  # Geometry (keyword "rectangle" hits in bucket 0 before
           #           "cost" could hit in bucket 3)

        >>> classify_concept("A car travels 60 miles per hour for 3 hours.")
        1  # Rates & Speed ("per hour" in bucket 1)

        >>> classify_concept("John bought 3 apples for $2 each. "
        ...                  "How much did he spend?")
        3  # Money & Pricing ("bought", "cost"-style words in bucket 3)

        >>> classify_concept("Sarah has 5 books and gives 2 away. "
        ...                  "How many does she have left?")
        6  # Catch-all (no keyword hits)

    See also:
        docs/concept_bucketing.md — full keyword lists, empirical
        distribution over the 8,792-row GSM8K train+test split, and
        known limitations (substring false positives, surface-lexicon
        vs problem-structure, English-only).
    """
    q = question.lower()
    for concept_id, keywords in enumerate(_CONCEPT_KEYWORDS):
        if any(kw in q for kw in keywords):
            return concept_id
    return 6  # Arithmetic & Multi-step fallback


# ── MATH concept taxonomy ─────────────────────────────────────────────────────

MATH_CONCEPT_NAMES: List[str] = [
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
]
_MATH_SUBJECT_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(MATH_CONCEPT_NAMES)}

MATH_DIFFICULTY_NAMES: List[str] = [
    "Level 1",
    "Level 2",
    "Level 3",
    "Level 4",
    "Level 5",
]


def _parse_math_level(level_str: str) -> int:
    """Parse 'Level N' string to 0-indexed int [0, 4]. Defaults to 2 on failure."""
    try:
        return max(0, min(4, int(str(level_str).replace("Level", "").strip()) - 1))
    except (ValueError, AttributeError):
        return 2


def _extract_boxed_answer(solution: str) -> str:
    """Extract the argument of the last \\boxed{...} macro, handling nested braces."""
    if not solution:
        return ""
    marker = r"\boxed{"
    start = solution.rfind(marker)
    if start < 0:
        return solution
    i = start + len(marker)
    depth = 1
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1
    return solution[start + len(marker): i - 1]


def eval_latex_expr(s: str) -> Optional[float]:
    """
    Evaluate a LaTeX math expression to float without external dependencies.

    Handles (in priority order):
      - Plain integers / decimals: ``16``, ``-3.14``
      - Simple fractions: ``\\frac{a}{b}``, ``\\dfrac{a}{b}``
      - Negative fractions: ``-\\frac{1}{2}``
      - Simple radicals: ``\\sqrt{n}``
      - Pi: ``\\pi``
      - Fallback: last signed number in the string (matches prior behaviour)

    This is the single source-of-truth evaluator used for both ground-truth
    and predicted answers so that comparisons are symmetric.
    """
    import math as _math
    if not s:
        return None
    s = s.strip()

    # 1. Plain number (fast path).
    # Do NOT strip commas here: "4,6,14,15" is a set of answers, not a
    # thousand-formatted integer.  The caller (extract_answer) already strips
    # commas from model output before calling us, so the predicted side is
    # handled correctly without losing that context.
    try:
        return float(s)
    except (ValueError, TypeError):
        pass

    # 2. Leading minus — strip, evaluate the rest, negate
    sign = 1.0
    rest = s
    if rest.startswith("-"):
        sign = -1.0
        rest = rest[1:].strip()

    # 3. \frac{a}{b} or \dfrac{a}{b}
    frac_m = re.fullmatch(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", rest)
    if frac_m:
        num = eval_latex_expr(frac_m.group(1))
        den = eval_latex_expr(frac_m.group(2))
        if num is not None and den is not None and den != 0:
            return sign * num / den

    # 4. \sqrt{n}
    sqrt_m = re.fullmatch(r"\\sqrt\{([^}]+)\}", rest)
    if sqrt_m:
        inner = eval_latex_expr(sqrt_m.group(1))
        if inner is not None and inner >= 0:
            return sign * _math.sqrt(inner)

    # 5. \pi
    if rest in (r"\pi", "pi"):
        return sign * _math.pi

    # 6. Fallback: last signed number in the original string
    numbers = re.findall(r"-?\d+\.?\d*", s)
    if numbers:
        try:
            return float(numbers[-1])
        except (ValueError, TypeError):
            pass

    return None


def _parse_latex_number(answer_str: str) -> Optional[float]:
    """Extract a numeric value from a LaTeX answer string."""
    return eval_latex_expr(answer_str)


# ── DataPrep class ────────────────────────────────────────────────────────────

class DataPrep:
    """
    Prepares data and executes latent reasoning methods.

    Loads the specified dataset, runs inference through the model wrapper,
    and saves latent thought trajectories for downstream analysis.

    When stratified=True (default), samples are drawn evenly across the
    7 math concept buckets defined by classify_concept(), giving
    floor(n_samples / 7) or ceil(n_samples / 7) samples per concept.
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        dataset_name: str = "gsm8k",
        split: str = "test",
        n_samples: int = 500,
        stratified: bool = True,
        seed: int = 42,
    ):
        self.wrapper = model_wrapper
        self.dataset_name = dataset_name
        self.split = split
        self.n_samples = n_samples
        self.stratified = stratified
        self.seed = seed
        self.dataset = None
        self.results: List[Dict[str, Any]] = []

    # ── Label accessors ───────────────────────────────────────────────────────

    def get_label_names(self) -> List[str]:
        """Return the 7 concept-bucket names for the active dataset."""
        if self.dataset_name == "math":
            return MATH_CONCEPT_NAMES
        return GSM8K_CONCEPT_NAMES

    def _extract_concept_ids(self, ds: datasets.Dataset) -> np.ndarray:
        """Return [len(ds)] array of integer concept IDs."""
        if self.dataset_name == "math":
            # concept_subject is added by load_dataset() via _parse_math_row
            return np.array(ds["concept_subject"], dtype=np.int32)
        return np.array(
            [classify_concept(q) for q in ds["question"]],
            dtype=np.int32,
        )

    # ── Dataset loading ───────────────────────────────────────────────────────

    def load_dataset(self) -> datasets.Dataset:
        """Load and (optionally stratified-) subsample the dataset."""
        if self.dataset_name == "gsm8k":
            ds = datasets.load_dataset("gsm8k", "main", split=self.split)
            ds = ds.map(
                lambda x: {
                    "answer_num": float(
                        x["answer"].split("####")[-1].replace(",", "").strip()
                    )
                }
            )

        elif self.dataset_name == "math500":
            # HuggingFaceH4/MATH-500 — exactly 500 problems, no subsampling needed.
            # Fields: 'problem' (question), 'solution' (full work with \boxed{…}),
            #         'answer' (extracted string), 'subject' (category), 'level'.
            ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split=self.split)

            # Normalise field names
            if "problem" in ds.column_names:
                ds = ds.rename_column("problem", "question")
            subject_col = "subject" if "subject" in ds.column_names else "type"

            def _parse_math500_row(x):
                # Prefer pre-extracted 'answer' field; fall back to \boxed{} parse
                raw_ans = x.get("answer", None)
                if raw_ans:
                    ans_num = _extract_boxed_answer(f"\\boxed{{{raw_ans}}}")
                else:
                    ans_num = _extract_boxed_answer(x.get("solution", ""))
                return {
                    "answer_num": ans_num if ans_num is not None else float("nan"),
                    "concept_subject": _MATH500_SUBJECT_TO_ID.get(
                        x[subject_col], 0
                    ),
                }

            ds = ds.map(_parse_math500_row)

        elif self.dataset_name == "math":
            # EleutherAI/hendrycks_math — full MATH dataset (~12.5K total)
            # loaded as 7 separate subject configs then concatenated.
            # Fields: 'problem' (question), 'solution' (full work with \boxed{…}),
            #         'level' ("Level 1"–"Level 5"), 'type' (subject name).
            _MATH_SUBJECTS = [
                "algebra",
                "counting_and_probability",
                "geometry",
                "intermediate_algebra",
                "number_theory",
                "prealgebra",
                "precalculus",
            ]
            _splits = ["train", "test"] if self.split == "all" else [self.split]
            subject_splits = [
                datasets.load_dataset("EleutherAI/hendrycks_math", subj, split=sp)
                for subj in _MATH_SUBJECTS
                for sp in _splits
            ]
            ds = datasets.concatenate_datasets(subject_splits)

            if "problem" in ds.column_names:
                ds = ds.rename_column("problem", "question")

            def _parse_math_row(x):
                ans_str = _extract_boxed_answer(x.get("solution", ""))
                ans_num = _parse_latex_number(ans_str)
                return {
                    "answer_num": ans_num if ans_num is not None else float("nan"),
                    "concept_subject": _MATH_SUBJECT_TO_ID.get(x.get("type", ""), 0),
                    "difficulty": _parse_math_level(x.get("level", "Level 3")),
                }

            ds = ds.map(_parse_math_row)

        else:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name!r}. "
                f"Choose 'gsm8k' | 'math' | 'math500'."
            )

        if self.stratified:
            ds = self._stratified_subsample(ds)
        elif self.n_samples and self.n_samples < len(ds):
            ds = ds.select(range(self.n_samples))
            # Non-stratified mode still records concept labels so downstream
            # per-concept plots keep working.
            self._pos_to_concept = self._extract_concept_ids(ds).tolist()
        else:
            # Full dataset unmodified (stratified=False and no sample cap).
            self._pos_to_concept = self._extract_concept_ids(ds).tolist()

        self.dataset = ds
        return ds

    def _stratified_subsample(self, ds: datasets.Dataset) -> datasets.Dataset:
        """
        Draw samples evenly across concept buckets.

        For GSM8K  : 7 buckets derived by keyword classification.
        For MATH-500: 6 buckets taken directly from the 'concept_subject' field.

        Steps:
          1. Classify every row via ``classify_concept``.
          2. Group indices by concept_id.
          3. Allocate floor(n/7) samples to each concept, distributing
             the n%7 remainder one-by-one to the first buckets.
          4. Sample without replacement (fall back to all available if a
             bucket is smaller than the quota).
          5. Shuffle the merged selection so concepts interleave.
        """
        rng = np.random.default_rng(self.seed)

        if self.dataset_name in ("math500", "math"):
            concept_names_local = MATH500_CONCEPT_NAMES
            concept_ids = np.array(ds["concept_subject"])
        else:
            concept_names_local = CONCEPT_NAMES
            # Classify all questions
            all_questions = ds["question"]
            concept_ids = np.array([classify_concept(q) for q in all_questions])

        n_concepts = len(concept_names_local)

        # Build per-label index buckets
        buckets: Dict[int, List[int]] = {c: [] for c in range(n_concepts)}
        for idx, cid in enumerate(concept_ids):
            buckets[int(cid)].append(idx)

        # Print distribution for transparency
        dataset_label = {
            "math500": "MATH-500",
            "math": "MATH",
            "gsm8k": "GSM8K",
        }.get(self.dataset_name, self.dataset_name.upper())
        print(f"\n{dataset_label} concept distribution "
              f"(full {self.split} split, {len(ds)} rows):")
        for cid in range(n_concepts):
            print(f"  {concept_names_local[cid]:<30s}: {len(buckets[cid])}")

        # Allocate quotas
        base = self.n_samples // n_concepts
        remainder = self.n_samples % n_concepts  # distribute to first `remainder` labels

        selected_indices: List[int] = []
        actual_counts: Dict[str, int] = {}

        for cid in range(n_concepts):
            quota = base + (1 if cid < remainder else 0)
            available = buckets[cid]
            if len(available) == 0:
                print(f"  WARNING: no samples for concept '{concept_names_local[cid]}' — skipping")
                actual_counts[concept_names_local[cid]] = 0
                continue
            take = min(quota, len(available))
            if take < quota:
                print(
                    f"  WARNING: concept '{concept_names_local[cid]}' has only {len(available)} "
                    f"samples (quota={quota}); using all {take}"
                )
            chosen = rng.choice(available, size=take, replace=False).tolist()
            selected_indices.extend(chosen)
            actual_counts[concept_names_local[cid]] = take

        # Shuffle so label groups are interleaved during inference
        rng.shuffle(selected_indices)

        print(f"\nStratified sample counts (total={len(selected_indices)}):")
        for name, count in actual_counts.items():
            print(f"  {name:<30s}: {count}")

        # Store label per position (aligned with the selected dataset order).
        self._pos_to_concept: List[int] = [
            int(concept_ids[idx]) for idx in selected_indices
        ]

        return ds.select(selected_indices)

    # ── Inference ─────────────────────────────────────────────────────────────

    def execute_method(self) -> List[Dict[str, Any]]:
        """
        Run the latent reasoning method on the loaded dataset.

        Returns list of dicts with keys:
            - idx          : position in this run's sample list (0-based)
            - ds_idx       : original dataset index (before subsampling)
            - question     : input question text
            - concept      : int concept ID in [0, 6]
            - answer_expected  : ground truth float
            - answer_predicted : model's float answer (or None)
            - text_output  : full decoded text
            - latent_thoughts  : tensor [num_steps, hidden_dim]
            - correct      : bool
        """
        if self.dataset is None:
            self.load_dataset()

        # _pos_to_concept is populated by _stratified_subsample and maps
        # each position in the (shuffled) selected dataset to a concept ID.
        # For math500 (non-stratified path), use the pre-mapped 'concept_subject' field.
        # Fall back to on-the-fly keyword classification for non-stratified GSM8K.
        pos_to_concept: Optional[List[int]] = getattr(self, "_pos_to_concept", None)
        concept_names_local = (
            MATH500_CONCEPT_NAMES if self.dataset_name in ("math500", "math") else CONCEPT_NAMES
        )

        self.results = []
        for pos, sample in enumerate(self.dataset):
            question = sample["question"]
            expected = sample["answer_num"]

            if pos_to_concept is not None:
                concept = pos_to_concept[pos]
            elif self.dataset_name in ("math500", "math"):
                concept = int(sample["concept_subject"])
            else:
                concept = classify_concept(question)

            difficulty = int(sample["difficulty"]) if "difficulty" in sample else -1

            try:
                output = self.wrapper.run_inference(question)
            except PromptTooLongError as e:
                print(f"  [pos {pos}] skipped — {e}")
                continue
            thoughts = self.wrapper.extract_latent_thoughts(output)  # moved to CPU
            text = self.wrapper.decode_output(output)
            predicted = self.wrapper.extract_answer(text)

            # Free GPU-resident tensors (hidden_states, KV cache, etc.)
            # that would otherwise accumulate across samples and OOM.
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Correctness drives Jerome's "correct vs incorrect metric curves"
            # requirement: the plotting module splits every metric on this
            # boolean so he can see how stable/fast trajectories diverge from
            # unstable ones.
            correct = False
            if predicted is not None:
                try:
                    correct = bool(abs(float(predicted) - float(expected)) < 1e-6)
                except (TypeError, ValueError):
                    correct = False

            self.results.append({
                "idx": pos,
                "question": question,
                "concept": concept,
                "difficulty": difficulty,
                "answer_expected": expected,
                "answer_predicted": predicted,
                "text_output": text,
                "latent_thoughts": thoughts,
                "correct": (
                    predicted is not None
                    and expected == expected  # filters out float("nan")
                    and predicted == expected
                ),
            })

            print(
                f"[{pos + 1}/{len(self.dataset)}] "
                f"Concept={concept_names_local[concept]:<28s} "
                f"Expected={expected}, Predicted={predicted}, "
                f"Correct={correct}, "
                f"Latent shape={thoughts.shape}"
            )

        return self.results

    # ── I/O helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def save_tensor(tensor: torch.Tensor, filepath: str, key: str = "data"):
        """Save a tensor to HDF5 format."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(filepath), "a") as f:
            if key in f:
                del f[key]
            f.create_dataset(key, data=tensor.numpy())

    @staticmethod
    def load_tensor(filepath: str, key: str = "data") -> torch.Tensor:
        """Load a tensor from HDF5 format."""
        with h5py.File(str(filepath), "r") as f:
            data = f[key][:]
        return torch.from_numpy(data)

    def save_latent_states(self, output_dir: str):
        """
        Save all latent thought trajectories to HDF5.

        Creates one file per sample: latent_states/sample_{idx}.h5
        Also saves a combined file: latent_states/all_states.h5 with
        datasets: latent_thoughts, correct, answer_expected, concept.
        """
        out = Path(output_dir) / "latent_states"
        out.mkdir(parents=True, exist_ok=True)

        all_thoughts = []
        metadata = []

        for result in self.results:
            idx = result["idx"]
            thoughts = result["latent_thoughts"]

            # Save individual
            self.save_tensor(thoughts, str(out / f"sample_{idx}.h5"), key="latent_thoughts")

            all_thoughts.append(thoughts)
            metadata.append({
                "idx": idx,
                "question": result["question"],
                "concept": result["concept"],
                "difficulty": result.get("difficulty", -1),
                "correct": result["correct"],
                "answer_expected": result["answer_expected"],
                "answer_predicted": result["answer_predicted"],
            })

        # Save combined — shape: [n_samples, num_steps, hidden_dim]
        combined = torch.stack(all_thoughts, dim=0)
        combined_path = str(out / "all_states.h5")
        self.save_tensor(combined, combined_path, key="latent_thoughts")

        # Save metadata (including concept and difficulty labels). The
        # question text rides along so the perturbation flow can re-run
        # the model from the H5 alone.
        with h5py.File(combined_path, "a") as f:
            for key in ("correct", "answer_expected", "concept", "difficulty",
                        "question"):
                if key in f:
                    del f[key]
            f.create_dataset(
                "correct",
                data=np.array([m["correct"] for m in metadata]),
            )
            f.create_dataset(
                "answer_expected",
                data=np.array([m["answer_expected"] for m in metadata]),
            )
            f.create_dataset(
                "concept",
                data=np.array([m["concept"] for m in metadata], dtype=np.int32),
            )
            f.create_dataset(
                "difficulty",
                data=np.array([m["difficulty"] for m in metadata], dtype=np.int32),
            )
            f.create_dataset(
                "question",
                data=np.array([m["question"] for m in metadata],
                              dtype=h5py.string_dtype(encoding="utf-8")),
            )

        print(f"Saved {len(self.results)} latent state trajectories to {out}")
        return str(out)
