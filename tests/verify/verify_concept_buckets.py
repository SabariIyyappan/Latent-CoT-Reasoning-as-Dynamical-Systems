"""
Audit the 7 keyword derived GSM8K concept buckets.

Answers the action items raised by Sabari in the April 17 SMRS meeting:
    1. Do the per bucket counts match what the project claims?
    2. Are there obvious misclassifications inside each bucket?

Does not require a GPU or the models. Pulls GSM8K directly from the
datasets library and runs classify_concept over every row. For each
bucket it:
    a. Prints the count and percentage of the 8792 total.
    b. Samples up to N_SAMPLE questions and shows the keyword that fired.
    c. Flags any question whose only matched keyword is also present in an
       earlier bucket (priority collision).

Usage:
    python scripts/verify_concept_buckets.py
    python scripts/verify_concept_buckets.py --n_sample 20 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.data_prep import (
    classify_concept,
    GSM8K_CONCEPT_NAMES,
    _CONCEPT_KEYWORDS,
)


EXPECTED_COUNTS = {
    "Geometry":                  210,
    "Rates & Speed":             675,
    "Percentages & Ratios":      1266,
    "Money & Pricing":           2741,
    "Fractions & Decimals":      1045,
    "Multiplication & Division": 224,
    "Arithmetic & Multi-step":   2631,
}


def first_hit(question_lower: str, concept_id: int) -> str:
    """Return the first matching keyword for the given concept, or empty."""
    for kw in _CONCEPT_KEYWORDS[concept_id]:
        if kw in question_lower:
            return kw
    return ""


def earlier_bucket_hit(question_lower: str, concept_id: int) -> tuple[int, str] | None:
    """If any earlier bucket also has a keyword in this question, return it."""
    for earlier_id in range(concept_id):
        kw = first_hit(question_lower, earlier_id)
        if kw:
            return earlier_id, kw
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_sample", type=int, default=10,
                        help="Questions to print per bucket (default 10)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    ds_train = load_dataset("gsm8k", "main", split="train")
    ds_test = load_dataset("gsm8k", "main", split="test")
    questions = list(ds_train["question"]) + list(ds_test["question"])
    print(f"Loaded GSM8K train+test: {len(questions)} questions\n")

    rng = random.Random(args.seed)

    per_bucket: dict[int, list[str]] = defaultdict(list)
    counts = Counter()
    for q in questions:
        cid = classify_concept(q)
        counts[cid] += 1
        per_bucket[cid].append(q)

    print("=" * 70)
    print("Per bucket counts and expected counts")
    print("=" * 70)
    for cid, name in enumerate(GSM8K_CONCEPT_NAMES):
        got = counts[cid]
        expected = EXPECTED_COUNTS.get(name, None)
        pct = 100.0 * got / len(questions)
        ok = ""
        if expected is not None:
            ok = " OK" if got == expected else f" MISMATCH expected {expected}"
        print(f"  [{cid}] {name:<30s} {got:>5d}  {pct:5.1f} %{ok}")
    total = sum(counts.values())
    print(f"  {'TOTAL':<34s} {total:>5d}")
    print()

    print("=" * 70)
    print(f"Sampled questions per bucket (up to {args.n_sample} each)")
    print("=" * 70)
    for cid, name in enumerate(GSM8K_CONCEPT_NAMES):
        bucket = per_bucket[cid]
        n = len(bucket)
        show = rng.sample(bucket, min(args.n_sample, n))
        print(f"\n--- [{cid}] {name}  (n={n})")
        for q in show:
            q_low = q.lower()
            kw = first_hit(q_low, cid) if cid < len(_CONCEPT_KEYWORDS) else ""
            if cid == 6:
                tag = "(no keywords, catch all)"
            else:
                tag = f'matched "{kw}"'
            short = q.replace("\n", " ")
            if len(short) > 140:
                short = short[:137] + "..."
            print(f"    {tag}")
            print(f"      {short}")

    print()
    print("=" * 70)
    print("Priority collision audit")
    print("=" * 70)
    print("How often does a question sitting in bucket C also contain a keyword")
    print("from some earlier bucket ? Priority ordering means the earlier bucket")
    print("would have won if keywords had been checked in a different order.")
    print()
    collision_counts = Counter()
    collision_examples: dict[int, list[tuple[int, str, str, str]]] = defaultdict(list)
    for cid in range(1, 6):
        for q in per_bucket[cid]:
            q_low = q.lower()
            earlier = earlier_bucket_hit(q_low, cid)
            if earlier is not None:
                earlier_id, earlier_kw = earlier
                collision_counts[cid] += 1
                if len(collision_examples[cid]) < 3:
                    own_kw = first_hit(q_low, cid)
                    collision_examples[cid].append(
                        (earlier_id, earlier_kw, own_kw, q[:120].replace("\n", " "))
                    )

    for cid, name in enumerate(GSM8K_CONCEPT_NAMES):
        if cid < 1 or cid > 5:
            continue
        n = len(per_bucket[cid])
        col = collision_counts[cid]
        rate = 100.0 * col / max(1, n)
        print(f"  [{cid}] {name:<30s} {col:>4d} of {n} ({rate:4.1f} %) collide")
        for earlier_id, earlier_kw, own_kw, text in collision_examples[cid]:
            print(f'      earlier bucket {earlier_id} ({GSM8K_CONCEPT_NAMES[earlier_id]}) '
                  f'via "{earlier_kw}" vs own "{own_kw}"')
            print(f"        {text}...")
    print()
    print("High collision rates mean the bucket is sensitive to priority order.")
    print("Low collision rates mean the keyword list is distinctive.")


if __name__ == "__main__":
    main()
