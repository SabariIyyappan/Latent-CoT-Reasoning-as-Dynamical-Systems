"""Tests for analysis.data_prep.classify_concept."""

from analysis.data_prep import classify_concept, GSM8K_CONCEPT_NAMES


def test_classify_concept_priority():
    # Plan spec (Jerome + jerryfrancis-97 review): "rectangle costs $5"
    # must go to Geometry (0), not Money (3) — priority ordering wins.
    assert classify_concept("A rectangle costs $5 per square foot") == 0

    # Same kind of crosswire: a rates question that also mentions pay.
    assert classify_concept("He pays $10 per hour of work") == 1


def test_classify_concept_unambiguous_buckets():
    # One representative question per bucket, phrased so only that
    # bucket's keywords can match.
    assert classify_concept("What is the area of a circle with radius 5") == 0
    assert classify_concept("A car travels at 60 mph for 3 hours") == 1
    assert classify_concept("What is 25% of 80?") == 2
    assert classify_concept("John bought 3 apples for a total of $6") == 3
    assert classify_concept("Convert 1/2 to a decimal") == 4
    assert classify_concept("15 multiplied by 7 is what") == 5


def test_classify_concept_catchall():
    # No keyword hit → bucket 6 (Arithmetic & Multi-step).
    assert classify_concept(
        "Sarah has 5 books and gives 2 away, how many does she have left"
    ) == 6


def test_classify_concept_deterministic():
    # Same input, same output, always.
    q = "A rectangle with area 12 costs $5"
    out = classify_concept(q)
    for _ in range(10):
        assert classify_concept(q) == out


def test_classify_concept_returns_valid_id():
    # All outputs in [0, len(GSM8K_CONCEPT_NAMES)-1].
    samples = [
        "",
        "What is 2+2?",
        "How fast does a car travel 60 miles in 1 hour?",
        "Find the volume of a cube with side 3",
    ]
    for q in samples:
        cid = classify_concept(q)
        assert 0 <= cid < len(GSM8K_CONCEPT_NAMES)
