"""Tests for confidence-weighted training and contamination control."""

from regulae import CognateSet, Form, MultiLectModel, Segment, train_model


def _form(lect: str, word: str) -> Form:
    return Form(lect, tuple(Segment(c) for c in word))


def test_pairwise_counts_scale_with_confidence() -> None:
    corpus = [
        CognateSet(
            cognate_id="c1",
            forms={"A": _form("A", "pa"), "B": _form("B", "fa")},
            confidence=1.0,
        ),
        CognateSet(
            cognate_id="c2",
            forms={"A": _form("A", "pa"), "B": _form("B", "fa")},
            confidence=0.25,
        ),
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"A", "B"})]
    p_to_f = sum(
        count
        for cc, count in pair.segment_table.counts.items()
        if cc.src == "p" and cc.tgt == "f" and cc.context.constraint_count() == 0
    )
    a_to_a = sum(
        count
        for cc, count in pair.segment_table.counts.items()
        if cc.src == "a" and cc.tgt == "a" and cc.context.constraint_count() == 0
    )
    assert p_to_f == 1.25
    assert a_to_a == 1.25


def test_multilect_unconditioned_counts_scale_with_confidence() -> None:
    corpus = [
        CognateSet(
            cognate_id="c1",
            forms={"A": _form("A", "ka"), "B": _form("B", "ta"), "C": _form("C", "ka")},
            confidence=1.0,
        ),
        CognateSet(
            cognate_id="c2",
            forms={"A": _form("A", "ka"), "B": _form("B", "ta"), "C": _form("C", "ka")},
            confidence=0.5,
        ),
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    counts = {
        tuple(sorted(klass.segments.items())): klass.count
        for klass in model.unconditioned_classes
    }
    assert counts[(("A", "k"), ("B", "t"), ("C", "k"))] == 1.5
    assert counts[(("A", "a"), ("B", "a"), ("C", "a"))] == 1.5


def test_zero_confidence_suppresses_false_merger_class() -> None:
    clean = [
        CognateSet(
            cognate_id=f"good_k.{i}",
            forms={"M": _form("M", "ʔa"), "P": _form("P", "ka"), "C": _form("C", "ka")},
            confidence=1.0,
        )
        for i in range(6)
    ] + [
        CognateSet(
            cognate_id=f"good_glot.{i}",
            forms={"M": _form("M", "ʔa"), "P": _form("P", "ʔa"), "C": _form("C", "ta")},
            confidence=1.0,
        )
        for i in range(6)
    ]
    contaminated = clean + [
        CognateSet(
            cognate_id=f"bad.{i}",
            forms={"M": _form("M", "ʔa"), "P": _form("P", "ka"), "C": _form("C", "ta")},
            confidence=1.0,
        )
        for i in range(4)
    ]
    downweighted = clean + [
        CognateSet(
            cognate_id=f"bad0.{i}",
            forms={"M": _form("M", "ʔa"), "P": _form("P", "ka"), "C": _form("C", "ta")},
            confidence=0.0,
        )
        for i in range(4)
    ]

    contaminated_model = train_model(contaminated)
    downweighted_model = train_model(downweighted)
    assert isinstance(contaminated_model, MultiLectModel)
    assert isinstance(downweighted_model, MultiLectModel)

    contaminated_counts = {
        tuple(sorted(klass.segments.items())): klass.count
        for klass in contaminated_model.unconditioned_classes
    }
    downweighted_counts = {
        tuple(sorted(klass.segments.items())): klass.count
        for klass in downweighted_model.unconditioned_classes
    }

    bogus_partial = (("M", "ʔ"), ("P", "k"))
    clean_k = (("C", "k"), ("M", "ʔ"), ("P", "k"))

    assert contaminated_counts.get(bogus_partial, 0.0) == 4.0
    assert downweighted_counts.get(bogus_partial, 0.0) == 0.0
    assert contaminated_counts.get(clean_k, 0.0) == 6.0
    assert downweighted_counts.get(clean_k, 0.0) == 6.0
