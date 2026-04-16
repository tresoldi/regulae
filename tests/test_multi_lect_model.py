"""Tests for the multi-lect data types.

These verify construction, defaults, immutability, and the basic
semantics described in the multi-lect model specification
(docs/consumer_guide.md). No training logic is exercised here —
training tests live elsewhere.
"""

import pytest

from regulae import (
    CognateSet,
    Context,
    Form,
    LearnedModel,
    MultiLectCorrespondenceClass,
    MultiLectModel,
    Segment,
)
from regulae.types import FeatureConstraint


def _form(lect: str, word: str) -> Form:
    return Form(lect, tuple(Segment(c) for c in word))


# ----- CognateSet --------------------------------------------------------


def test_cognate_set_minimal_construction() -> None:
    """COMMITMENT: a CognateSet can be built from just an id and forms."""
    cs = CognateSet(
        cognate_id="hand.001",
        forms={"latin": _form("latin", "manus"), "spanish": _form("spanish", "mano")},
    )
    assert cs.cognate_id == "hand.001"
    assert "latin" in cs.forms
    assert "spanish" in cs.forms
    assert cs.alignments is None
    assert cs.morpheme_boundaries is None
    assert cs.confidence == 1.0


def test_cognate_set_single_lect_is_valid() -> None:
    """COMMITMENT: N=1 is a legal degenerate case."""
    cs = CognateSet(cognate_id="x", forms={"latin": _form("latin", "pater")})
    assert len(cs.forms) == 1


def test_cognate_set_absent_lect_is_just_missing_from_forms() -> None:
    """COMMITMENT: Q6 — no sentinel value for missing lects; absence
    is represented by the lect not being in ``forms``."""
    cs = CognateSet(
        cognate_id="x",
        forms={"a": _form("a", "pa"), "c": _form("c", "fa")},
    )
    assert "b" not in cs.forms
    assert set(cs.forms) == {"a", "c"}


def test_cognate_set_holds_optional_alignments() -> None:
    """COMMITMENT: alignments is an optional warm-start hint."""
    align = {
        "latin": (Segment("m"), Segment("a"), Segment("n"), Segment("u"), Segment("s")),
        "spanish": (Segment("m"), Segment("a"), Segment("n"), Segment("o"), None),
    }
    cs = CognateSet(
        cognate_id="hand",
        forms={"latin": _form("latin", "manus"), "spanish": _form("spanish", "mano")},
        alignments=align,
    )
    assert cs.alignments is not None
    assert cs.alignments["spanish"][-1] is None


def test_cognate_set_confidence_is_settable() -> None:
    cs = CognateSet(
        cognate_id="x",
        forms={"a": _form("a", "pa")},
        confidence=0.6,
    )
    assert cs.confidence == 0.6


def test_cognate_set_morpheme_boundaries_reserved_field() -> None:
    """COMMITMENT: morpheme_boundaries is captured but not used in ."""
    cs = CognateSet(
        cognate_id="x",
        forms={"a": _form("a", "patre")},
        morpheme_boundaries={"a": (4,)},
    )
    assert cs.morpheme_boundaries == {"a": (4,)}


def test_cognate_set_is_frozen() -> None:
    cs = CognateSet(cognate_id="x", forms={"a": _form("a", "pa")})
    with pytest.raises(Exception):
        cs.cognate_id = "y"  # type: ignore[misc]


# ----- MultiLectCorrespondenceClass --------------------------------------


def test_multilect_class_unconditioned_defaults() -> None:
    """COMMITMENT: contexts=None means the class is unconditioned."""
    klass = MultiLectCorrespondenceClass(
        class_id=0,
        segments={"latin": "p", "spanish": "p", "french": "p"},
    )
    assert klass.contexts is None
    assert klass.count == 0.0
    assert klass.supporting_cognates == ()


def test_multilect_class_with_contexts_is_conditioned() -> None:
    """COMMITMENT: non-None contexts means the class is conditioned."""
    front = Context(following=(FeatureConstraint("front", "+"),))
    klass = MultiLectCorrespondenceClass(
        class_id=7,
        segments={"latin": "k", "spanish": "θ", "french": "s"},
        contexts={"latin": front, "spanish": Context(), "french": Context()},
        count=12.0,
        supporting_cognates=("caelum", "cera"),
    )
    assert klass.contexts is not None
    assert klass.contexts["latin"] == front
    assert klass.count == 12.0
    assert "caelum" in klass.supporting_cognates


def test_multilect_class_is_frozen() -> None:
    klass = MultiLectCorrespondenceClass(class_id=0, segments={"a": "p"})
    with pytest.raises(Exception):
        klass.class_id = 99  # type: ignore[misc]


# ----- MultiLectModel ----------------------------------------------------


def test_multilect_model_empty_factory() -> None:
    """COMMITMENT: empty() yields a model with no pairs and no classes."""
    m = MultiLectModel.empty()
    assert m.pairwise_models == {}
    assert m.unconditioned_classes == ()
    assert m.conditioned_classes == ()
    assert m.cognate_corpus == ()
    assert m.lect_ids == ()


def test_multilect_model_pairs_keyed_by_frozenset() -> None:
    """COMMITMENT: pairs are keyed by frozenset so (a,b) == (b,a)."""
    pair = frozenset({"latin", "spanish"})
    m = MultiLectModel(
        pairwise_models={pair: LearnedModel.empty()},
        unconditioned_classes=(),
        conditioned_classes=(),
        cognate_corpus=(),
        lect_ids=("latin", "spanish"),
    )
    assert frozenset({"spanish", "latin"}) in m.pairwise_models


def test_multilect_model_holds_classes_and_corpus() -> None:
    cs = CognateSet(
        cognate_id="c1",
        forms={"a": _form("a", "pa"), "b": _form("b", "fa")},
    )
    uncond = MultiLectCorrespondenceClass(
        class_id=0, segments={"a": "p", "b": "f"}, count=1.0
    )
    m = MultiLectModel(
        pairwise_models={frozenset({"a", "b"}): LearnedModel.empty()},
        unconditioned_classes=(uncond,),
        conditioned_classes=(),
        cognate_corpus=(cs,),
        lect_ids=("a", "b"),
    )
    assert m.unconditioned_classes[0].segments["a"] == "p"
    assert m.cognate_corpus[0].cognate_id == "c1"
    assert m.lect_ids == ("a", "b")


def test_multilect_model_is_frozen() -> None:
    m = MultiLectModel.empty()
    with pytest.raises(Exception):
        m.lect_ids = ("x",)  # type: ignore[misc]
