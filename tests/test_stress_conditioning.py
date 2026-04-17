"""Tests for stress-conditioned context discovery.

Stress is a user-supplied suprasegmental on ``Segment.stress``. The
discovery layer learns conditioning rules over it through three
new ``Context`` slots: ``self_stress``, ``preceding_stress``,
``following_stress``.

These tests pin:

1. Context propagates stress from the Segment into the appropriate
   slot during alignment.
2. With stress annotations in the corpus, the discovery layer
   commits 1-to-1 stress-conditioned rules.
3. Without stress annotations (unchanged corpora), nothing about
   training behavior is affected — stress predicates only enter
   the candidate space when observations actually carry stress.
"""

from pathlib import Path

from regulae import (
    CognateSet,
    Context,
    FeatureConstraint,
    Form,
    LearnedModel,
    MultiLectModel,
    Segment,
    align_forms,
    cognate_sets_from_pairs,
    train_model,
)


def _form(lect: str, word: str, stress_pos: int | None = None) -> Form:
    segs = []
    for i, c in enumerate(word):
        segs.append(Segment(c, stress="+" if i == stress_pos else None))
    return Form(lect, tuple(segs))


# ----- context propagation ------------------------------------------------


def test_self_stress_populated_from_segment_stress() -> None:
    # Link contexts carry stress only when a model is supplied
    # (prior-only alignment uses empty Context).
    model = LearnedModel.empty()
    src = _form("A", "pet", stress_pos=1)
    tgt = _form("B", "pet", stress_pos=1)
    alignment = align_forms(src, tgt, model=model)
    middle_link = alignment.links[1]
    assert any(
        fc.feature == "stress" and fc.value == "+"
        for fc in middle_link.context.self_stress
    )


def test_preceding_and_following_stress_populated() -> None:
    model = LearnedModel.empty()
    src = _form("A", "pet", stress_pos=1)
    tgt = _form("B", "pet", stress_pos=1)
    alignment = align_forms(src, tgt, model=model)
    first_link = alignment.links[0]
    assert any(
        fc.feature == "stress" and fc.value == "+"
        for fc in first_link.context.following_stress
    )
    last_link = alignment.links[2]
    assert any(
        fc.feature == "stress" and fc.value == "+"
        for fc in last_link.context.preceding_stress
    )


def test_context_subset_honors_stress_slots() -> None:
    """Context.is_subset_of should now consider stress slots."""
    base = Context()
    stressed = Context(self_stress=(FeatureConstraint("stress", "+"),))
    assert base.is_subset_of(stressed)  # empty is subset of anything
    assert not stressed.is_subset_of(base)  # stressed isn't subset of empty


# ----- discovery on synthetic fixture ------------------------------------


def test_stress_synthetic_commits_vowel_lowering_rules() -> None:
    """COMMITMENT: on a stress-conditioned vowel-lowering corpus,
    discovery commits ``e → ɛ / self_stress=[stress:+]`` and
    ``o → ɔ / self_stress=[stress:+]``."""
    # Replicate the fixture programmatically so the test does not
    # depend on disk layout.
    vowels = frozenset("aeiouɛɔ")

    def form(lect: str, ipa: str) -> Form:
        segs: list[Segment] = []
        pending = None
        for ch in ipa:
            if ch == "ˈ":
                pending = "+"
                continue
            if ch == "-":
                continue
            if ch in vowels and pending is not None:
                segs.append(Segment(ch, stress=pending))
                pending = None
            else:
                segs.append(Segment(ch))
        return Form(lect, tuple(segs))

    pairs = []
    for p, d in [
        ("ˈpe-ta", "ˈpɛ-ta"), ("ˈpe-ro", "ˈpɛ-ro"), ("ˈpe-li", "ˈpɛ-li"),
        ("ˈte-na", "ˈtɛ-na"), ("ˈte-ru", "ˈtɛ-ru"), ("ˈte-mo", "ˈtɛ-mo"),
        ("ˈke-pa", "ˈkɛ-pa"), ("ˈke-no", "ˈkɛ-no"), ("ˈme-ru", "ˈmɛ-ru"),
        ("ˈme-na", "ˈmɛ-na"), ("ˈde-po", "ˈdɛ-po"), ("ˈde-la", "ˈdɛ-la"),
        ("ˈpo-ta", "ˈpɔ-ta"), ("ˈpo-mi", "ˈpɔ-mi"), ("ˈto-la", "ˈtɔ-la"),
        ("ˈto-pe", "ˈtɔ-pe"), ("ˈko-na", "ˈkɔ-na"), ("ˈko-ri", "ˈkɔ-ri"),
        ("ˈmo-la", "ˈmɔ-la"), ("ˈmo-ni", "ˈmɔ-ni"), ("ˈdo-ra", "ˈdɔ-ra"),
        ("ˈdo-mi", "ˈdɔ-mi"),
        ("pe-ˈta", "pe-ˈta"), ("pe-ˈro", "pe-ˈro"),
        ("te-ˈna", "te-ˈna"), ("te-ˈru", "te-ˈru"),
        ("ke-ˈpa", "ke-ˈpa"), ("me-ˈna", "me-ˈna"), ("de-ˈpo", "de-ˈpo"),
        ("po-ˈta", "po-ˈta"), ("po-ˈmi", "po-ˈmi"),
        ("to-ˈla", "to-ˈla"), ("to-ˈpe", "to-ˈpe"),
        ("ko-ˈna", "ko-ˈna"), ("mo-ˈla", "mo-ˈla"), ("do-ˈra", "do-ˈra"),
    ]:
        pairs.append((form("proto", p), form("derived", d)))

    corpus = cognate_sets_from_pairs(pairs, ("proto", "derived"))
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"proto", "derived"})]

    e_lowered = [
        (cc, c)
        for cc, c in pair.segment_table.counts.items()
        if cc.src == "e" and cc.tgt == "ɛ"
        and any(
            fc.feature == "stress" and fc.value == "+"
            for fc in cc.context.self_stress
        )
    ]
    o_lowered = [
        (cc, c)
        for cc, c in pair.segment_table.counts.items()
        if cc.src == "o" and cc.tgt == "ɔ"
        and any(
            fc.feature == "stress" and fc.value == "+"
            for fc in cc.context.self_stress
        )
    ]
    assert e_lowered, "expected e → ɛ conditioned on self_stress=[stress:+]"
    assert o_lowered, "expected o → ɔ conditioned on self_stress=[stress:+]"
    # Rule should be supported by at least 10 of the 12 stressed-e pairs.
    assert sum(c for _, c in e_lowered) >= 10
    assert sum(c for _, c in o_lowered) >= 8


# ----- unchanged behavior without stress ---------------------------------


def test_corpus_without_stress_annotation_is_unaffected() -> None:
    """Corpora without any stress annotation produce no stress-conditioned
    entries — stress predicates are only emitted when observations carry
    stress values, so unannotated training is unchanged."""
    corpus = [
        CognateSet(
            cognate_id=f"c{i}",
            forms={
                "A": Form("A", (Segment("p"), Segment("a"))),
                "B": Form("B", (Segment("f"), Segment("a"))),
            },
        )
        for i in range(5)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"A", "B"})]
    for cc in pair.segment_table.counts:
        ctx = cc.context
        assert not ctx.self_stress
        assert not ctx.preceding_stress
        assert not ctx.following_stress
