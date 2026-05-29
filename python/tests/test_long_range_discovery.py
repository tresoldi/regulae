"""Tests for : long-range context discovery (cross-dimensional discovery split loop).

Validates that ``_long_range_discovery`` commits a long-range
conditioned correspondence when — and only when — the signal is
genuine. Three kinds of fixture:

- a **clean umlaut** fixture (V1=a → æ / next_syl front) that the
  loop must recover
- a **clean harmony** fixture (V2=a → o / prev_syl back) that the
  loop must recover on a different slot
- a **noise** fixture with no long-range structure that the loop
  must leave untouched
"""

from __future__ import annotations

import warnings

import pytest

from regulae import (
    Form,
    Segment,
    cognate_sets_from_pairs,
    train_model,
)
from regulae.types import Context


@pytest.fixture(autouse=True)
def _suppress_pair_deprecation():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    yield


def _form(lect: str, ipa: str) -> Form:
    return Form(lect_id=lect, segments=tuple(Segment(c) for c in ipa))


def _lr_entries_for_source(model, src: str) -> list:
    out = []
    for cc, count in model.segment_table.counts.items():
        if cc.src != src:
            continue
        ctx = cc.context
        lr = (
            len(ctx.preceding_at_distance)
            + len(ctx.following_at_distance)
            + len(ctx.somewhere_preceding)
            + len(ctx.somewhere_following)
            + len(ctx.same_syllable)
            + len(ctx.next_syllable)
            + len(ctx.previous_syllable)
        )
        if lr > 0:
            out.append((cc, count))
    return out


def _umlaut_corpus() -> list[tuple[Form, Form]]:
    """10 umlaut-firing pairs and 10 controls. Rule: a → æ iff V2 front."""
    umlaut = [
        ("pati", "pæti"), ("bani", "bæni"), ("kati", "kæti"),
        ("dani", "dæni"), ("gati", "gæti"),
        ("pabe", "pæbe"), ("bade", "bæde"), ("kame", "kæme"),
        ("dape", "dæpe"), ("gabe", "gæbe"),
    ]
    control = [
        ("pato", "pato"), ("bado", "bado"), ("kako", "kako"),
        ("dago", "dago"), ("gamo", "gamo"),
        ("papu", "papu"), ("babu", "babu"), ("kamu", "kamu"),
        ("datu", "datu"), ("gapu", "gapu"),
    ]
    # Control V1!=a pairs to populate the search space.
    nonhit = [
        ("piti", "piti"), ("biki", "biki"), ("kiti", "kiti"),
        ("gidi", "gidi"), ("pito", "pito"),
    ]
    all_pairs = umlaut + control + nonhit
    return [(_form("A", s), _form("B", t)) for s, t in all_pairs]


def _harmony_corpus() -> list[tuple[Form, Form]]:
    """15 harmony-firing pairs (V1 back → V2 a→o) + controls."""
    hit = [
        ("puka", "puko"), ("buka", "buko"), ("tuma", "tumo"),
        ("duna", "duno"), ("kuba", "kubo"), ("gupa", "gupo"),
        ("poka", "poko"), ("boma", "bomo"), ("toka", "toko"),
        ("doba", "dobo"), ("nupa", "nupo"), ("muka", "muko"),
        ("gopa", "gopo"), ("noma", "nomo"), ("kopa", "kopo"),
    ]
    nonhit = [
        ("pika", "pika"), ("bika", "bika"), ("tima", "tima"),
        ("dina", "dina"), ("kiba", "kiba"), ("gipa", "gipa"),
        ("peka", "peka"), ("beka", "beka"), ("tema", "tema"),
        ("dena", "dena"),
    ]
    return [(_form("A", s), _form("B", t)) for s, t in hit + nonhit]


def _noise_corpus() -> list[tuple[Form, Form]]:
    """40 identity pairs with no long-range structure whatsoever."""
    words = [
        "pata", "kata", "tapa", "paka", "maku", "kuma", "puki", "piki",
        "toke", "keto", "buba", "dudu", "gaga", "kiki", "papa", "tata",
        "nana", "mama", "gogo", "pipi", "tutu", "bubu", "dada", "nene",
        "mimi", "kuki", "guga", "toto", "dodo", "baba", "giga", "tiki",
        "mako", "kumi", "pitu", "takö", "bika", "doke", "noma", "kupu",
    ]
    return [(_form("A", w), _form("B", w)) for w in words]


def _train_pair(corpus):
    multi = train_model(cognate_sets_from_pairs(corpus, ("A", "B")))
    return multi.pairwise_models[frozenset({"A", "B"})]


def test_long_range_recovers_umlaut_rule() -> None:
    """COMMITMENT: on a clean umlaut fixture, long-range discovery
    discovery commits at least one ``a → æ`` entry whose context
    carries a ``next_syllable=[front:+]`` constraint."""
    model = _train_pair(_umlaut_corpus())
    entries = _lr_entries_for_source(model, "a")
    # Should include an a→æ with next_syllable front constraint.
    hits = [
        (cc, cnt) for cc, cnt in entries
        if cc.tgt == "æ"
        and any(
            fc.feature == "front" and fc.value == "+"
            for fc in cc.context.next_syllable
        )
    ]
    assert len(hits) >= 1, f"no umlaut rule committed; got {entries}"
    # The committed rule should capture most of the umlaut firings.
    (_, count), = hits[:1]
    assert count >= 8


def test_long_range_recovers_harmony_rule() -> None:
    """COMMITMENT: on a clean harmony fixture, cross-dimensional discovery commits a
    ``previous_syllable``-conditioned entry on source ``a``. Either
    framing is acceptable:

    - ``a → o / previous_syllable=[back:+]`` (the rule-firing side)
    - ``a → a / previous_syllable=[front:+]`` (the dual — the rule
      "a stays a iff previous vowel is front", which is
      equivalent on this fixture because every non-back vowel is
      front and every back vowel triggers harmony)

    The greedy loop picks whichever predicate comes first in the
    candidate enumeration and meets BIC — both are perfect
    partitions on this fixture.
    """
    model = _train_pair(_harmony_corpus())
    entries = _lr_entries_for_source(model, "a")
    assert len(entries) >= 1, "no long-range rule on source 'a'"
    hits = [
        (cc, cnt) for cc, cnt in entries
        if (
            cc.context.previous_syllable
            and any(
                fc.feature in ("front", "back") and fc.value == "+"
                for fc in cc.context.previous_syllable
            )
        )
    ]
    assert len(hits) >= 1, f"no prev_syl harmony rule committed; got {entries}"
    # At least one commit carries a substantive count of observations.
    assert max(cnt for _, cnt in hits) >= 10


def test_long_range_commits_nothing_on_noise_corpus() -> None:
    """COMMITMENT: on an identity noise corpus with no long-range
    structure, cross-dimensional discovery must not commit any long-range entry."""
    model = _train_pair(_noise_corpus())
    any_lr: list = []
    for cc, count in model.segment_table.counts.items():
        ctx = cc.context
        lr = (
            len(ctx.preceding_at_distance)
            + len(ctx.following_at_distance)
            + len(ctx.somewhere_preceding)
            + len(ctx.somewhere_following)
            + len(ctx.same_syllable)
            + len(ctx.next_syllable)
            + len(ctx.previous_syllable)
        )
        if lr > 0:
            any_lr.append((cc, count))
    assert any_lr == [], f"unexpected long-range commits on noise: {any_lr}"


def test_long_range_is_deterministic() -> None:
    """COMMITMENT: repeated training on the same corpus produces
    bitwise-identical long-range entries."""
    corpus = _umlaut_corpus()
    m1 = _train_pair(corpus)
    m2 = _train_pair(corpus)
    assert m1.segment_table.counts == m2.segment_table.counts


# ----- : multi-lect long-range lifting ------------------------------


def test_multi_lect_long_range_split_helper_fires_on_clean_signal() -> None:
    """COMMITMENT: ``_commit_multi_lect_long_range_splits_for_pivot``
    commits at least one entry when fed observations with a clean
    long-range signal — a pivot bucket where 12 of 24 observations
    carry one sister tuple and a ``next_syllable=[front:+]``
    context, and the other 12 carry a different sister tuple
    without it. The end-to-end multi-lect training pipeline can
    fail to surface this because  reconciliation may split the
    proto pivot's a / æ positions into separate classes (different
    issue,  territory); this test isolates the  helper."""
    from regulae.training import (
        _commit_multi_lect_long_range_splits_for_pivot,
    )
    from regulae.types import FeatureConstraint

    front_ctx = Context(next_syllable=(FeatureConstraint("front", "+"),))
    plain_ctx = Context()
    sister_front = (("D1", "æ"), ("D2", "æ"))
    sister_plain = (("D1", "a"), ("D2", "a"))
    sister_by_key = {
        "front_key": sister_front,
        "plain_key": sister_plain,
    }
    observations = (
        [("front_key", front_ctx)] * 12 + [("plain_key", plain_ctx)] * 12
    )
    committed: list = []
    _commit_multi_lect_long_range_splits_for_pivot(
        pivot_lect="PROTO",
        pivot_grapheme="a",
        observations=observations,
        sister_by_key=sister_by_key,
        committed=committed,
        min_commit_scale=0.5,
    )
    assert len(committed) >= 1, "no long-range commit on clean signal"
    pivot_lect, pivot_g, ctx, sister, count, n_total = committed[0]
    assert pivot_lect == "PROTO"
    assert pivot_g == "a"
    assert any(
        fc.feature == "front" and fc.value == "+"
        for fc in ctx.next_syllable
    )
    assert sister == sister_front
    assert count == 12
    assert n_total == 24


def test_multi_lect_long_range_split_helper_silent_on_noise() -> None:
    """COMMITMENT: same helper commits nothing when observations
    have no long-range structure — every observation carries the
    same sister tuple, so no split improves entropy."""
    from regulae.training import (
        _commit_multi_lect_long_range_splits_for_pivot,
    )
    from regulae.types import FeatureConstraint

    same_sister = (("D1", "x"), ("D2", "x"))
    sister_by_key = {"only": same_sister}
    obs = [
        ("only", Context(next_syllable=(FeatureConstraint("front", "+"),)))
    ] * 10 + [("only", Context())] * 10
    committed: list = []
    _commit_multi_lect_long_range_splits_for_pivot(
        pivot_lect="PROTO",
        pivot_grapheme="a",
        observations=obs,
        sister_by_key=sister_by_key,
        committed=committed,
        min_commit_scale=0.5,
    )
    assert committed == []
