"""Smoke tests for the diverse real-data corpora.

Parametrised tests that:

1. Verify each corpus can be loaded from its TSV and tokenised
   through the experiment's parse_form without UnknownGrapheme
   errors.
2. Verify training completes on the corpus.
3. Verify the trained model has non-trivial content (segment
   table is populated).

Pinned as COMMITMENT tests so future edits that break grapheme
coverage, parser logic, or training-stage flow surface
immediately.
"""

import importlib.util
from pathlib import Path

import pytest

from regulae import MultiLectModel, cognate_sets_from_pairs, train_model


_DATASETS = [
    ("finnish_estonian", "finnish", "estonian"),
    ("turkish_azerbaijani", "turkish", "azerbaijani"),
    ("arabic_hebrew", "arabic", "hebrew"),
    ("georgian_svan", "georgian", "svan"),
    ("mandarin_historical", "middle_chinese", "mandarin"),
    ("swahili_zulu", "swahili", "zulu"),
]


def _load(dataset: str):
    """Import the experiment's ``load_corpus`` and TSV, returning
    the (gloss, src_form, tgt_form) triples."""
    path = (
        Path(__file__).parent.parent / "experiments" / dataset / "run_experiment.py"
    )
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"_exp_{dataset}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tsv = path.parent / "cognates.tsv"
    return module.load_corpus(tsv)


@pytest.mark.parametrize("dataset,src_lect,tgt_lect", _DATASETS)
def test_corpus_loads_and_trains(dataset: str, src_lect: str, tgt_lect: str) -> None:
    """COMMITMENT: every diverse corpus loads without UnknownGrapheme
    errors and trains to a non-trivial MultiLectModel."""
    labeled = _load(dataset)
    if labeled is None:
        pytest.skip(f"{dataset} experiment not available yet")
    assert len(labeled) > 0, f"{dataset} corpus is empty"
    pairs = [(s, t) for _, s, t in labeled]
    corpus = cognate_sets_from_pairs(pairs, (src_lect, tgt_lect))
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({src_lect, tgt_lect})]
    # Non-trivial content: at least some segment correspondences learned.
    assert len(pair.segment_table.counts) > 0, (
        f"{dataset} produced empty segment table"
    )


@pytest.mark.parametrize("dataset,src_lect,tgt_lect", _DATASETS)
def test_corpus_produces_multi_lect_classes(
    dataset: str, src_lect: str, tgt_lect: str
) -> None:
    """The reconciliation layer produces at least one multi-lect
    correspondence class from each corpus."""
    labeled = _load(dataset)
    if labeled is None:
        pytest.skip(f"{dataset} experiment not available yet")
    pairs = [(s, t) for _, s, t in labeled]
    corpus = cognate_sets_from_pairs(pairs, (src_lect, tgt_lect))
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    assert len(model.unconditioned_classes) > 0


# ----- Mandarin-specific: tonogenesis commitment --------------------------


def test_mandarin_recovers_voicing_tonogenesis() -> None:
    """COMMITMENT: on the Middle-Chinese-to-Mandarin corpus, the
    cross-dimensional discovery committed at least one rule where
    voiced onsets predict tone 2 on the following vowel — the
    classic MC voiced-onset tonogenesis."""
    labeled = _load("mandarin_historical")
    if labeled is None:
        pytest.skip("mandarin_historical experiment not available")
    pairs = [(s, t) for _, s, t in labeled]
    corpus = cognate_sets_from_pairs(pairs, ("middle_chinese", "mandarin"))
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"middle_chinese", "mandarin"})]
    rules = pair.cross_dimensional_table.entries
    assert len(rules) >= 1, "expected at least one cross-dim rule"
    # Find a voicing → tone-2 rule.
    voicing_rules = [
        r for r in rules
        if r.src_feature.feature == "voiced"
        and r.src_feature.value == "+"
        and r.tgt_dimension == "tone"
        and r.tgt_value == "2"
    ]
    assert voicing_rules, (
        "expected voiced=+ → tone=2 rule (MC voiced-onset tonogenesis); "
        f"got rules: {[(r.src_feature, r.tgt_value) for r in rules]}"
    )


# ----- Contaminated-cognates fixture --------------------------------------


def test_contaminated_cognates_fixture_suppresses_noise() -> None:
    """COMMITMENT: bad pairs at confidence=0 contribute zero mass
    to the segment counts; clean pairs dominate."""
    labeled = _load("contaminated_cognates_synthetic")
    # This fixture uses CognateSet directly via its own loader; call the
    # experiment's load_corpus.
    path = (
        Path(__file__).parent.parent
        / "experiments/contaminated_cognates_synthetic/run_experiment.py"
    )
    spec = importlib.util.spec_from_file_location("_contam", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tsv = path.parent / "cognates.tsv"
    corpus = module.load_corpus(tsv)
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"proto", "derived"})]
    # p → f should dominate; p → {q, x, k, z} should not appear.
    p_tgts = {cc.tgt: c for cc, c in pair.segment_table.counts.items() if cc.src == "p"}
    assert p_tgts.get("f", 0.0) >= 15.0  # clean pairs contribute
    for noisy in ("q", "x", "z", "k"):
        assert p_tgts.get(noisy, 0.0) == 0.0, (
            f"p → {noisy} got nonzero count despite confidence=0 on bad pairs"
        )
