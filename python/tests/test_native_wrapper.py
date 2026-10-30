"""The Python wrapper is a thin layer over the C core, so these tests check the
seam: that a corpus trains, that the JSON parses into the dataclasses, and that
the fields this project added at the C level (distinct supporting sets, the
contrast link, gap classes, lect-internal cross-dimensional rules) survive the
crossing. The modelling itself is tested against the C, not here.
"""

from __future__ import annotations

import pytest
import regulae

# A minimal cognate table: A keeps a final -n, B drops it (a deletion), and both
# keep the onset so nothing else moves.
DELETION_TSV = "cognate_id\tlect_id\tsegments\n" + "\n".join(
    f"w{i}\t{lect}\t{seg}"
    for i, (a, b) in enumerate(
        [("a p a n", "a p a"), ("a t a n", "a t a"), ("a k a n", "a k a"),
         ("a m a n", "a m a"), ("a s a n", "a s a"), ("a l a n", "a l a")]
    )
    for lect, seg in (("A", a), ("B", b))
)


def test_version_and_abi() -> None:
    assert regulae.version()
    assert regulae.abi_version() >= 34


def test_train_returns_a_model() -> None:
    model = regulae.train_model(DELETION_TSV, fmt="tsv")
    assert isinstance(model, regulae.MultiLectModel)
    assert set(model.lects) == {"A", "B"}
    assert model.unconditioned_classes


def test_supporting_cognates_are_distinct_sets() -> None:
    model = regulae.train_model(DELETION_TSV, fmt="tsv")
    for klass in model.unconditioned_classes:
        # A set is listed once, so its length is a count of sets, not positions.
        assert len(set(klass.supporting_cognates)) == len(klass.supporting_cognates)


def test_deletion_has_a_class_row() -> None:
    model = regulae.train_model(DELETION_TSV, fmt="tsv")
    deletion = [
        c for c in model.unconditioned_classes
        if c.graphemes.get("A") == "n" and c.graphemes.get("B") == regulae.GAP_GRAPHEME
    ]
    assert len(deletion) == 1
    assert deletion[0].count == 6.0
    assert any(s.is_gap for s in deletion[0].segments)


def test_null_correspondence_states_the_loss() -> None:
    model = regulae.train_model(DELETION_TSV, fmt="tsv")
    losses = [
        g
        for p in model.pairwise_models
        for g in p.null_correspondences
        if g.deletion and g.grapheme == "n"
    ]
    assert losses
    assert losses[0].source == "n"
    assert losses[0].target == regulae.GAP_GRAPHEME
    assert losses[0].count == losses[0].present_total  # dropped every time


def test_verner_conditioned_classes_carry_a_contrast_link(tmp_path: object) -> None:
    import pathlib

    verner = pathlib.Path(__file__).resolve().parents[2] / "testdata" / "soundlaws" / "verner.tsv"
    model = regulae.train_model(verner, fmt="tsv")
    assert model.conditioned_classes
    for klass in model.conditioned_classes:
        assert klass.conditioned
        # A conditioned class points at the row holding the pivot's other reflex.
        assert klass.contrast_class_id >= 0


def test_lect_internal_tonogenesis_is_stated() -> None:
    internal = "cognate_id\tlect_id\tsegments\n" + "\n".join(
        f"w{i}\t{lect}\t{seg}"
        for i, (p, d) in enumerate(
            [(f"{ons} {v}", f"{ons} {v} {tone}")
             for ons, tone in (("b", "¹¹"), ("p", "⁵⁵"),
                               ("d", "¹¹"), ("t", "⁵⁵"),
                               ("g", "¹¹"), ("k", "⁵⁵"))
             for v in ("a", "e", "i", "o", "u")]
        )
        for lect, seg in (("proto", p), ("daughter", d))
    )
    model = regulae.train_model(internal, fmt="tsv")
    lect_internal = [r for r in model.cross_dimensional if r.dimension_from_environment]
    assert lect_internal
    # An internal rule conditions the tone in the lect that states the onset.
    for rule in lect_internal:
        assert rule.environment_lect == rule.conditioned_lect == "daughter"


def test_unknown_grapheme_raises() -> None:
    with pytest.raises(regulae.RegulaeError):
        regulae.train_model("cognate_id\tlect_id\tsegments\nw\tA\tQ\nw\tB\tZ\n", fmt="tsv")


def test_segment_lifts_tone() -> None:
    assert regulae.segment("t a⁵⁵") == ["t", "a"]


def test_train_ensemble_returns_a_c1_package() -> None:
    import pathlib

    corpus = pathlib.Path(__file__).resolve().parents[2] / "testdata" / "corpora" / "real_latin_spanish.tsv"
    package = regulae.train_ensemble(corpus, fmt="tsv", members=3, temperature=0.25, partitions=True)
    assert package["export_kind"] == "regulae_package"
    assert package["approx_hint"] == "C1"
    assert package["ensemble"]["members"] == 3
    assert len(package["members"]) == 3
    assert all("partition" in m for m in package["members"])
    inclusion = [c["inclusion"] for c in package["summary"]["classes"]]
    assert inclusion and all(0.0 < p <= 1.0 for p in inclusion)
    again = regulae.train_ensemble(corpus, fmt="tsv", members=3, temperature=0.25, partitions=True)
    assert again == package


def test_train_ensemble_with_soft_cognacy(tmp_path) -> None:
    forms = ["form_id\tlect_id\tmeaning\tsegments"]
    pairs = ["form_a\tform_b\tweight"]
    words = [("p a t a", "p a d a"), ("k a p a", "k a b a"), ("t o p a", "t o b a"),
             ("m a t a", "m a d a"), ("l u p a", "l u b a"), ("s a p o", "s a b o")]
    for i, (old, new) in enumerate(words):
        forms.append(f"o{i}\told\tm{i}\t{old}")
        forms.append(f"n{i}\tnew\tm{i}\t{new}")
        pairs.append(f"o{i}\tn{i}\t0.8")
    (tmp_path / "forms.tsv").write_text("\n".join(forms) + "\n", encoding="utf-8")
    (tmp_path / "pairs.tsv").write_text("\n".join(pairs) + "\n", encoding="utf-8")
    package = regulae.train_ensemble(
        tmp_path / "forms.tsv", cognacy_pairs=tmp_path / "pairs.tsv", members=4, temperature=0
    )
    assert package["condition_on"] == "sampled_partitions"
    counts = {len(m["cognate_sets"]) for m in package["members"]}
    assert all(1 <= c <= len(words) for c in counts)
    for member in package["members"]:
        for cognate_set in member["cognate_sets"]:
            assert len(cognate_set["forms"].split()) == 2


def test_soft_em_option_crosses_the_seam() -> None:
    model = regulae.train_model(
        DELETION_TSV, fmt="tsv", options={"alignment_posterior_temperature": 0.1}
    )
    assert model.unconditioned_classes
    with pytest.raises(regulae.RegulaeError):
        regulae.train_model(
            DELETION_TSV, fmt="tsv", options={"alignment_posterior_temperature": -1.0}
        )
