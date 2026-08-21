"""Dataclasses parsed from the model JSON the C core renders.

The wrapper does no modelling of its own: it reads exactly what
``regulae._native.train`` returns. Every field here mirrors a key in that
payload, so the schema is the C library's, not a second definition that could
drift from it. Unknown keys are ignored rather than rejected, so a newer core
that adds a field does not break an older wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class Uncertainty:
    """A rate and its interval, as the C ``rg_uncertainty_estimate``."""

    estimate: float
    lower: float
    upper: float
    n: float
    method: str
    observation_unit: str

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Uncertainty:
        return cls(
            estimate=data.get("estimate", 0.0),
            lower=data.get("lower", 0.0),
            upper=data.get("upper", 0.0),
            n=data.get("n", 0.0),
            method=data.get("method", ""),
            observation_unit=data.get("observation_unit", ""),
        )


@dataclass(frozen=True)
class PredictiveEvidence:
    """Held-out generalisation for one rule or the whole fit."""

    status: str
    folds: int
    log_loss_gain: float

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PredictiveEvidence:
        return cls(
            status=data.get("status", "unmeasured"),
            folds=int(data.get("folds", 0)),
            log_loss_gain=data.get("log_loss_gain", 0.0),
        )


@dataclass(frozen=True)
class Segment:
    """One lect's grapheme in a class, with its per-lect environment if any.

    ``grapheme`` is :data:`GAP_GRAPHEME` when the lect deleted the segment the
    others keep. ``context`` is present only on a conditioned class, giving the
    environment read from this lect's form.
    """

    lect: str
    grapheme: str
    context: dict[str, Any] | None = None

    @property
    def is_gap(self) -> bool:
        return self.grapheme == GAP_GRAPHEME


#: The grapheme a class carries for a lect that lost the segment (C
#: ``RG_GAP_GRAPHEME``). Compare against it to tell a deletion from a lect that
#: never had the word.
GAP_GRAPHEME = "∅"


@dataclass(frozen=True)
class CorrespondenceClass:
    """One multi-lect correspondence class, conditioned or not.

    ``contexts`` are empty on an unconditioned class. ``contrast_class_id`` is
    the class holding the pivot's other reflex out of the environment (-1 when
    there is none); ``supporting_cognates`` are the distinct cognate sets the
    class rests on -- read its length, not ``count``, for how many sets support
    it.
    """

    id: int
    segments: tuple[Segment, ...]
    count: float
    confidence: float
    contrast_count: float
    contrast_class_id: int
    contrast_alternative_count: float
    supporting_cognates: tuple[str, ...]
    decision_index: int
    standing: str
    uncertainty: Uncertainty
    predictive: PredictiveEvidence

    @property
    def conditioned(self) -> bool:
        return any(s.context for s in self.segments)

    @property
    def graphemes(self) -> dict[str, str]:
        """``lect -> grapheme`` mapping, gaps included as :data:`GAP_GRAPHEME`."""
        return {s.lect: s.grapheme for s in self.segments}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> CorrespondenceClass:
        return cls(
            id=int(data.get("id", -1)),
            segments=tuple(
                Segment(lect=s["lect"], grapheme=s["grapheme"], context=s.get("context"))
                for s in data.get("segments", ())
            ),
            count=data.get("count", 0.0),
            confidence=data.get("confidence", 0.0),
            contrast_count=data.get("contrast_count", 0.0),
            contrast_class_id=int(data.get("contrast_class_id", -1)),
            contrast_alternative_count=data.get("contrast_alternative_count", 0.0),
            supporting_cognates=tuple(data.get("supporting_cognates", ())),
            decision_index=int(data.get("decision_index", -1)),
            standing=data.get("standing", "unmeasured"),
            uncertainty=Uncertainty.from_json(data.get("uncertainty", {})),
            predictive=PredictiveEvidence.from_json(data.get("predictive", {})),
        )


@dataclass(frozen=True)
class CrossDimensionalRule:
    """A feature conditioning a suprasegmental dimension.

    ``dimension_from_environment`` names the rule shape: false is the cross-lect
    rule (one lect's onset predicts the other's tone), true is lect-internal
    tonogenesis (an onset and the tone it conditions in one lect). For a
    lect-internal rule ``conditioned_lect`` equals ``environment_lect``.
    """

    environment: dict[str, Any]
    context_is_target: bool
    dimension_from_environment: bool
    environment_lect: str
    conditioned_lect: str
    dimension: str
    value: str
    position_offset: int
    count: float
    source_count: float
    confidence: float
    contrast_count: float
    contrast_confidence: float
    uncertainty: Uncertainty
    predictive: PredictiveEvidence

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> CrossDimensionalRule:
        return cls(
            environment=dict(data.get("environment", {})),
            context_is_target=bool(data.get("context_is_target", False)),
            dimension_from_environment=bool(data.get("dimension_from_environment", False)),
            environment_lect=data.get("environment_lect", ""),
            conditioned_lect=data.get("conditioned_lect", ""),
            dimension=data.get("dimension", ""),
            value=data.get("value", ""),
            position_offset=int(data.get("position_offset", 0)),
            count=data.get("count", 0.0),
            source_count=data.get("source_count", 0.0),
            confidence=data.get("confidence", 0.0),
            contrast_count=data.get("contrast_count", 0.0),
            contrast_confidence=data.get("contrast_confidence", 0.0),
            uncertainty=Uncertainty.from_json(data.get("uncertainty", {})),
            predictive=PredictiveEvidence.from_json(data.get("predictive", {})),
        )


@dataclass(frozen=True)
class NullCorrespondence:
    """A segment answering to nothing, per pair: a correspondence like any other,
    only with :data:`GAP_GRAPHEME` on one side.

    ``source ~ target`` is ``g ~ ∅`` for a loss and ``∅ ~ g`` for an epenthesis;
    :attr:`deletion` reports which. ``count`` over the kept side's total (the one
    that is not ``∅``) is the rate the grapheme is dropped or inserted.
    """

    source: str
    target: str
    count: float
    source_total: float
    target_total: float

    @property
    def deletion(self) -> bool:
        return self.target == GAP_GRAPHEME

    @property
    def grapheme(self) -> str:
        """The segment on the side that keeps it."""
        return self.source if self.deletion else self.target

    @property
    def present_total(self) -> float:
        """How often the kept grapheme appears on its side; ``count`` over this
        is the loss (or epenthesis) rate."""
        return self.source_total if self.deletion else self.target_total

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> NullCorrespondence:
        return cls(
            source=data.get("source", ""),
            target=data.get("target", ""),
            count=data.get("count", 0.0),
            source_total=data.get("source_total", 0.0),
            target_total=data.get("target_total", 0.0),
        )


@dataclass(frozen=True)
class ConditionedCorrespondence:
    """One pair's context-conditioned segment correspondence."""

    source: str
    target: str
    context: dict[str, Any]
    context_is_target: bool
    count: float
    source_total: float
    contrast_count: float

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ConditionedCorrespondence:
        return cls(
            source=data.get("source", ""),
            target=data.get("target", ""),
            context=dict(data.get("context", {})),
            context_is_target=bool(data.get("context_is_target", False)),
            count=data.get("count", 0.0),
            source_total=data.get("source_total", 0.0),
            contrast_count=data.get("contrast_count", 0.0),
        )


@dataclass(frozen=True)
class PairwiseModel:
    """One lect pair's tables: conditioned correspondences and correspondences
    to ∅ (losses and epentheses)."""

    source_lect: str
    target_lect: str
    conditioned: tuple[ConditionedCorrespondence, ...]
    null_correspondences: tuple[NullCorrespondence, ...]

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PairwiseModel:
        return cls(
            source_lect=data.get("source_lect", ""),
            target_lect=data.get("target_lect", ""),
            conditioned=tuple(
                ConditionedCorrespondence.from_json(r) for r in data.get("conditioned", ())
            ),
            null_correspondences=tuple(
                NullCorrespondence.from_json(r) for r in data.get("null_correspondences", ())
            ),
        )


@dataclass(frozen=True)
class MultiLectModel:
    """A trained model, parsed from the C core's JSON.

    The raw payload is kept on :attr:`raw` so a consumer can reach any field the
    dataclasses do not surface, without the wrapper having to mirror every one.
    """

    lects: tuple[str, ...]
    unconditioned_classes: tuple[CorrespondenceClass, ...]
    conditioned_classes: tuple[CorrespondenceClass, ...]
    cross_dimensional: tuple[CrossDimensionalRule, ...]
    pairwise_models: tuple[PairwiseModel, ...]
    fit: dict[str, Any]
    regulae_version: str
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> MultiLectModel:
        classes = data.get("classes", {})
        return cls(
            lects=tuple(data.get("lects", ())),
            unconditioned_classes=tuple(
                CorrespondenceClass.from_json(c) for c in classes.get("unconditioned", ())
            ),
            conditioned_classes=tuple(
                CorrespondenceClass.from_json(c) for c in classes.get("conditioned", ())
            ),
            cross_dimensional=tuple(
                CrossDimensionalRule.from_json(r) for r in data.get("cross_dimensional", ())
            ),
            pairwise_models=tuple(
                PairwiseModel.from_json(p) for p in data.get("pairwise", ())
            ),
            fit=dict(data.get("fit", {})),
            regulae_version=data.get("regulae_version", ""),
            raw=dict(data),
        )

    @property
    def classes(self) -> Sequence[CorrespondenceClass]:
        """Every class, unconditioned then conditioned."""
        return self.unconditioned_classes + self.conditioned_classes
