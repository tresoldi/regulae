"""Learned sound-correspondence rules for historical-linguistic inference."""

from regulae.anomaly import (
    HYPOTHESIS_KINDS,
    PatternHypothesis,
    find_residual_patterns,
)
from regulae.diagnostics import CognateOutlierReport, find_cognate_outliers
from regulae.format import (
    describe_cross_dimensional_rule,
    describe_multi_lect_class,
    describe_source,
    format_alignment,
    format_link,
    format_model,
    format_multi_lect_model,
    format_segments,
)
from regulae.loaders import load_arcaverborum, load_cognates_from_tsv, load_gled
from regulae.model import (
    ChunkPhraseTable,
    CognateSet,
    ConditionedCorrespondence,
    CrossDimensionalLink,
    CrossDimensionalLinkTable,
    DisplacementDistribution,
    LearnedModel,
    MultiLectCorrespondenceClass,
    MultiLectCrossDimensionalLink,
    MultiLectCrossDimensionalLinkTable,
    MultiLectModel,
    SegmentCorrespondenceTable,
    TonalCorrespondence,
    TonalCorrespondenceTable,
)
from regulae.scoring import UnknownGrapheme, compute_displacement, score_link
from regulae.search import align_forms, alignment_cost
from regulae.syllabification import compute_syllable_breaks
from regulae.training import align_corpus, cognate_sets_from_pairs, train_model
from regulae.types import (
    Alignment,
    Context,
    FeatureConstraint,
    FeatureDisplacement,
    Form,
    Lect,
    Link,
    Segment,
)

__all__ = [
    "Alignment",
    "ChunkPhraseTable",
    "CognateOutlierReport",
    "CognateSet",
    "ConditionedCorrespondence",
    "Context",
    "CrossDimensionalLink",
    "CrossDimensionalLinkTable",
    "DisplacementDistribution",
    "FeatureConstraint",
    "FeatureDisplacement",
    "Form",
    "HYPOTHESIS_KINDS",
    "Lect",
    "LearnedModel",
    "Link",
    "MultiLectCorrespondenceClass",
    "MultiLectCrossDimensionalLink",
    "MultiLectCrossDimensionalLinkTable",
    "MultiLectModel",
    "PatternHypothesis",
    "Segment",
    "SegmentCorrespondenceTable",
    "TonalCorrespondence",
    "TonalCorrespondenceTable",
    "UnknownGrapheme",
    "align_corpus",
    "align_forms",
    "alignment_cost",
    "cognate_sets_from_pairs",
    "compute_displacement",
    "compute_syllable_breaks",
    "describe_cross_dimensional_rule",
    "describe_multi_lect_class",
    "describe_source",
    "find_cognate_outliers",
    "find_residual_patterns",
    "format_alignment",
    "format_link",
    "format_model",
    "format_multi_lect_model",
    "format_segments",
    "load_arcaverborum",
    "load_cognates_from_tsv",
    "load_gled",
    "score_link",
    "train_model",
]
