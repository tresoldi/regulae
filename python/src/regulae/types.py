"""Core data types for phonological alignment.

All types here are immutable (frozen dataclasses). They carry no behavior
beyond construction and equality. Operations on these types live in
sibling modules (scoring, alignment, etc.).

Design notes:

* A ``Segment`` is a single IPA grapheme plus optional suprasegmental
  annotations (tone, length, stress). This is a richer-than-merkmal
  representation; merkmal works at the segment level and stores tone on
  the tone-bearing unit. We separate the suprasegmental annotations so the
  framework can reason about them as cross-dimensional correspondence
  targets (cf. tonogenesis: a consonantal voicing contrast in one lect
  corresponds to a tonal contrast on the vowel in another).

* A ``Form`` is a sequence of segments, optionally annotated with syllable
  break indices. No morphological structure yet (that comes from a separate
  segmentation layer).

* A ``Link`` is a phrase-based correspondence: chunks of segments on each
  side, plus a conditioning context, plus the feature displacement vector
  computed from the chunks. The displacement is the structured signature of
  the correspondence: what features differ, by how much. This is what makes
  feature-level regularities discoverable across natural classes.

* A ``Context`` carries the conditioning environment for a link. Constraints
  are named (FeatureConstraint, not bare strings) so they can be
  introspected and manipulated. Combining context (multiple constraints
  active simultaneously) is supported via the conjunctive lists.

* Correspondences are bidirectional at this stage. Diachrony (who came
  from whom) is a separate later inference. The scoring function is
  symmetric in the two chunks.
"""

from dataclasses import dataclass, field
from typing import Literal, NewType


# ----- domain Literals ---------------------------------------------------
#
# Closed string vocabularies. Use these in dataclass fields and function
# signatures that accept / return one of the enumerated values so mypy
# rejects typos and unexpected values.

TargetDimension = Literal["grapheme", "tone", "length", "stress"]
HypothesisKind = Literal["cross_dimensional_tonogenesis"]
ProcessProfile = Literal[
    "nasal_fusion",
    "glide_or_vocalization_fusion",
    "compact_fusion",
    "residual_reduction",
    "balanced_restructuring",
    "bundled_reduction",
    "mixed_or_unclear",
]


# ----- domain NewTypes ---------------------------------------------------
#
# Transparent at runtime (each is just ``str`` under the hood) but carry a
# distinct static type so mypy catches e.g. passing a grapheme where a lect
# id is expected. Coercion happens at loader boundaries — see
# :mod:`regulae.loaders`. Internal code that must build one of these from a
# bare string must call the corresponding constructor explicitly.

LectId = NewType("LectId", str)
CognateId = NewType("CognateId", str)
Grapheme = NewType("Grapheme", str)
FeatureName = NewType("FeatureName", str)
FeatureValue = NewType("FeatureValue", str)
Tone = NewType("Tone", str)
SlotName = NewType("SlotName", str)


@dataclass(frozen=True)
class Segment:
    """A single phonological position: an IPA grapheme plus suprasegmental annotations.

    The grapheme is the base IPA symbol (e.g., ``"p"``, ``"a"``, ``"t͡ʃ"``).
    Suprasegmental annotations (tone, length, stress) are stored separately
    so the framework can reason about them as independent feature dimensions.

    When interfacing with merkmal, only the grapheme is passed (merkmal
    operates at the segmental level). The suprasegmental fields are used by
    higher layers (system-level pattern detection, cross-dimensional
    correspondence discovery).
    """

    grapheme: Grapheme
    tone: Tone | None = None
    length: str | None = None
    stress: str | None = None


@dataclass(frozen=True)
class Form:
    """A phonological form in some lect: a sequence of segments.

    Optionally carries:

    * ``syllable_breaks``: positions where a new syllable starts;
      index 0 is implied as a syllable start.
    * ``morpheme_breaks``: positions where a new morpheme starts;
      index 0 is implied as a stem-initial position. Each entry is
      a position ``b`` meaning the morpheme starts at index ``b``
      (so the boundary lies between segment ``b-1`` and segment
      ``b``). Used by chunk promotion to reject chunks that
      straddle morpheme boundaries — those are morphological
      artifacts, not phonological correspondences.
    """

    lect_id: LectId
    segments: tuple[Segment, ...]
    syllable_breaks: tuple[int, ...] = ()
    morpheme_breaks: tuple[int, ...] = ()


@dataclass(frozen=True)
class Lect:
    """A lect: a structurally defined linguistic system.

    Minimal version — just an identifier and the merkmal feature system to
    use when interpreting its segments. The full lect representation
    (contrast graph, alternations, paradigms) lives in higher layers.
    """

    lect_id: LectId
    feature_system: str = "descriptive"


@dataclass(frozen=True)
class FeatureConstraint:
    """A named constraint on a single phonological feature.

    Used in conditioning contexts. The ``feature`` is a feature name from
    the relevant merkmal feature system; ``value`` is the expected value
    (or polarity, for binary features). For privative features, ``value``
    may be ``"present"`` or ``"absent"``.
    """

    feature: FeatureName
    value: FeatureValue


@dataclass(frozen=True)
class Context:
    """The conditioning context for a correspondence link.

    Each field constrains a different aspect of the environment:

    * ``position``: where in the word (initial, medial, final, or any).
    * ``preceding``: feature constraints on the immediately preceding
      segment. A non-empty tuple is a conjunction (all must hold).
    * ``following``: feature constraints on the immediately following
      segment. Same conjunctive semantics.
    * ``morphological``: morphological position (stem, prefix, suffix,
      boundary, or any).

    Long-range fields:

    * ``preceding_at_distance``: constraints keyed by a positive integer
      offset, meaning "the segment this many positions to the left of the
      link." Offset 1 is equivalent to ``preceding``; the long-range
      loop uses offsets {2, 3}.
    * ``following_at_distance``: same, to the right.
    * ``somewhere_preceding``: existential constraints — "some segment
      strictly before the link satisfies this." Position-agnostic.
    * ``somewhere_following``: same, after.
    * ``same_syllable``: constraints on any segment in the same syllable
      as the link's first position (excluding the link itself).
    * ``next_syllable``: constraints on any segment in the syllable after
      the link's last position.
    * ``previous_syllable``: constraints on any segment in the syllable
      before the link's first position.

    Suprasegmental slots (populated from ``Segment.stress`` directly,
    not via merkmal features):

    * ``self_stress``: constraints on the link's own segment's stress
      value. Only meaningful for 1-to-1 links; left empty for chunks.
    * ``preceding_stress``: constraints on the immediately preceding
      segment's stress value. Same 1-to-1 restriction.
    * ``following_stress``: same, for the immediately following
      segment.

    Empty tuples mean "no constraint at this slot." ``None`` for
    ``position`` or ``morphological`` likewise means unconstrained.
    """

    position: SlotName | None = None
    preceding: tuple[FeatureConstraint, ...] = ()
    following: tuple[FeatureConstraint, ...] = ()
    morphological: SlotName | None = None
    preceding_at_distance: tuple[tuple[int, FeatureConstraint], ...] = ()
    following_at_distance: tuple[tuple[int, FeatureConstraint], ...] = ()
    somewhere_preceding: tuple[FeatureConstraint, ...] = ()
    somewhere_following: tuple[FeatureConstraint, ...] = ()
    same_syllable: tuple[FeatureConstraint, ...] = ()
    next_syllable: tuple[FeatureConstraint, ...] = ()
    previous_syllable: tuple[FeatureConstraint, ...] = ()
    self_stress: tuple[FeatureConstraint, ...] = ()
    preceding_stress: tuple[FeatureConstraint, ...] = ()
    following_stress: tuple[FeatureConstraint, ...] = ()

    def is_subset_of(self, other: "Context") -> bool:
        """Return True if every constraint in ``self`` is satisfied by
        ``other``.

        Used at scoring time to look up the most specific matching
        conditioned correspondence. A table entry with context ``C_table``
        matches a link context ``C_link`` iff
        ``C_table.is_subset_of(C_link)``.

        Semantics:

        * If ``self.position`` is set, ``other.position`` must equal it.
          If ``self.position`` is None, no constraint on other.
        * Every :class:`FeatureConstraint` in ``self.preceding`` must
          appear in ``other.preceding``. Likewise for ``following``.
        * The morphological field works like position.
        """
        if self.position is not None and self.position != other.position:
            return False
        if self.morphological is not None and self.morphological != other.morphological:
            return False
        if any(c not in other.preceding for c in self.preceding):
            return False
        if any(c not in other.following for c in self.following):
            return False
        if any(c not in other.preceding_at_distance for c in self.preceding_at_distance):
            return False
        if any(c not in other.following_at_distance for c in self.following_at_distance):
            return False
        if any(c not in other.somewhere_preceding for c in self.somewhere_preceding):
            return False
        if any(c not in other.somewhere_following for c in self.somewhere_following):
            return False
        if any(c not in other.same_syllable for c in self.same_syllable):
            return False
        if any(c not in other.next_syllable for c in self.next_syllable):
            return False
        if any(c not in other.previous_syllable for c in self.previous_syllable):
            return False
        if any(c not in other.self_stress for c in self.self_stress):
            return False
        if any(c not in other.preceding_stress for c in self.preceding_stress):
            return False
        if any(c not in other.following_stress for c in self.following_stress):
            return False
        return True

    def constraint_count(self) -> int:
        """Number of non-trivial constraints.

        Used at scoring time to pick the most specific match when
        multiple conditioned correspondences apply to the same link.
        Higher count = more specific.
        """
        count = 0
        if self.position is not None:
            count += 1
        if self.morphological is not None:
            count += 1
        count += len(self.preceding)
        count += len(self.following)
        count += len(self.preceding_at_distance)
        count += len(self.following_at_distance)
        count += len(self.somewhere_preceding)
        count += len(self.somewhere_following)
        count += len(self.same_syllable)
        count += len(self.next_syllable)
        count += len(self.previous_syllable)
        count += len(self.self_stress)
        count += len(self.preceding_stress)
        count += len(self.following_stress)
        return count


@dataclass(frozen=True)
class FeatureDisplacement:
    """A change in a single feature dimension between two segments or chunks.

    For categorical features with explicit values, ``from_value`` and
    ``to_value`` carry those values. For privative features (presence /
    absence), use ``"present"`` and ``"absent"``.

    Two correspondences with the same displacement vector are formally the
    same kind of correspondence — this is how feature-level regularities
    are detected across natural classes (e.g., Grimm-type p~f, t~θ, k~x
    all share ``{stop → fricative}``).
    """

    feature: FeatureName
    from_value: FeatureValue
    to_value: FeatureValue


@dataclass(frozen=True)
class Link:
    """A correspondence link between chunks of two lects.

    Either chunk may be empty (representing 0-to-N or N-to-0 links:
    insertion / deletion at the level of the alignment, NOT at the level
    of historical change — direction is not claimed at this stage).

    The ``feature_displacement`` is the structured signature of the
    correspondence — typically computed from the chunks via merkmal, but
    stored here so it can be inspected and compared without recomputation.

    Confidence is in [0, 1]; 1.0 means "this link is certain."
    """

    source_chunk: tuple[Segment, ...]
    target_chunk: tuple[Segment, ...]
    context: Context = field(default_factory=Context)
    feature_displacement: tuple[FeatureDisplacement, ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True)
class Alignment:
    """A complete alignment between two forms: an ordered sequence of links.

    The links should cover both forms exhaustively (every segment of either
    form belongs to exactly one link). This invariant is not enforced by
    the dataclass itself — the alignment search procedure is responsible
    for producing well-formed alignments.
    """

    source_form: Form
    target_form: Form
    links: tuple[Link, ...]
