package regulae

import "strings"

// Core data types for phonological alignment.
//
// All types here are immutable by convention: construct them, compare them,
// but do not mutate their fields in place. Operations live in sibling files
// (scoring, search, etc.).
//
// The Python original modeled these as frozen dataclasses; several were used
// as dictionary keys. In Go, the scalar-only structs (Segment,
// FeatureConstraint, FeatureDisplacement) are naturally comparable and usable
// as map keys directly. The slice-bearing structs (Context, Form, Link) are
// not comparable, so each carries a canonical key() method that builds a
// deterministic, order-preserving string for map use.

// Optional string fields (Segment.Tone/Length/Stress) use "" to mean "absent"
// (Python's None). Those dimensions never legitimately carry an empty
// non-absent value, so the sentinel is unambiguous and keeps Segment
// comparable.

// keyField separators. Unit/record separators are used because phonological
// tokens are ASCII linguistic strings that never contain control characters.
const (
	keySep   = "\x1f" // between elements within a field
	fieldSep = "\x1e" // between fields
)

// Segment is a single phonological position: an IPA grapheme plus optional
// suprasegmental annotations (tone, length, stress) stored separately so the
// framework can treat them as independent feature dimensions. When
// interfacing with merkmal, only Grapheme is passed.
type Segment struct {
	Grapheme string
	Tone     string
	Length   string
	Stress   string
}

// Form is a phonological form in some lect: a sequence of segments, optionally
// annotated with syllable and morpheme break indices. Index 0 is an implied
// syllable and morpheme start. A morpheme break b means the morpheme starts at
// index b (boundary between segment b-1 and b); chunk promotion uses these to
// reject chunks that straddle morpheme boundaries.
type Form struct {
	LectID         string
	Segments       []Segment
	SyllableBreaks []int
	MorphemeBreaks []int
}

// Lect is a structurally defined linguistic system: an identifier plus the
// merkmal feature system used to interpret its segments.
type Lect struct {
	LectID        string
	FeatureSystem string // default "descriptive"
}

// NewLect constructs a Lect, defaulting the feature system to "descriptive".
func NewLect(lectID, featureSystem string) Lect {
	if featureSystem == "" {
		featureSystem = "descriptive"
	}
	return Lect{LectID: lectID, FeatureSystem: featureSystem}
}

// FeatureConstraint is a named constraint on a single phonological feature,
// used in conditioning contexts. For privative features Value may be
// "present" or "absent".
type FeatureConstraint struct {
	Feature string
	Value   string
}

// distanceConstraint pairs a positive integer offset with a feature
// constraint, for the long-range preceding_at_distance / following_at_distance
// context slots. Comparable, so usable in membership checks.
type distanceConstraint struct {
	Offset     int
	Constraint FeatureConstraint
}

// Context carries the conditioning environment for a correspondence link.
// Empty slices mean "no constraint at this slot"; an empty Position or
// Morphological string means unconstrained.
type Context struct {
	Position      string
	Preceding     []FeatureConstraint
	Following     []FeatureConstraint
	Morphological string

	PrecedingAtDistance []distanceConstraint
	FollowingAtDistance []distanceConstraint
	SomewherePreceding  []FeatureConstraint
	SomewhereFollowing  []FeatureConstraint
	SameSyllable        []FeatureConstraint
	NextSyllable        []FeatureConstraint
	PreviousSyllable    []FeatureConstraint

	SelfStress      []FeatureConstraint
	PrecedingStress []FeatureConstraint
	FollowingStress []FeatureConstraint
}

// containsConstraint reports whether c appears in the slice.
func containsConstraint(slice []FeatureConstraint, c FeatureConstraint) bool {
	for _, x := range slice {
		if x == c {
			return true
		}
	}
	return false
}

// containsDistanceConstraint reports whether c appears in the slice.
func containsDistanceConstraint(slice []distanceConstraint, c distanceConstraint) bool {
	for _, x := range slice {
		if x == c {
			return true
		}
	}
	return false
}

// IsSubsetOf reports whether every constraint in c is satisfied by other.
// A table entry with context cTable matches a link context cLink iff
// cTable.IsSubsetOf(cLink). Used at scoring time to find the most specific
// matching conditioned correspondence.
func (c Context) IsSubsetOf(other Context) bool {
	if c.Position != "" && c.Position != other.Position {
		return false
	}
	if c.Morphological != "" && c.Morphological != other.Morphological {
		return false
	}
	for _, x := range c.Preceding {
		if !containsConstraint(other.Preceding, x) {
			return false
		}
	}
	for _, x := range c.Following {
		if !containsConstraint(other.Following, x) {
			return false
		}
	}
	for _, x := range c.PrecedingAtDistance {
		if !containsDistanceConstraint(other.PrecedingAtDistance, x) {
			return false
		}
	}
	for _, x := range c.FollowingAtDistance {
		if !containsDistanceConstraint(other.FollowingAtDistance, x) {
			return false
		}
	}
	for _, x := range c.SomewherePreceding {
		if !containsConstraint(other.SomewherePreceding, x) {
			return false
		}
	}
	for _, x := range c.SomewhereFollowing {
		if !containsConstraint(other.SomewhereFollowing, x) {
			return false
		}
	}
	for _, x := range c.SameSyllable {
		if !containsConstraint(other.SameSyllable, x) {
			return false
		}
	}
	for _, x := range c.NextSyllable {
		if !containsConstraint(other.NextSyllable, x) {
			return false
		}
	}
	for _, x := range c.PreviousSyllable {
		if !containsConstraint(other.PreviousSyllable, x) {
			return false
		}
	}
	for _, x := range c.SelfStress {
		if !containsConstraint(other.SelfStress, x) {
			return false
		}
	}
	for _, x := range c.PrecedingStress {
		if !containsConstraint(other.PrecedingStress, x) {
			return false
		}
	}
	for _, x := range c.FollowingStress {
		if !containsConstraint(other.FollowingStress, x) {
			return false
		}
	}
	return true
}

// ConstraintCount returns the number of non-trivial constraints. Used at
// scoring time to pick the most specific match; higher = more specific.
func (c Context) ConstraintCount() int {
	count := 0
	if c.Position != "" {
		count++
	}
	if c.Morphological != "" {
		count++
	}
	count += len(c.Preceding)
	count += len(c.Following)
	count += len(c.PrecedingAtDistance)
	count += len(c.FollowingAtDistance)
	count += len(c.SomewherePreceding)
	count += len(c.SomewhereFollowing)
	count += len(c.SameSyllable)
	count += len(c.NextSyllable)
	count += len(c.PreviousSyllable)
	count += len(c.SelfStress)
	count += len(c.PrecedingStress)
	count += len(c.FollowingStress)
	return count
}

// key returns a canonical, order-preserving string for use as a map key.
// Two Contexts produce the same key iff they have identical field values in
// identical order, matching Python frozen-dataclass dict-key semantics.
func (c Context) key() string {
	var b strings.Builder
	b.WriteString(c.Position)
	b.WriteString(fieldSep)
	writeConstraints(&b, c.Preceding)
	b.WriteString(fieldSep)
	writeConstraints(&b, c.Following)
	b.WriteString(fieldSep)
	b.WriteString(c.Morphological)
	b.WriteString(fieldSep)
	writeDistanceConstraints(&b, c.PrecedingAtDistance)
	b.WriteString(fieldSep)
	writeDistanceConstraints(&b, c.FollowingAtDistance)
	b.WriteString(fieldSep)
	writeConstraints(&b, c.SomewherePreceding)
	b.WriteString(fieldSep)
	writeConstraints(&b, c.SomewhereFollowing)
	b.WriteString(fieldSep)
	writeConstraints(&b, c.SameSyllable)
	b.WriteString(fieldSep)
	writeConstraints(&b, c.NextSyllable)
	b.WriteString(fieldSep)
	writeConstraints(&b, c.PreviousSyllable)
	b.WriteString(fieldSep)
	writeConstraints(&b, c.SelfStress)
	b.WriteString(fieldSep)
	writeConstraints(&b, c.PrecedingStress)
	b.WriteString(fieldSep)
	writeConstraints(&b, c.FollowingStress)
	return b.String()
}

func writeConstraints(b *strings.Builder, cs []FeatureConstraint) {
	for i, c := range cs {
		if i > 0 {
			b.WriteString(keySep)
		}
		b.WriteString(c.Feature)
		b.WriteString("=")
		b.WriteString(c.Value)
	}
}

func writeDistanceConstraints(b *strings.Builder, cs []distanceConstraint) {
	for i, c := range cs {
		if i > 0 {
			b.WriteString(keySep)
		}
		b.WriteByte(byte('0' + c.Offset%10))
		b.WriteString(":")
		b.WriteString(c.Constraint.Feature)
		b.WriteString("=")
		b.WriteString(c.Constraint.Value)
	}
}

// FeatureDisplacement is a change in a single feature dimension between two
// segments or chunks. For privative features use "present"/"absent". Two
// correspondences with the same displacement vector are formally the same kind
// of correspondence — this is how feature-level regularities are detected
// across natural classes (e.g. Grimm-type p~f, t~θ, k~x share stop→fricative).
type FeatureDisplacement struct {
	Feature   string
	FromValue string
	ToValue   string
}

// Link is a phrase-based correspondence between chunks of two lects. Either
// chunk may be empty (0-to-N or N-to-0). FeatureDisplacement is the structured
// signature, stored so it can be inspected without recomputation. Confidence
// is in [0, 1]; 1.0 means certain.
type Link struct {
	SourceChunk         []Segment
	TargetChunk         []Segment
	Context             Context
	FeatureDisplacement []FeatureDisplacement
	Confidence          float64
}

// NewLink builds a Link with Confidence defaulting to 1.0.
func NewLink(source, target []Segment, ctx Context, disp []FeatureDisplacement) Link {
	return Link{
		SourceChunk:         source,
		TargetChunk:         target,
		Context:             ctx,
		FeatureDisplacement: disp,
		Confidence:          1.0,
	}
}

// Alignment is a complete alignment between two forms: an ordered sequence of
// links that should cover both forms exhaustively. The invariant is the
// responsibility of the alignment search, not enforced here.
type Alignment struct {
	SourceForm Form
	TargetForm Form
	Links      []Link
}

// segmentsKey builds a canonical string key for a slice of segments.
func segmentsKey(segs []Segment) string {
	var b strings.Builder
	for i, s := range segs {
		if i > 0 {
			b.WriteString(keySep)
		}
		b.WriteString(s.Grapheme)
		b.WriteString("|")
		b.WriteString(s.Tone)
		b.WriteString("|")
		b.WriteString(s.Length)
		b.WriteString("|")
		b.WriteString(s.Stress)
	}
	return b.String()
}
