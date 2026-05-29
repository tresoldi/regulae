package regulae

import "testing"

// Mirror of python/tests/test_types.py

// ----- Segment -------------------------------------------------------------

func TestTypesSegmentCarriesGraphemeOnlyWhenUnannotated(t *testing.T) {
	s := Segment{Grapheme: "p"}
	if s.Grapheme != "p" {
		t.Errorf("grapheme = %q, want %q", s.Grapheme, "p")
	}
	if s.Tone != "" {
		t.Errorf("expected empty tone, got %q", s.Tone)
	}
	if s.Length != "" {
		t.Errorf("expected empty length, got %q", s.Length)
	}
	if s.Stress != "" {
		t.Errorf("expected empty stress, got %q", s.Stress)
	}
}

func TestTypesSegmentCarriesSuprasegmentalAnnotations(t *testing.T) {
	s := Segment{Grapheme: "a", Tone: "high", Length: "long", Stress: "primary"}
	if s.Tone != "high" {
		t.Errorf("tone = %q, want %q", s.Tone, "high")
	}
	if s.Length != "long" {
		t.Errorf("length = %q, want %q", s.Length, "long")
	}
	if s.Stress != "primary" {
		t.Errorf("stress = %q, want %q", s.Stress, "primary")
	}
}

// (skipped: test_segments_are_immutable — Python-internal frozen dataclass
// behaviour; Go structs do not enforce immutability at compile time.)

func TestTypesSegmentsWithSameFieldsAreEqual(t *testing.T) {
	if (Segment{Grapheme: "p"}) != (Segment{Grapheme: "p"}) {
		t.Error("identical segments should be equal")
	}
	if (Segment{Grapheme: "a", Tone: "high"}) != (Segment{Grapheme: "a", Tone: "high"}) {
		t.Error("identical toned segments should be equal")
	}
	if (Segment{Grapheme: "a", Tone: "high"}) == (Segment{Grapheme: "a", Tone: "low"}) {
		t.Error("segments with different tones should not be equal")
	}
}

// ----- Form ----------------------------------------------------------------

func TestTypesFormHoldsSegmentsAndLectID(t *testing.T) {
	f := Form{
		LectID:   "latin",
		Segments: []Segment{{Grapheme: "p"}, {Grapheme: "a"}, {Grapheme: "t"}, {Grapheme: "e"}, {Grapheme: "r"}},
	}
	if f.LectID != "latin" {
		t.Errorf("lect_id = %q, want %q", f.LectID, "latin")
	}
	if len(f.Segments) != 5 {
		t.Errorf("len(segments) = %d, want 5", len(f.Segments))
	}
	if f.Segments[0].Grapheme != "p" {
		t.Errorf("segments[0] = %q, want %q", f.Segments[0].Grapheme, "p")
	}
}

func TestTypesFormSupportsSyllableBreaks(t *testing.T) {
	// pa.ter — syllable starts at index 0 and at index 2
	f := Form{
		LectID:         "latin",
		Segments:       []Segment{{Grapheme: "p"}, {Grapheme: "a"}, {Grapheme: "t"}, {Grapheme: "e"}, {Grapheme: "r"}},
		SyllableBreaks: []int{0, 2},
	}
	if len(f.SyllableBreaks) != 2 || f.SyllableBreaks[0] != 0 || f.SyllableBreaks[1] != 2 {
		t.Errorf("syllable_breaks = %v, want [0 2]", f.SyllableBreaks)
	}
}

// ----- Lect ----------------------------------------------------------------

func TestTypesLectHasIDAndDefaultFeatureSystem(t *testing.T) {
	lect := NewLect("latin", "")
	if lect.LectID != "latin" {
		t.Errorf("lect_id = %q, want %q", lect.LectID, "latin")
	}
	if lect.FeatureSystem != "descriptive" {
		t.Errorf("feature_system = %q, want %q", lect.FeatureSystem, "descriptive")
	}
}

// ----- Context and FeatureConstraint --------------------------------------

func TestTypesEmptyContextImposesNoConstraints(t *testing.T) {
	c := Context{}
	if c.Position != "" {
		t.Errorf("position should be empty, got %q", c.Position)
	}
	if len(c.Preceding) != 0 {
		t.Errorf("preceding should be empty, got %v", c.Preceding)
	}
	if len(c.Following) != 0 {
		t.Errorf("following should be empty, got %v", c.Following)
	}
	if c.Morphological != "" {
		t.Errorf("morphological should be empty, got %q", c.Morphological)
	}
}

func TestTypesContextSupportsCombiningConjunctiveConstraints(t *testing.T) {
	// "before [+front] AND [+high]" — both constraints must hold
	front := FeatureConstraint{Feature: "front", Value: "+"}
	high := FeatureConstraint{Feature: "high", Value: "+"}
	c := Context{Following: []FeatureConstraint{front, high}}
	if len(c.Following) != 2 {
		t.Errorf("expected 2 following constraints, got %d", len(c.Following))
	}
	if !containsConstraint(c.Following, front) {
		t.Error("front not in following")
	}
	if !containsConstraint(c.Following, high) {
		t.Error("high not in following")
	}
}

func TestTypesContextPositionCanBeSpecified(t *testing.T) {
	c := Context{Position: "initial"}
	if c.Position != "initial" {
		t.Errorf("position = %q, want %q", c.Position, "initial")
	}
}

// ----- FeatureDisplacement -------------------------------------------------

func TestTypesFeatureDisplacementRecordsNamedChange(t *testing.T) {
	d := FeatureDisplacement{Feature: "continuant", FromValue: "-", ToValue: "+"}
	if d.Feature != "continuant" {
		t.Errorf("feature = %q, want %q", d.Feature, "continuant")
	}
	if d.FromValue != "-" {
		t.Errorf("from_value = %q, want %q", d.FromValue, "-")
	}
	if d.ToValue != "+" {
		t.Errorf("to_value = %q, want %q", d.ToValue, "+")
	}
}

func TestTypesFeatureDisplacementsAreHashable(t *testing.T) {
	// Segment is comparable in Go, so FeatureDisplacement (comparable) can be
	// used in maps directly.
	d1 := FeatureDisplacement{Feature: "continuant", FromValue: "-", ToValue: "+"}
	d2 := FeatureDisplacement{Feature: "continuant", FromValue: "-", ToValue: "+"}
	m := map[FeatureDisplacement]bool{}
	m[d1] = true
	m[d2] = true
	if len(m) != 1 {
		t.Errorf("expected 1 unique key, got %d", len(m))
	}
}

// ----- Link ----------------------------------------------------------------

func TestTypesOneToOneLinkHasSingleSegmentChunks(t *testing.T) {
	link := Link{SourceChunk: []Segment{{Grapheme: "p"}}, TargetChunk: []Segment{{Grapheme: "f"}}, Confidence: 1.0}
	if len(link.SourceChunk) != 1 {
		t.Errorf("expected 1 source segment, got %d", len(link.SourceChunk))
	}
	if len(link.TargetChunk) != 1 {
		t.Errorf("expected 1 target segment, got %d", len(link.TargetChunk))
	}
}

func TestTypesLinkDefaultContextIsEmpty(t *testing.T) {
	link := Link{SourceChunk: []Segment{{Grapheme: "p"}}, TargetChunk: []Segment{{Grapheme: "f"}}, Confidence: 1.0}
	if link.Context.ConstraintCount() != 0 {
		t.Errorf("expected empty context, got %v", link.Context)
	}
}

func TestTypesZeroToOneLinkRepresentsInsertion(t *testing.T) {
	link := Link{SourceChunk: nil, TargetChunk: []Segment{{Grapheme: "e"}}, Confidence: 1.0}
	if len(link.SourceChunk) != 0 {
		t.Errorf("expected empty source chunk, got %v", link.SourceChunk)
	}
	if len(link.TargetChunk) != 1 {
		t.Errorf("expected 1 target segment, got %d", len(link.TargetChunk))
	}
}

func TestTypesOneToZeroLinkRepresentsDeletion(t *testing.T) {
	link := Link{SourceChunk: []Segment{{Grapheme: "u"}}, TargetChunk: nil, Confidence: 1.0}
	if len(link.SourceChunk) != 1 {
		t.Errorf("expected 1 source segment, got %d", len(link.SourceChunk))
	}
	if len(link.TargetChunk) != 0 {
		t.Errorf("expected empty target chunk, got %v", link.TargetChunk)
	}
}

func TestTypesManyToManyLinkHoldsMultiSegmentChunks(t *testing.T) {
	// Latin /kt/ ~ Italian /tt/
	link := Link{
		SourceChunk: []Segment{{Grapheme: "k"}, {Grapheme: "t"}},
		TargetChunk: []Segment{{Grapheme: "t"}, {Grapheme: "t"}},
		Confidence:  1.0,
	}
	if len(link.SourceChunk) != 2 {
		t.Errorf("expected 2 source segments, got %d", len(link.SourceChunk))
	}
	if len(link.TargetChunk) != 2 {
		t.Errorf("expected 2 target segments, got %d", len(link.TargetChunk))
	}
}

// ----- Alignment -----------------------------------------------------------

func TestTypesAlignmentHoldsFormsAndLinks(t *testing.T) {
	src := Form{LectID: "latin", Segments: []Segment{{Grapheme: "p"}, {Grapheme: "a"}, {Grapheme: "t"}}}
	tgt := Form{LectID: "germanic", Segments: []Segment{{Grapheme: "f"}, {Grapheme: "a"}, {Grapheme: "d"}}}
	links := []Link{
		{SourceChunk: []Segment{{Grapheme: "p"}}, TargetChunk: []Segment{{Grapheme: "f"}}, Confidence: 1.0},
		{SourceChunk: []Segment{{Grapheme: "a"}}, TargetChunk: []Segment{{Grapheme: "a"}}, Confidence: 1.0},
		{SourceChunk: []Segment{{Grapheme: "t"}}, TargetChunk: []Segment{{Grapheme: "d"}}, Confidence: 1.0},
	}
	a := Alignment{SourceForm: src, TargetForm: tgt, Links: links}
	if a.SourceForm.LectID != src.LectID {
		t.Error("source form not preserved")
	}
	if a.TargetForm.LectID != tgt.LectID {
		t.Error("target form not preserved")
	}
	if len(a.Links) != 3 {
		t.Errorf("expected 3 links, got %d", len(a.Links))
	}
}

// ----- Hashability ---------------------------------------------------------
// All core comparable types can be used as map/set keys.

func TestTypesAllCoreTypesAreHashable(t *testing.T) {
	// COMMITMENT: every core comparable data type can be used as a map key.
	_ = map[Segment]bool{
		{Grapheme: "p"}: true,
		{Grapheme: "a", Tone: "high", Length: "long"}: true,
	}
	_ = map[FeatureConstraint]bool{
		{Feature: "voice", Value: "+"}: true,
	}
	_ = map[FeatureDisplacement]bool{
		{Feature: "continuant", FromValue: "-", ToValue: "+"}: true,
	}
	_ = map[TonalCorrespondence]bool{
		{SrcTone: "H", TgtTone: "L"}: true,
	}
	// Lect is also comparable.
	_ = map[Lect]bool{
		{LectID: "latin", FeatureSystem: "descriptive"}: true,
	}
	// Passed without panic — all types work as map keys.
}

// ----- Suprasegmental data flow -------------------------------------------

func TestTypesSuprasegmentalsSurviveFormConstruction(t *testing.T) {
	// COMMITMENT: suprasegmental annotations are preserved verbatim through Form.
	segT := Segment{Grapheme: "a", Tone: "55", Length: "long", Stress: "primary"}
	form := Form{LectID: "hmong", Segments: []Segment{{Grapheme: "p"}, segT}}
	if form.Segments[1].Tone != "55" {
		t.Errorf("tone = %q, want %q", form.Segments[1].Tone, "55")
	}
	if form.Segments[1].Length != "long" {
		t.Errorf("length = %q, want %q", form.Segments[1].Length, "long")
	}
	if form.Segments[1].Stress != "primary" {
		t.Errorf("stress = %q, want %q", form.Segments[1].Stress, "primary")
	}
}

func TestTypesSuprasegmentalsDistinguishSegmentsForEquality(t *testing.T) {
	// COMMITMENT: tone-bearing minimal pairs are distinct objects.
	aHigh := Segment{Grapheme: "a", Tone: "high"}
	aLow := Segment{Grapheme: "a", Tone: "low"}
	aPlain := Segment{Grapheme: "a"}
	if aHigh == aLow {
		t.Error("a_high should differ from a_low")
	}
	if aHigh == aPlain {
		t.Error("a_high should differ from a_plain")
	}
	if aLow == aPlain {
		t.Error("a_low should differ from a_plain")
	}
}

func TestTypesFormWithNoSyllableBreaksHasEmptySlice(t *testing.T) {
	// COMMITMENT: syllable_breaks defaults to empty (not nil) — actually in
	// Go the zero value is nil; we just verify no panic and treat nil as empty.
	form := Form{LectID: "latin", Segments: []Segment{{Grapheme: "p"}, {Grapheme: "a"}}}
	if len(form.SyllableBreaks) != 0 {
		t.Errorf("expected 0 syllable breaks, got %d", len(form.SyllableBreaks))
	}
}

func TestTypesFormWithMultipleSyllableBreaks(t *testing.T) {
	// COMMITMENT: multi-syllable forms carry per-syllable-start indices.
	// ka.pu.tem (3 syllables)
	form := Form{
		LectID: "latin",
		Segments: []Segment{
			{Grapheme: "k"}, {Grapheme: "a"}, {Grapheme: "p"},
			{Grapheme: "u"}, {Grapheme: "t"}, {Grapheme: "e"}, {Grapheme: "m"},
		},
		SyllableBreaks: []int{0, 2, 4},
	}
	if len(form.SyllableBreaks) != 3 {
		t.Errorf("expected 3 syllable breaks, got %d", len(form.SyllableBreaks))
	}
	if form.SyllableBreaks[0] != 0 || form.SyllableBreaks[1] != 2 || form.SyllableBreaks[2] != 4 {
		t.Errorf("syllable_breaks = %v, want [0 2 4]", form.SyllableBreaks)
	}
}

// ----- Lect with custom feature system ------------------------------------

func TestTypesLectWithCustomFeatureSystem(t *testing.T) {
	// COMMITMENT: a lect can be configured to use any merkmal system.
	lect := NewLect("sanskrit", "distinctive")
	if lect.FeatureSystem != "distinctive" {
		t.Errorf("feature_system = %q, want %q", lect.FeatureSystem, "distinctive")
	}
}

// ----- Context with combined preceding and following constraints ---------

func TestTypesContextCanCombinePrecedingAndFollowingConstraints(t *testing.T) {
	// COMMITMENT: a single context can constrain BOTH neighbours.
	voc := FeatureConstraint{Feature: "vowel", Value: "+"}
	front := FeatureConstraint{Feature: "front", Value: "+"}
	ctx := Context{Preceding: []FeatureConstraint{voc}, Following: []FeatureConstraint{front}}
	if !containsConstraint(ctx.Preceding, voc) {
		t.Error("voc not in preceding")
	}
	if !containsConstraint(ctx.Following, front) {
		t.Error("front not in following")
	}
	if len(ctx.Preceding) != 1 {
		t.Errorf("expected 1 preceding, got %d", len(ctx.Preceding))
	}
	if len(ctx.Following) != 1 {
		t.Errorf("expected 1 following, got %d", len(ctx.Following))
	}
}

// ----- Context subset matching and specificity ----------------------------

func TestTypesEmptyContextIsSubsetOfEverything(t *testing.T) {
	// COMMITMENT: the unconditioned context matches every link.
	empty := Context{}
	specific := Context{
		Position:  "initial",
		Following: []FeatureConstraint{{Feature: "vowel", Value: "+"}},
	}
	if !empty.IsSubsetOf(specific) {
		t.Error("empty context should be subset of specific context")
	}
	if !empty.IsSubsetOf(empty) {
		t.Error("empty context should be subset of itself")
	}
}

func TestTypesContextSubsetRespectsPosition(t *testing.T) {
	initial := Context{Position: "initial"}
	medial := Context{Position: "medial"}
	if initial.IsSubsetOf(medial) {
		t.Error("initial should not be subset of medial")
	}
	if medial.IsSubsetOf(initial) {
		t.Error("medial should not be subset of initial")
	}
	if !initial.IsSubsetOf(Context{Position: "initial"}) {
		t.Error("initial should be subset of itself")
	}
}

func TestTypesContextSubsetRespectsFollowingFeatures(t *testing.T) {
	// COMMITMENT: every feature constraint in self must be in other.
	voc := FeatureConstraint{Feature: "vowel", Value: "+"}
	front := FeatureConstraint{Feature: "front", Value: "+"}
	c1 := Context{Following: []FeatureConstraint{voc}}
	c2 := Context{Following: []FeatureConstraint{voc, front}}
	if !c1.IsSubsetOf(c2) {
		t.Error("c1 (just vowel) should be subset of c2 (vowel+front)")
	}
	if c2.IsSubsetOf(c1) {
		t.Error("c2 (vowel+front) should not be subset of c1 (just vowel)")
	}
}

func TestTypesContextConstraintCountCountsAllSlots(t *testing.T) {
	c := Context{
		Position:  "medial",
		Preceding: []FeatureConstraint{{Feature: "vowel", Value: "+"}},
		Following: []FeatureConstraint{
			{Feature: "vowel", Value: "+"},
			{Feature: "front", Value: "+"},
		},
	}
	if got := c.ConstraintCount(); got != 4 {
		t.Errorf("constraint_count = %d, want 4 (position + 1 preceding + 2 following)", got)
	}
}

func TestTypesEmptyContextHasZeroConstraintCount(t *testing.T) {
	if got := (Context{}).ConstraintCount(); got != 0 {
		t.Errorf("empty context constraint count = %d, want 0", got)
	}
}

// ----- Context long-range fields ------------------------------------------

func typesFC(f, v string) FeatureConstraint { return FeatureConstraint{Feature: f, Value: v} }
func typesFront() FeatureConstraint         { return typesFC("front", "+") }
func typesVoc() FeatureConstraint           { return typesFC("vowel", "+") }

func TestTypesContextPrecedingAtDistanceDefaultsEmpty(t *testing.T) {
	c := Context{}
	if len(c.PrecedingAtDistance) != 0 {
		t.Errorf("PrecedingAtDistance should be empty, got %v", c.PrecedingAtDistance)
	}
	if len(c.FollowingAtDistance) != 0 {
		t.Errorf("FollowingAtDistance should be empty, got %v", c.FollowingAtDistance)
	}
	if len(c.SomewherePreceding) != 0 {
		t.Errorf("SomewherePreceding should be empty, got %v", c.SomewherePreceding)
	}
	if len(c.SomewhereFollowing) != 0 {
		t.Errorf("SomewhereFollowing should be empty, got %v", c.SomewhereFollowing)
	}
	if len(c.SameSyllable) != 0 {
		t.Errorf("SameSyllable should be empty, got %v", c.SameSyllable)
	}
	if len(c.NextSyllable) != 0 {
		t.Errorf("NextSyllable should be empty, got %v", c.NextSyllable)
	}
	if len(c.PreviousSyllable) != 0 {
		t.Errorf("PreviousSyllable should be empty, got %v", c.PreviousSyllable)
	}
}

// (skipped: test_context_long_range_fields_frozen — Python-internal frozen
// dataclass behaviour; no Go analogue.)

func TestTypesIsSubsetPrecedingAtDistance(t *testing.T) {
	c1 := Context{PrecedingAtDistance: []distanceConstraint{{Offset: 2, Constraint: typesFront()}}}
	c2 := Context{PrecedingAtDistance: []distanceConstraint{
		{Offset: 2, Constraint: typesFront()},
		{Offset: 3, Constraint: typesVoc()},
	}}
	if !c1.IsSubsetOf(c2) {
		t.Error("c1 should be subset of c2")
	}
	if c2.IsSubsetOf(c1) {
		t.Error("c2 should not be subset of c1")
	}
	// Different offset is not a match.
	c3 := Context{PrecedingAtDistance: []distanceConstraint{{Offset: 3, Constraint: typesFront()}}}
	if c1.IsSubsetOf(c3) {
		t.Error("c1 (offset 2) should not be subset of c3 (offset 3)")
	}
}

func TestTypesIsSubsetFollowingAtDistance(t *testing.T) {
	c1 := Context{FollowingAtDistance: []distanceConstraint{{Offset: 2, Constraint: typesFront()}}}
	c2 := Context{FollowingAtDistance: []distanceConstraint{{Offset: 2, Constraint: typesFront()}}}
	if !c1.IsSubsetOf(c2) {
		t.Error("c1 should be subset of c2")
	}
}

func TestTypesIsSubsetSomewherePreceding(t *testing.T) {
	c1 := Context{SomewherePreceding: []FeatureConstraint{typesVoc()}}
	c2 := Context{SomewherePreceding: []FeatureConstraint{typesVoc(), typesFront()}}
	if !c1.IsSubsetOf(c2) {
		t.Error("c1 should be subset of c2")
	}
	if c2.IsSubsetOf(c1) {
		t.Error("c2 should not be subset of c1")
	}
}

func TestTypesIsSubsetSomewhereFollowing(t *testing.T) {
	c1 := Context{SomewhereFollowing: []FeatureConstraint{typesFront()}}
	c2 := Context{SomewhereFollowing: []FeatureConstraint{typesFront()}}
	if !c1.IsSubsetOf(c2) {
		t.Error("c1 should be subset of c2")
	}
}

func TestTypesIsSubsetSameSyllable(t *testing.T) {
	c1 := Context{SameSyllable: []FeatureConstraint{typesVoc()}}
	c2 := Context{SameSyllable: []FeatureConstraint{typesVoc()}}
	if !c1.IsSubsetOf(c2) {
		t.Error("c1 should be subset of c2")
	}
	cFront := Context{SameSyllable: []FeatureConstraint{typesFront()}}
	if cFront.IsSubsetOf(c2) {
		t.Error("front should not be subset of voc")
	}
}

func TestTypesIsSubsetNextSyllable(t *testing.T) {
	c1 := Context{NextSyllable: []FeatureConstraint{typesFront()}}
	c2 := Context{NextSyllable: []FeatureConstraint{typesFront(), typesVoc()}}
	if !c1.IsSubsetOf(c2) {
		t.Error("c1 should be subset of c2")
	}
	if c2.IsSubsetOf(c1) {
		t.Error("c2 should not be subset of c1")
	}
}

func TestTypesIsSubsetPreviousSyllable(t *testing.T) {
	c1 := Context{PreviousSyllable: []FeatureConstraint{typesVoc()}}
	if !c1.IsSubsetOf(Context{PreviousSyllable: []FeatureConstraint{typesVoc()}}) {
		t.Error("c1 should be subset of identical context")
	}
	if c1.IsSubsetOf(Context{PreviousSyllable: []FeatureConstraint{}}) {
		t.Error("c1 should not be subset of empty previous syllable")
	}
}

func TestTypesEmptyContextSubsetOfLongRangeContext(t *testing.T) {
	// Backward compatibility: empty context still matches anything.
	specific := Context{
		NextSyllable:        []FeatureConstraint{typesFront()},
		SomewherePreceding:  []FeatureConstraint{typesVoc()},
		PrecedingAtDistance: []distanceConstraint{{Offset: 2, Constraint: typesFront()}},
	}
	empty := Context{}
	if !empty.IsSubsetOf(specific) {
		t.Error("empty context should be subset of any context")
	}
}

func TestTypesConstraintCountIncludesLongRangeSlots(t *testing.T) {
	c := Context{
		Position:            "medial",
		Preceding:           []FeatureConstraint{typesVoc()},
		PrecedingAtDistance: []distanceConstraint{{Offset: 2, Constraint: typesFront()}},
		FollowingAtDistance: []distanceConstraint{
			{Offset: 2, Constraint: typesVoc()},
			{Offset: 3, Constraint: typesFront()},
		},
		SomewherePreceding: []FeatureConstraint{typesVoc()},
		SomewhereFollowing: []FeatureConstraint{typesFront()},
		SameSyllable:       []FeatureConstraint{typesVoc()},
		NextSyllable:       []FeatureConstraint{typesFront()},
		PreviousSyllable:   []FeatureConstraint{typesVoc()},
	}
	// position(1) + preceding(1) + pre@d(1) + fol@d(2) + s_pre(1)
	// + s_fol(1) + same(1) + next(1) + prev(1) = 10
	if got := c.ConstraintCount(); got != 10 {
		t.Errorf("constraint_count = %d, want 10", got)
	}
}

func TestTypesLongRangeIndependentOfImmediateNeighbours(t *testing.T) {
	// Constraints on different slots don't interfere with each other.
	tableCtx := Context{Preceding: []FeatureConstraint{typesVoc()}}
	link := Context{
		Preceding:    []FeatureConstraint{typesVoc()},
		NextSyllable: []FeatureConstraint{typesFront()},
	}
	if !tableCtx.IsSubsetOf(link) {
		t.Error("table context (just preceding) should be subset of link (preceding+next)")
	}
	// A long-range-only table entry matches a long-range-carrying link.
	table2 := Context{NextSyllable: []FeatureConstraint{typesFront()}}
	if !table2.IsSubsetOf(link) {
		t.Error("table2 (next syllable only) should be subset of link")
	}
}
