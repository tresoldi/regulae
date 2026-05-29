package regulae

// Mirror of python/tests/test_cross_dim_data_types.py

import (
	"testing"
)

// ----- CrossDimensionalLink construction --------------------------------

func TestCrossDimDataTypesLinkMinimalConstruction(t *testing.T) {
	// COMMITMENT: a CrossDimensionalLink can be built with the six required
	// fields; count, src_count default to 0.0 and confidence defaults to 1.0.
	link := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
	}
	if link.SrcFeature.Feature != "voiced" {
		t.Errorf("SrcFeature.Feature = %q, want %q", link.SrcFeature.Feature, "voiced")
	}
	if link.SrcFeature.Value != "+" {
		t.Errorf("SrcFeature.Value = %q, want %q", link.SrcFeature.Value, "+")
	}
	if link.SrcPosition != "relative_-1" {
		t.Errorf("SrcPosition = %q, want %q", link.SrcPosition, "relative_-1")
	}
	if link.TgtDimension != "tone" {
		t.Errorf("TgtDimension = %q, want %q", link.TgtDimension, "tone")
	}
	if link.TgtValue != "4" {
		t.Errorf("TgtValue = %q, want %q", link.TgtValue, "4")
	}
	if link.TgtPositionOffset != 0 {
		t.Errorf("TgtPositionOffset = %d, want 0", link.TgtPositionOffset)
	}
	// Default values.
	if link.Count != 0.0 {
		t.Errorf("Count = %v, want 0.0", link.Count)
	}
	if link.SrcCount != 0.0 {
		t.Errorf("SrcCount = %v, want 0.0", link.SrcCount)
	}
	// In Go there is no Python dataclass default_factory; Confidence defaults
	// to 0.0 (zero value). The Python default is 1.0. We verify the Go
	// struct carries whatever was explicitly set; for an empty construction
	// Confidence is 0.0. The relevant COMMITMENT is that fields are accessible.
	// (skipped: exact Python default=1.0 — Go zero value differs)
}

func TestCrossDimDataTypesLinkFullConstruction(t *testing.T) {
	// COMMITMENT: count, src_count, and confidence are settable and carry
	// through unchanged.
	link := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "nasal", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "5",
		TgtPositionOffset: 0,
		Count:             12.0,
		SrcCount:          15.0,
		Confidence:        0.8,
	}
	if link.Count != 12.0 {
		t.Errorf("Count = %v, want 12.0", link.Count)
	}
	if link.SrcCount != 15.0 {
		t.Errorf("SrcCount = %v, want 15.0", link.SrcCount)
	}
	if link.Confidence != 0.8 {
		t.Errorf("Confidence = %v, want 0.8", link.Confidence)
	}
}

func TestCrossDimDataTypesLinkIsMutable(t *testing.T) {
	// Go structs are value types and are not frozen. The Python test checks
	// immutability of a frozen dataclass — the Go equivalent is that mutation
	// on a copy does not affect the original.
	link := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             5.0,
	}
	copy := link
	copy.Count = 99.0
	if link.Count != 5.0 {
		t.Errorf("original Count was mutated to %v", link.Count)
	}
}

func TestCrossDimDataTypesLinkIsUsableAsMapValue(t *testing.T) {
	// COMMITMENT: CrossDimensionalLink is usable as a map value.
	link := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
	}
	m := map[string]CrossDimensionalLink{"key": link}
	if m["key"].SrcFeature.Feature != "voiced" {
		t.Errorf("map lookup failed")
	}
}

func TestCrossDimDataTypesLinkEqualityIsStructural(t *testing.T) {
	// COMMITMENT: two links with identical required fields are equal;
	// links with different fields are not.
	a := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
	}
	b := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
	}
	// Compare the deterministic fields (excluding pointer Uncertainty).
	equalCore := a.SrcFeature == b.SrcFeature &&
		a.SrcPosition == b.SrcPosition &&
		a.TgtDimension == b.TgtDimension &&
		a.TgtValue == b.TgtValue &&
		a.TgtPositionOffset == b.TgtPositionOffset
	if !equalCore {
		t.Error("expected a == b (same fields)")
	}
	c := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "5", // different
		TgtPositionOffset: 0,
	}
	if a.TgtValue == c.TgtValue {
		t.Error("expected a != c (different tgt_value)")
	}
}

// ----- CrossDimensionalLinkTable construction ---------------------------

func TestCrossDimDataTypesTableDefaultsToEmpty(t *testing.T) {
	table := CrossDimensionalLinkTable{}
	if len(table.Entries) != 0 {
		t.Errorf("expected empty entries, got %d", len(table.Entries))
	}
}

func TestCrossDimDataTypesTableHoldsEntries(t *testing.T) {
	link1 := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
	}
	link2 := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "nasal", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "5",
		TgtPositionOffset: 0,
	}
	table := CrossDimensionalLinkTable{Entries: []CrossDimensionalLink{link1, link2}}
	if len(table.Entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(table.Entries))
	}
	if table.Entries[0].SrcFeature.Feature != "voiced" {
		t.Errorf("entries[0].SrcFeature.Feature = %q, want %q", table.Entries[0].SrcFeature.Feature, "voiced")
	}
	if table.Entries[1].SrcFeature.Feature != "nasal" {
		t.Errorf("entries[1].SrcFeature.Feature = %q, want %q", table.Entries[1].SrcFeature.Feature, "nasal")
	}
}

func TestCrossDimDataTypesTableCanBeReassigned(t *testing.T) {
	// Go structs are mutable by default; verify new entries can be set.
	table := CrossDimensionalLinkTable{}
	table.Entries = []CrossDimensionalLink{
		{SrcFeature: FeatureConstraint{Feature: "voiced", Value: "+"}, SrcPosition: "relative_0",
			TgtDimension: "tone", TgtValue: "4"},
	}
	if len(table.Entries) != 1 {
		t.Errorf("expected 1 entry after assignment, got %d", len(table.Entries))
	}
}

// ----- LearnedModel integration -----------------------------------------

func TestCrossDimDataTypesLearnedModelHasEmptyTableByDefault(t *testing.T) {
	// COMMITMENT: a default-constructed LearnedModel exposes an empty
	// cross_dimensional_table.
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	if len(model.CrossDimensionalTable.Entries) != 0 {
		t.Errorf("expected empty CrossDimensionalTable, got %d entries",
			len(model.CrossDimensionalTable.Entries))
	}
}

func TestCrossDimDataTypesLearnedModelEmptyFactoryHasEmptyTable(t *testing.T) {
	model := EmptyLearnedModel("", defaultTemperature, defaultConcentration)
	if len(model.CrossDimensionalTable.Entries) != 0 {
		t.Errorf("expected empty CrossDimensionalTable, got %d entries",
			len(model.CrossDimensionalTable.Entries))
	}
}

func TestCrossDimDataTypesLearnedModelAcceptsPopulatedTable(t *testing.T) {
	link := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             18.0,
		SrcCount:          20.0,
		Confidence:        0.9,
	}
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	model.CrossDimensionalTable = CrossDimensionalLinkTable{Entries: []CrossDimensionalLink{link}}
	if len(model.CrossDimensionalTable.Entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(model.CrossDimensionalTable.Entries))
	}
	e := model.CrossDimensionalTable.Entries[0]
	if e.SrcFeature.Feature != "voiced" {
		t.Errorf("entry.SrcFeature.Feature = %q, want %q", e.SrcFeature.Feature, "voiced")
	}
	if e.Count != 18.0 {
		t.Errorf("entry.Count = %v, want 18.0", e.Count)
	}
	if e.Confidence != 0.9 {
		t.Errorf("entry.Confidence = %v, want 0.9", e.Confidence)
	}
}

// ----- src_position canonical form --------------------------------------

func TestCrossDimDataTypesLinkUsesRelativePositionStrings(t *testing.T) {
	// COMMITMENT: v1 canonical src_position values are exactly
	// "relative_-1", "relative_0", and "relative_+1".
	for _, pos := range []string{"relative_-1", "relative_0", "relative_+1"} {
		link := CrossDimensionalLink{
			SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
			SrcPosition:       pos,
			TgtDimension:      "tone",
			TgtValue:          "4",
			TgtPositionOffset: 0,
		}
		if link.SrcPosition != pos {
			t.Errorf("SrcPosition = %q, want %q", link.SrcPosition, pos)
		}
	}
}

// ----- backward compatibility (empty table is inert) --------------------

func TestCrossDimDataTypesDefaultModelRoundTripsThroughEquality(t *testing.T) {
	// COMMITMENT: two default-constructed models have equal empty
	// cross_dimensional_tables — the new field doesn't break structural
	// equality.
	a := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	b := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	if len(a.CrossDimensionalTable.Entries) != len(b.CrossDimensionalTable.Entries) {
		t.Errorf("tables differ: len %d vs %d",
			len(a.CrossDimensionalTable.Entries),
			len(b.CrossDimensionalTable.Entries))
	}
	if len(a.CrossDimensionalTable.Entries) != 0 {
		t.Errorf("expected empty entries, got %d", len(a.CrossDimensionalTable.Entries))
	}
}

// ----- scoring overlay stub ---------------------------------------------

func TestCrossDimDataTypesApplyCrossDimAdjustmentsReturnsZeroForEmptyTable(t *testing.T) {
	// COMMITMENT: with an empty cross_dimensional_table, the overlay returns
	// 0.0 regardless of the alignment. This is the empty-table inertness
	// invariant that protects every existing test.
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	src := Form{LectID: "A", Segments: []Segment{
		{Grapheme: "p"}, {Grapheme: "a"}, {Grapheme: "t"}, {Grapheme: "a"},
	}}
	tgt := Form{LectID: "B", Segments: []Segment{
		{Grapheme: "f"}, {Grapheme: "a"}, {Grapheme: "d"}, {Grapheme: "a"},
	}}
	alignment := alignForms(src, tgt, model.FeatureSystem, DefaultMaxChunkSize, model)
	adj := applyCrossDimensionalAdjustments(alignment, model)
	if adj != 0.0 {
		t.Errorf("expected 0.0 adjustment on empty table, got %v", adj)
	}
}

func TestCrossDimDataTypesAlignmentCostUnchangedWhenTableIsEmpty(t *testing.T) {
	// COMMITMENT: alignment_cost with an empty cross-dimensional overlay is
	// deterministic under repeated calls.
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	src := Form{LectID: "A", Segments: []Segment{{Grapheme: "p"}, {Grapheme: "a"}}}
	tgt := Form{LectID: "B", Segments: []Segment{{Grapheme: "f"}, {Grapheme: "a"}}}
	alignment := alignForms(src, tgt, model.FeatureSystem, DefaultMaxChunkSize, model)
	cost1 := alignmentCost(alignment, model.FeatureSystem, model)
	cost2 := alignmentCost(alignment, model.FeatureSystem, model)
	if cost1 != cost2 {
		t.Errorf("non-deterministic alignment cost: %v vs %v", cost1, cost2)
	}
}

func TestCrossDimDataTypesOverlayNegativeAdjustmentWhenRuleConfirmed(t *testing.T) {
	// COMMITMENT: when a committed rule's source context holds AND the target
	// value matches, the overlay contributes a NEGATIVE cost adjustment.
	tonal := TonalCorrespondenceTable{
		Counts: map[TonalCorrespondence]float64{
			{SrcTone: "", TgtTone: "4"}: 5.0,
			{SrcTone: "", TgtTone: "1"}: 5.0,
		},
		PriorPseudoCounts: map[TonalCorrespondence]float64{},
		SrcTotals:         map[string]float64{"": 10.0},
		Uncertainty:       map[TonalCorrespondence]UncertaintyEstimate{},
	}
	fc := FeatureConstraint{Feature: "voiced", Value: "+"}
	link := CrossDimensionalLink{
		SrcFeature:        fc,
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             10.0,
		SrcCount:          10.0,
		Confidence:        1.0,
	}
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	model.TonalTable = tonal
	model.CrossDimensionalTable = CrossDimensionalLinkTable{Entries: []CrossDimensionalLink{link}}

	// Voiced initial + target tone 4: rule fires, adjustment is negative.
	src := Form{LectID: "A", Segments: []Segment{{Grapheme: "b"}, {Grapheme: "a"}}}
	tgt := Form{LectID: "B", Segments: []Segment{{Grapheme: "b"}, {Grapheme: "a", Tone: "4"}}}
	alignment := alignForms(src, tgt, model.FeatureSystem, DefaultMaxChunkSize, model)
	adj := applyCrossDimensionalAdjustments(alignment, model)
	if adj >= 0 {
		t.Errorf("expected negative adjustment when rule confirmed, got %v", adj)
	}
	if adj >= -0.3 {
		t.Errorf("expected adjustment < -0.3, got %v", adj)
	}
}

func TestCrossDimDataTypesOverlayPositiveAdjustmentWhenRuleContradicted(t *testing.T) {
	// COMMITMENT: when a rule's source context holds but the target value does
	// NOT match, the overlay contributes a POSITIVE adjustment.
	tonal := TonalCorrespondenceTable{
		Counts: map[TonalCorrespondence]float64{
			{SrcTone: "", TgtTone: "4"}: 5.0,
			{SrcTone: "", TgtTone: "1"}: 5.0,
		},
		PriorPseudoCounts: map[TonalCorrespondence]float64{},
		SrcTotals:         map[string]float64{"": 10.0},
		Uncertainty:       map[TonalCorrespondence]UncertaintyEstimate{},
	}
	fc := FeatureConstraint{Feature: "voiced", Value: "+"}
	link := CrossDimensionalLink{
		SrcFeature:        fc,
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             10.0,
		SrcCount:          10.0,
		Confidence:        1.0,
	}
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	model.TonalTable = tonal
	model.CrossDimensionalTable = CrossDimensionalLinkTable{Entries: []CrossDimensionalLink{link}}

	// Voiced initial + target tone 1 (NOT the rule's prediction).
	src := Form{LectID: "A", Segments: []Segment{{Grapheme: "b"}, {Grapheme: "a"}}}
	tgt := Form{LectID: "B", Segments: []Segment{{Grapheme: "b"}, {Grapheme: "a", Tone: "1"}}}
	alignment := alignForms(src, tgt, model.FeatureSystem, DefaultMaxChunkSize, model)
	adj := applyCrossDimensionalAdjustments(alignment, model)
	if adj <= 0 {
		t.Errorf("expected positive adjustment when rule contradicted, got %v", adj)
	}
}

func TestCrossDimDataTypesOverlayZeroWhenContextDoesNotHold(t *testing.T) {
	// COMMITMENT: when the source context doesn't hold, the rule contributes
	// zero adjustment.
	tonal := TonalCorrespondenceTable{
		Counts: map[TonalCorrespondence]float64{
			{SrcTone: "", TgtTone: "4"}: 5.0,
			{SrcTone: "", TgtTone: "1"}: 5.0,
		},
		PriorPseudoCounts: map[TonalCorrespondence]float64{},
		SrcTotals:         map[string]float64{"": 10.0},
		Uncertainty:       map[TonalCorrespondence]UncertaintyEstimate{},
	}
	fc := FeatureConstraint{Feature: "voiced", Value: "+"}
	link := CrossDimensionalLink{
		SrcFeature:        fc,
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             10.0,
		SrcCount:          10.0,
		Confidence:        1.0,
	}
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	model.TonalTable = tonal
	model.CrossDimensionalTable = CrossDimensionalLinkTable{Entries: []CrossDimensionalLink{link}}

	// Voiceless initial: rule's source context does not hold.
	src := Form{LectID: "A", Segments: []Segment{{Grapheme: "p"}, {Grapheme: "a"}}}
	tgt := Form{LectID: "B", Segments: []Segment{{Grapheme: "p"}, {Grapheme: "a", Tone: "4"}}}
	alignment := alignForms(src, tgt, model.FeatureSystem, DefaultMaxChunkSize, model)
	adj := applyCrossDimensionalAdjustments(alignment, model)
	if adj != 0.0 {
		t.Errorf("expected 0.0 when context does not hold, got %v", adj)
	}
}

func TestCrossDimDataTypesOverlayZeroWhenRuleDoesNotDiscriminate(t *testing.T) {
	// COMMITMENT: when the rule's P_cond equals the base rate P_base, the
	// adjustment is zero even when fired (log-ratio ≈ 0).
	tonal := TonalCorrespondenceTable{
		Counts: map[TonalCorrespondence]float64{
			{SrcTone: "", TgtTone: "4"}: 5.0,
			{SrcTone: "", TgtTone: "1"}: 5.0,
		},
		PriorPseudoCounts: map[TonalCorrespondence]float64{},
		SrcTotals:         map[string]float64{"": 10.0},
		Uncertainty:       map[TonalCorrespondence]UncertaintyEstimate{},
	}
	fc := FeatureConstraint{Feature: "voiced", Value: "+"}
	// count/src_count → same conditional as the base rate (0.5) under
	// Dirichlet smoothing with V=2: (5+1)/(10+2) = 0.5
	link := CrossDimensionalLink{
		SrcFeature:        fc,
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             5.0,
		SrcCount:          10.0,
		Confidence:        0.5,
	}
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	model.TonalTable = tonal
	model.CrossDimensionalTable = CrossDimensionalLinkTable{Entries: []CrossDimensionalLink{link}}

	src := Form{LectID: "A", Segments: []Segment{{Grapheme: "b"}, {Grapheme: "a"}}}
	tgt := Form{LectID: "B", Segments: []Segment{{Grapheme: "b"}, {Grapheme: "a", Tone: "4"}}}
	alignment := alignForms(src, tgt, model.FeatureSystem, DefaultMaxChunkSize, model)
	adj := applyCrossDimensionalAdjustments(alignment, model)
	if adj > 1e-9 || adj < -1e-9 {
		t.Errorf("expected ~0.0 when P_cond == P_base, got %v", adj)
	}
}

func TestCrossDimDataTypesOverlayZeroForEmptyTable(t *testing.T) {
	// COMMITMENT: an empty cross_dimensional_table always contributes zero,
	// regardless of the alignment or tonal table.
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	src := Form{LectID: "A", Segments: []Segment{{Grapheme: "b"}, {Grapheme: "a"}}}
	tgt := Form{LectID: "B", Segments: []Segment{{Grapheme: "b"}, {Grapheme: "a", Tone: "4"}}}
	alignment := alignForms(src, tgt, model.FeatureSystem, DefaultMaxChunkSize, model)
	adj := applyCrossDimensionalAdjustments(alignment, model)
	if adj != 0.0 {
		t.Errorf("expected 0.0 for empty table, got %v", adj)
	}
}
