package regulae

// Mirror of python/tests/test_format_model.py
//
// Skipped tests (no Go DescribeSource equivalent):
//   test_describe_source_on_untrained_grapheme_returns_empty_sections
//   test_describe_source_shows_unconditioned_entries
//   test_describe_source_groups_conditioned_entries_by_context
//   test_describe_source_shows_relevant_chunks
//   test_describe_source_on_trained_model
//
// The skipped tests are marked with inline comments below.
//
// Already covered by existing Go suite:
//   TestFormatModelContainsCorrespondence — format_test.go

import (
	"strings"
	"testing"
)

// scrMakeModel is a local helper that builds a LearnedModel from a
// SegmentCorrespondenceTable. Prefixed "scr" per naming contract.
func scrMakeModel(tbl SegmentCorrespondenceTable) *LearnedModel {
	m := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	m.SegmentTable = tbl
	return m
}

// ----- format_model empty -------------------------------------------------

func TestFormatModelFormatModelEmptyShowsNoDataMarkers(t *testing.T) {
	// COMMITMENT: an empty model renders with explicit 'no data' markers.
	out := FormatModel(EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration), DefaultFormatModelOptions())
	lower := strings.ToLower(out)
	if !strings.Contains(lower, "no observations") {
		t.Errorf("empty model output missing 'no observations':\n%s", out)
	}
	if !strings.Contains(lower, "none") {
		t.Errorf("empty model output missing 'none' (for displacement/chunk sections):\n%s", out)
	}
}

func TestFormatModelFormatModelShowsFeatureSystem(t *testing.T) {
	out := FormatModel(EmptyLearnedModel("distinctive", defaultTemperature, defaultConcentration), DefaultFormatModelOptions())
	if !strings.Contains(out, "distinctive") {
		t.Errorf("output %q missing 'distinctive'", out)
	}
}

// ----- format_model segment correspondences ------------------------------

func TestFormatModelFormatModelShowsTopSegmentCorrespondences(t *testing.T) {
	tbl := scrBuildTable(
		[]scrEntry{
			{scrMakeCC("p", "f"), 10.0},
			{scrMakeCC("t", "d"), 5.0},
			{scrMakeCC("k", "h"), 3.0},
		},
		[]scrEntry{},
		map[string]float64{"p": 10.0, "t": 5.0, "k": 3.0},
	)
	model := scrMakeModel(tbl)
	out := FormatModel(model, DefaultFormatModelOptions())
	if !strings.Contains(out, "p -> f") {
		t.Errorf("output missing 'p -> f':\n%s", out)
	}
	if !strings.Contains(out, "t -> d") {
		t.Errorf("output missing 't -> d':\n%s", out)
	}
	if !strings.Contains(out, "k -> h") {
		t.Errorf("output missing 'k -> h':\n%s", out)
	}
	if !strings.Contains(out, "10") {
		t.Errorf("output missing count '10':\n%s", out)
	}
	if !strings.Contains(out, "5") {
		t.Errorf("output missing count '5':\n%s", out)
	}
	if !strings.Contains(out, "3") {
		t.Errorf("output missing count '3':\n%s", out)
	}
}

func TestFormatModelFormatModelRespectsTopSegmentsCap(t *testing.T) {
	// COMMITMENT: top_segments caps the number of correspondences shown.
	// Build 10 identities a->a … j->j with counts 1…10.
	counts := []scrEntry{}
	srcTotals := map[string]float64{}
	letters := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
	for i, c := range letters {
		counts = append(counts, scrEntry{scrMakeCC(c, c), float64(i + 1)})
		srcTotals[c] = float64(i + 1)
	}
	tbl := scrBuildTable(counts, []scrEntry{}, srcTotals)
	model := scrMakeModel(tbl)

	opts := DefaultFormatModelOptions()
	opts.TopSegments = 3
	out := FormatModel(model, opts)

	// Top 3 by count should be j (10), i (9), h (8).
	if !strings.Contains(out, "j -> j") {
		t.Errorf("output missing 'j -> j' (highest count):\n%s", out)
	}
	if !strings.Contains(out, "i -> i") {
		t.Errorf("output missing 'i -> i' (second):\n%s", out)
	}
	if !strings.Contains(out, "h -> h") {
		t.Errorf("output missing 'h -> h' (third):\n%s", out)
	}
	// Lowest count should be omitted.
	if strings.Contains(out, "a -> a") {
		t.Errorf("output unexpectedly contains 'a -> a' (should be omitted at top=3):\n%s", out)
	}
	// Total count indicator.
	if !strings.Contains(out, "of 10") {
		t.Errorf("output missing 'of 10':\n%s", out)
	}
}

// ----- format_model displacement -----------------------------------------

func TestFormatModelFormatModelShowsIdentityDisplacementAsSuch(t *testing.T) {
	// COMMITMENT: empty-tuple displacement renders as 'identity'.
	m := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	m.DisplacementDist.setCount([]FeatureDisplacement{}, 5.0)
	m.DisplacementDist.Total = 5.0
	out := FormatModel(m, DefaultFormatModelOptions())
	if !strings.Contains(out, "identity") {
		t.Errorf("output %q missing 'identity' for empty displacement", out)
	}
}

func TestFormatModelFormatModelShowsDisplacementFeaturesCompactly(t *testing.T) {
	// COMMITMENT: non-identity displacements render feature names and '->'.
	disp := []FeatureDisplacement{
		{Feature: "stop", FromValue: "present", ToValue: "absent"},
		{Feature: "fricative", FromValue: "absent", ToValue: "present"},
	}
	m := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	m.DisplacementDist.setCount(disp, 7.0)
	m.DisplacementDist.Total = 7.0
	out := FormatModel(m, DefaultFormatModelOptions())
	if !strings.Contains(out, "stop") {
		t.Errorf("output %q missing 'stop'", out)
	}
	if !strings.Contains(out, "fricative") {
		t.Errorf("output %q missing 'fricative'", out)
	}
	if !strings.Contains(out, "->") {
		t.Errorf("output %q missing '->'", out)
	}
}

// ----- format_model chunk table ------------------------------------------

func TestFormatModelFormatModelShowsPromotedChunks(t *testing.T) {
	kt := []Segment{seg("k"), seg("t")}
	tt := []Segment{seg("t"), seg("t")}
	m := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	m.ChunkTable.setEntry(kt, tt, -1.234)
	out := FormatModel(m, DefaultFormatModelOptions())
	if !strings.Contains(out, "kt") {
		t.Errorf("output %q missing 'kt'", out)
	}
	if !strings.Contains(out, "tt") {
		t.Errorf("output %q missing 'tt'", out)
	}
	if !strings.Contains(out, "1.234") {
		t.Errorf("output %q missing '1.234'", out)
	}
}

// ----- format_model integration with trained corpus ----------------------

func TestFormatModelFormatModelIntegratesWithTrainedCorpus(t *testing.T) {
	// SMOKE: FormatModel runs cleanly on a real trained model.
	corpus := []FormPair{
		pairOf("A", "pata", "B", "fada"),
		pairOf("A", "pata", "B", "fada"),
	}
	trained := mustTrain(t, corpus)
	out := FormatModel(trained, DefaultFormatModelOptions())
	if !strings.Contains(out, "LearnedModel") {
		t.Errorf("output missing 'LearnedModel' header:\n%s", out)
	}
	if !strings.Contains(out, "Segment correspondences") {
		t.Errorf("output missing 'Segment correspondences' section:\n%s", out)
	}
	lowerOut := strings.ToLower(out)
	if !strings.Contains(lowerOut, "displacements") {
		t.Errorf("output missing 'displacements' section:\n%s", out)
	}
	if !strings.Contains(lowerOut, "chunks") {
		t.Errorf("output missing 'chunks' section:\n%s", out)
	}
}

// ----- describe_source tests (skipped: no Go DescribeSource) -------------
//
// test_describe_source_on_untrained_grapheme_returns_empty_sections
// test_describe_source_shows_unconditioned_entries
// test_describe_source_groups_conditioned_entries_by_context
// test_describe_source_shows_relevant_chunks
// test_describe_source_on_trained_model
//
// (skipped: no Go DescribeSource)
