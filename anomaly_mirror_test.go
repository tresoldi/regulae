package regulae

// Mirror of python/tests/test_anomaly.py

import (
	"testing"
)

// xanVoicedCorpus returns the 20-pair corpus with a clean tonogenesis signal:
// voiced initial → tone "4" on vowel, voiceless → tone "1".
func xanVoicedCorpus() []FormPair {
	plainPairs := [][2]string{
		// Voiced initial → tone 4 on vowel.
		{"ba", "ba"}, {"da", "da"}, {"ga", "ga"}, {"ma", "ma"}, {"na", "na"},
		{"bi", "bi"}, {"di", "di"}, {"gu", "gu"}, {"bo", "bo"}, {"ma", "ma"},
		// Voiceless initial → tone 1 on vowel.
		{"pa", "pa"}, {"ta", "ta"}, {"ka", "ka"}, {"sa", "sa"}, {"fa", "fa"},
		{"pi", "pi"}, {"ti", "ti"}, {"ku", "ku"}, {"po", "po"}, {"ta", "ta"},
	}
	voiced := map[byte]bool{'b': true, 'd': true, 'g': true, 'm': true, 'n': true}
	var pairs []FormPair
	for _, p := range plainPairs {
		srcWord, tgtWord := p[0], p[1]
		srcSegs := make([]Segment, len(srcWord))
		for i := range srcWord {
			srcSegs[i] = Segment{Grapheme: string(srcWord[i])}
		}
		src := Form{LectID: "A", Segments: srcSegs}
		initial := srcWord[0]
		tone := "1"
		if voiced[initial] {
			tone = "4"
		}
		tgtSegs := make([]Segment, len(tgtWord))
		for i := range tgtWord {
			if i == len(tgtWord)-1 {
				tgtSegs[i] = Segment{Grapheme: string(tgtWord[i]), Tone: tone}
			} else {
				tgtSegs[i] = Segment{Grapheme: string(tgtWord[i])}
			}
		}
		tgt := Form{LectID: "B", Segments: tgtSegs}
		pairs = append(pairs, FormPair{Src: src, Tgt: tgt})
	}
	return pairs
}

// xanIndependentCorpus returns a corpus with no cross-dimensional signal —
// target tones are assigned arbitrarily, unrelated to source consonant.
func xanIndependentCorpus() []FormPair {
	rows := []struct {
		src, tgt, tone string
	}{
		{"ba", "ba", "1"}, {"da", "da", "1"}, {"ga", "ga", "2"},
		{"ma", "ma", "2"}, {"pa", "pa", "1"}, {"ta", "ta", "2"},
		{"ka", "ka", "1"}, {"sa", "sa", "2"}, {"bi", "bi", "2"},
		{"pi", "pi", "1"},
	}
	var out []FormPair
	for _, row := range rows {
		srcSegs := make([]Segment, len(row.src))
		for i := range row.src {
			srcSegs[i] = Segment{Grapheme: string(row.src[i])}
		}
		tgtSegs := make([]Segment, len(row.tgt))
		for i := range row.tgt {
			if i == len(row.tgt)-1 {
				tgtSegs[i] = Segment{Grapheme: string(row.tgt[i]), Tone: row.tone}
			} else {
				tgtSegs[i] = Segment{Grapheme: string(row.tgt[i])}
			}
		}
		out = append(out, FormPair{
			Src: Form{LectID: "A", Segments: srcSegs},
			Tgt: Form{LectID: "B", Segments: tgtSegs},
		})
	}
	return out
}

// xanTrainPairwise is a test helper that trains pairwise on a corpus.
func xanTrainPairwise(t *testing.T, corpus []FormPair) *LearnedModel {
	t.Helper()
	m, err := TrainPairwise(corpus, nil, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainPairwise: %v", err)
	}
	return m
}

// ----- headline: tonogenesis detection ----------------------------------

func TestAnomalyFindResidualPatternsSurfacesTonogenesisAsTopHypothesis(t *testing.T) {
	// COMMITMENT: on a corpus where source voicing cleanly determines target
	// tone, the top hypothesis must be a cross-dimensional tonogenesis rule
	// that predicts the observed tone from a voicing-related feature.
	corpus := xanVoicedCorpus()
	model := xanTrainPairwise(t, corpus)
	hypotheses := findResidualPatterns(corpus, model, 5, 100, 0.95, "", 0, nil)
	if len(hypotheses) == 0 {
		t.Fatal("expected at least one hypothesis, got none")
	}
	top := hypotheses[0]
	if top.Kind != "cross_dimensional_tonogenesis" {
		t.Errorf("top hypothesis kind = %q, want %q", top.Kind, "cross_dimensional_tonogenesis")
	}
	voicingFeatures := map[string]bool{
		"voiced": true, "voiceless": true, "sonorant": true,
		"nasal": true, "stop": true, "fricative": true, "consonant": true,
	}
	if !voicingFeatures[top.SrcFeature] {
		t.Errorf("top hypothesis SrcFeature = %q, want a voicing-related feature", top.SrcFeature)
	}
	if top.NullPercentile < 0.95 {
		t.Errorf("top hypothesis NullPercentile = %v, want >= 0.95", top.NullPercentile)
	}
	if top.ResidualMI <= 0.1 {
		t.Errorf("top hypothesis ResidualMI = %v, want > 0.1", top.ResidualMI)
	}
}

func TestAnomalyFindResidualPatternsProducesOnlyTonalHypothesesForV1(t *testing.T) {
	// COMMITMENT: v1 only implements the cross-dimensional tonogenesis kind.
	corpus := xanVoicedCorpus()
	model := xanTrainPairwise(t, corpus)
	hypotheses := findResidualPatterns(corpus, model, 5, 100, 0.95, "", 0, nil)
	for _, h := range hypotheses {
		if h.Kind != "cross_dimensional_tonogenesis" {
			t.Errorf("unexpected hypothesis kind %q", h.Kind)
		}
	}
}

// ----- no signal baseline -----------------------------------------------

func TestAnomalyFindResidualPatternsReturnsNoHypothesesOnIndependentData(t *testing.T) {
	// COMMITMENT: when source and target features are independent by
	// construction, no hypothesis should exceed the permutation null at
	// the 0.95 threshold (or at most only with very low MI).
	corpus := xanIndependentCorpus()
	model := xanTrainPairwise(t, corpus)
	hypotheses := findResidualPatterns(corpus, model, 5, 100, 0.95, "", 0, nil)
	if len(hypotheses) > 0 {
		// Allow small leakage on tiny corpora; reject high-MI hypotheses.
		maxMI := 0.0
		for _, h := range hypotheses {
			if h.ResidualMI > maxMI {
				maxMI = h.ResidualMI
			}
		}
		if maxMI >= 0.5 {
			t.Errorf("independent corpus has spurious hypotheses with max MI %v >= 0.5", maxMI)
		}
	}
}

func TestAnomalyFindResidualPatternsOnNoTonesReturnsEmpty(t *testing.T) {
	// COMMITMENT: a corpus with no tonal data produces no cross-dimensional
	// tonogenesis hypotheses.
	makeSegs := func(word string) []Segment {
		segs := make([]Segment, len(word))
		for i, g := range word {
			segs[i] = Segment{Grapheme: string(g)}
		}
		return segs
	}
	pairs := []FormPair{
		{Src: Form{LectID: "A", Segments: makeSegs("pata")}, Tgt: Form{LectID: "B", Segments: makeSegs("fada")}},
		{Src: Form{LectID: "A", Segments: makeSegs("kata")}, Tgt: Form{LectID: "B", Segments: makeSegs("hada")}},
	}
	model := xanTrainPairwise(t, pairs)
	hypotheses := findResidualPatterns(pairs, model, 5, 100, 0.95, "", 0, nil)
	for _, h := range hypotheses {
		if h.Kind == "cross_dimensional_tonogenesis" {
			t.Errorf("unexpected tonogenesis hypothesis on non-tonal corpus: %v", h)
		}
	}
}

// ----- edge cases -------------------------------------------------------

func TestAnomalyFindResidualPatternsOnEmptyCorpus(t *testing.T) {
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	result := findResidualPatterns(nil, model, 5, 100, 0.95, "", 0, nil)
	if len(result) != 0 {
		t.Errorf("expected empty result on empty corpus, got %v", result)
	}
}

func TestAnomalyFindResidualPatternsRejectsUnknownKind(t *testing.T) {
	// COMMITMENT: findResidualPatterns panics on unknown kinds (Go equivalent
	// of Python's ValueError).
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on unknown hypothesis kind, got none")
		}
	}()
	_ = findResidualPatterns(nil, model, 5, 100, 0.95, "", 0, []string{"made_up_kind"})
}

func TestAnomalyFindResidualPatternsKindsSubsetFiltersOutput(t *testing.T) {
	// COMMITMENT: passing an empty kinds list returns no hypotheses.
	corpus := xanVoicedCorpus()
	model := xanTrainPairwise(t, corpus)
	hypotheses := findResidualPatterns(corpus, model, 5, 100, 0.95, "", 0, []string{})
	if len(hypotheses) != 0 {
		t.Errorf("empty kinds should return no hypotheses, got %d", len(hypotheses))
	}
}

// ----- determinism ------------------------------------------------------

func TestAnomalyFindResidualPatternsIsDeterministicUnderSameSeed(t *testing.T) {
	// COMMITMENT: repeated calls with the same random_seed produce identical
	// hypothesis lists.
	corpus := xanVoicedCorpus()
	model := xanTrainPairwise(t, corpus)
	r1 := findResidualPatterns(corpus, model, 5, 100, 0.95, "", 42, nil)
	r2 := findResidualPatterns(corpus, model, 5, 100, 0.95, "", 42, nil)
	if len(r1) != len(r2) {
		t.Fatalf("hypothesis count differs: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].Kind != r2[i].Kind ||
			r1[i].SrcFeature != r2[i].SrcFeature ||
			r1[i].SrcPositionSpec != r2[i].SrcPositionSpec ||
			r1[i].TgtFeatureOrValue != r2[i].TgtFeatureOrValue ||
			r1[i].ResidualMI != r2[i].ResidualMI {
			t.Errorf("hypothesis[%d] differs between calls", i)
		}
	}
}

func TestAnomalyFindResidualPatternsIsNotMutating(t *testing.T) {
	// COMMITMENT: the function never mutates the model or the corpus.
	corpus := xanVoicedCorpus()
	model := xanTrainPairwise(t, corpus)
	// Capture segment table counts before.
	beforeCounts := make(map[string]float64, len(model.SegmentTable.Counts))
	for k, v := range model.SegmentTable.Counts {
		beforeCounts[k] = v
	}
	_ = findResidualPatterns(corpus, model, 5, 100, 0.95, "", 0, nil)
	// Counts must be unchanged.
	for k, v := range beforeCounts {
		if model.SegmentTable.Counts[k] != v {
			t.Errorf("SegmentTable.Counts[%q] changed from %v to %v", k, v, model.SegmentTable.Counts[k])
		}
	}
	if len(model.SegmentTable.Counts) != len(beforeCounts) {
		t.Errorf("SegmentTable.Counts size changed: %d → %d",
			len(beforeCounts), len(model.SegmentTable.Counts))
	}
}

// ----- sort order -------------------------------------------------------

func TestAnomalyFindResidualPatternsSortedByMIDesc(t *testing.T) {
	// COMMITMENT: the result list is sorted by ResidualMI descending.
	corpus := xanVoicedCorpus()
	model := xanTrainPairwise(t, corpus)
	hypotheses := findResidualPatterns(corpus, model, 5, 100, 0.95, "", 0, nil)
	for i := 1; i < len(hypotheses); i++ {
		if hypotheses[i].ResidualMI > hypotheses[i-1].ResidualMI {
			t.Errorf("hypotheses not sorted by MI desc: [%d]=%v > [%d]=%v",
				i, hypotheses[i].ResidualMI, i-1, hypotheses[i-1].ResidualMI)
		}
	}
}

// ----- PatternHypothesis type -------------------------------------------

func TestAnomalyHypothesisTypeHoldsExpectedFields(t *testing.T) {
	// COMMITMENT: PatternHypothesis is a value type holding all required fields.
	// (skipped: Python frozen-hashable check — Go structs with pointer fields
	// are not directly usable as map keys; we verify structural field access instead.)
	h := PatternHypothesis{
		Kind:                   "cross_dimensional_tonogenesis",
		SrcFeature:             "voiced",
		SrcPositionSpec:        "relative_0",
		TgtFeatureOrValue:      "4",
		TgtDimension:           "tone",
		TgtPositionOffset:      0,
		ResidualMI:             0.5,
		NullPercentile:         0.98,
		SupportingObservations: 20,
		SrcValue:               "+",
	}
	if h.Kind != "cross_dimensional_tonogenesis" {
		t.Errorf("Kind = %q, want cross_dimensional_tonogenesis", h.Kind)
	}
	if h.ResidualMI != 0.5 {
		t.Errorf("ResidualMI = %v, want 0.5", h.ResidualMI)
	}
	if h.NullPercentile != 0.98 {
		t.Errorf("NullPercentile = %v, want 0.98", h.NullPercentile)
	}
	if h.SupportingObservations != 20 {
		t.Errorf("SupportingObservations = %d, want 20", h.SupportingObservations)
	}
	// Verify a copy is independent.
	h2 := h
	h2.Kind = "other"
	if h.Kind == h2.Kind {
		t.Error("original was modified by copy mutation")
	}
}

func TestAnomalyHypothesisKindsTupleIsExposed(t *testing.T) {
	// COMMITMENT: HypothesisKinds lists every supported kind; v1 covers
	// only cross-dimensional tonogenesis.
	found := false
	for _, k := range HypothesisKinds {
		if k == "cross_dimensional_tonogenesis" {
			found = true
		}
	}
	if !found {
		t.Error("HypothesisKinds missing 'cross_dimensional_tonogenesis'")
	}
	if len(HypothesisKinds) != 1 {
		t.Errorf("HypothesisKinds length = %d, want 1", len(HypothesisKinds))
	}
}
