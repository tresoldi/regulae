package regulae

// Mirror of python/tests/test_tonal_predictors.py

import (
	"testing"
)

// xtpToned builds a Form (same convention as xjpToned) — trailing digit
// is attached as a tone to the last vowel.
func xtpToned(lectID, word string) Form {
	vowels := map[byte]bool{'a': true, 'e': true, 'i': true, 'o': true, 'u': true}
	tone := ""
	if len(word) > 0 && word[len(word)-1] >= '0' && word[len(word)-1] <= '9' {
		tone = string(word[len(word)-1])
		word = word[:len(word)-1]
	}
	segs := make([]Segment, len(word))
	toneAttached := false
	for i := len(word) - 1; i >= 0; i-- {
		ch := word[i]
		if !toneAttached && vowels[ch] && tone != "" {
			segs[i] = Segment{Grapheme: string(ch), Tone: tone}
			toneAttached = true
		} else {
			segs[i] = Segment{Grapheme: string(ch)}
		}
	}
	return Form{LectID: lectID, Segments: segs}
}

func xtpPair(src, tgt string) FormPair {
	return FormPair{
		Src: Form{LectID: "src", Segments: xtpToned("src", src).Segments},
		Tgt: Form{LectID: "tgt", Segments: xtpToned("tgt", tgt).Segments},
	}
}

func xtpPairModel(t *testing.T, corpus []FormPair) *LearnedModel {
	t.Helper()
	csets := CognateSetsFromPairs(corpus, [2]string{"src", "tgt"}, "")
	multi, err := TrainModel(csets, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel: %v", err)
	}
	m, ok := multi.PairwiseModel("src", "tgt")
	if !ok {
		t.Fatal("no pairwise model for (src, tgt)")
	}
	return m
}

// xtpToneShiftCorpus builds 30 pairs with clean source→target tone mapping:
// source tone 1 → target tone 3, source tone 2 → target tone 1.
func xtpToneShiftCorpus() []FormPair {
	words1 := []string{
		"pa1", "ka1", "ta1", "ba1", "ma1",
		"pi1", "ki1", "ti1", "bi1", "mi1",
		"pu1", "ku1", "tu1", "bu1", "mu1",
	}
	words2 := []string{
		"pa2", "ka2", "ta2", "ba2", "ma2",
		"pi2", "ki2", "ti2", "bi2", "mi2",
		"pu2", "ku2", "tu2", "bu2", "mu2",
	}
	var pairs []FormPair
	for _, w := range words1 {
		tgt := w[:len(w)-1] + "3"
		pairs = append(pairs, xtpPair(w, tgt))
	}
	for _, w := range words2 {
		tgt := w[:len(w)-1] + "1"
		pairs = append(pairs, xtpPair(w, tgt))
	}
	return pairs
}

// ----- commit: tonal predictor ------------------------------------------

func TestTonalPredictorsCrossDimCommitsTonalPredictorRule(t *testing.T) {
	// COMMITMENT: on a corpus where the cross-dim rule is source-tone →
	// target-tone, cross-dimensional discovery commits at least one rule
	// whose SrcFeature.Feature is "tone" and SrcFeature.Value is a source
	// tone value from the corpus.
	model := xtpPairModel(t, xtpToneShiftCorpus())
	entries := model.CrossDimensionalTable.Entries
	if len(entries) == 0 {
		t.Fatal("no cross-dim rules committed at all")
	}
	var tonalEntries []CrossDimensionalLink
	for _, e := range entries {
		if e.SrcFeature.Feature == "tone" {
			tonalEntries = append(tonalEntries, e)
		}
	}
	if len(tonalEntries) < 1 {
		t.Fatalf("no tonal source predictor committed; entries: %v", entries)
	}
	// Validate one of the two expected shifts.
	var matches []CrossDimensionalLink
	for _, e := range tonalEntries {
		if (e.SrcFeature.Value == "1" && e.TgtValue == "3") ||
			(e.SrcFeature.Value == "2" && e.TgtValue == "1") {
			matches = append(matches, e)
		}
	}
	if len(matches) < 1 {
		t.Errorf("no expected tone-shift rule found; tonal entries: %v", tonalEntries)
	}
}

func TestTonalPredictorsCrossDimTonalRuleHighConfidenceOnCleanShift(t *testing.T) {
	// COMMITMENT: a clean tonal shift (no noise) produces a rule at
	// confidence >= 0.95.
	model := xtpPairModel(t, xtpToneShiftCorpus())
	var tonalEntries []CrossDimensionalLink
	for _, e := range model.CrossDimensionalTable.Entries {
		if e.SrcFeature.Feature == "tone" {
			tonalEntries = append(tonalEntries, e)
		}
	}
	found := false
	for _, e := range tonalEntries {
		if e.Confidence >= 0.95 {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected at least one near-1.0-confidence tonal rule; got %v", tonalEntries)
	}
}

func TestTonalPredictorsScoringAppliesTonalPredictorAdjustment(t *testing.T) {
	// COMMITMENT: a CrossDimensionalLink with src_feature=tone applies its
	// adjustment when the source tone matches and skips it when it doesn't.
	rule := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "tone", Value: "1"},
		SrcPosition:       "relative_0",
		TgtDimension:      "tone",
		TgtValue:          "3",
		TgtPositionOffset: 0,
		Count:             10.0,
		SrcCount:          10.0,
		Confidence:        1.0,
	}
	posAdj := -1.5
	negAdj := +0.8

	// "pa1": vowel at index 1 carries tone 1.
	src := xtpToned("src", "pa1")
	tgtMatch := xtpToned("tgt", "pa3") // tone 3 = rule's target
	tgtMiss := xtpToned("tgt", "pa5")  // tone 5 ≠ rule's target

	// Position 1 is the vowel; relative_0 from pos 1 = pos 1 (the vowel itself).
	costMatch := applyRuleToLink(rule, posAdj, negAdj, src, 1, tgtMatch, 1, "descriptive")
	if costMatch != posAdj {
		t.Errorf("match: cost = %v, want %v", costMatch, posAdj)
	}

	costMiss := applyRuleToLink(rule, posAdj, negAdj, src, 1, tgtMiss, 1, "descriptive")
	if costMiss != negAdj {
		t.Errorf("miss: cost = %v, want %v", costMiss, negAdj)
	}

	// Wrong source tone: rule does not fire (returns 0.0).
	srcWrongTone := xtpToned("src", "pa2")
	costNoFire := applyRuleToLink(rule, posAdj, negAdj, srcWrongTone, 1, tgtMatch, 1, "descriptive")
	if costNoFire != 0.0 {
		t.Errorf("wrong source tone: cost = %v, want 0.0", costNoFire)
	}
}

func TestTonalPredictorsAnomalyEmitsTonalHypotheses(t *testing.T) {
	// COMMITMENT: findResidualPatterns surfaces source-tone predictors as
	// PatternHypothesis entries with SrcFeature == "tone" and a non-default
	// SrcValue.
	corpus := xtpToneShiftCorpus()
	csets := CognateSetsFromPairs(corpus, [2]string{"src", "tgt"}, "")
	multi, err := TrainModel(csets, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel: %v", err)
	}
	pairModel, _ := multi.PairwiseModel("src", "tgt")
	hypotheses := findResidualPatterns(corpus, pairModel, 3, 100, 0.95, "", 0, nil)
	var tonal []PatternHypothesis
	for _, h := range hypotheses {
		if h.SrcFeature == "tone" {
			tonal = append(tonal, h)
		}
	}
	if len(tonal) < 1 {
		t.Fatalf("expected at least one tonal hypothesis, got none; all: %v", hypotheses)
	}
	// Every tonal hypothesis carries a non-"+" src_value (actual source tone).
	for _, h := range tonal {
		if h.SrcValue != "1" && h.SrcValue != "2" {
			t.Errorf("tonal hypothesis has unexpected SrcValue %q (want '1' or '2')", h.SrcValue)
		}
	}
}

func TestTonalPredictorsTonalPredictorNotCommittedOnNonTonalCorpus(t *testing.T) {
	// COMMITMENT: on a corpus without any source-side tones, no tonal
	// predictor rules are committed.
	raw := []FormPair{
		xtpPair("pata", "pada"),
		xtpPair("kata", "kada"),
		xtpPair("tama", "tada"),
		xtpPair("pima", "pida"),
		xtpPair("tama", "tada"),
	}
	// Inflate to meet min counts.
	var corpus []FormPair
	for range [3]struct{}{} {
		corpus = append(corpus, raw...)
	}
	model := xtpPairModel(t, corpus)
	for _, e := range model.CrossDimensionalTable.Entries {
		if e.SrcFeature.Feature == "tone" {
			t.Errorf("unexpected tonal predictor rule on non-tonal corpus: %v", e)
		}
	}
}
