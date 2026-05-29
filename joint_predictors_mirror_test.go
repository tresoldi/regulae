package regulae

// Mirror of python/tests/test_joint_predictors.py

import (
	"testing"
)

// xjpToned builds a Form from a word like "ba2" by attaching the trailing
// digit as a tone to the last vowel. Lect ID is set to the provided lectID.
func xjpToned(lectID, word string) Form {
	vowels := map[byte]bool{'a': true, 'e': true, 'i': true, 'o': true, 'u': true}
	tone := ""
	if len(word) > 0 && word[len(word)-1] >= '0' && word[len(word)-1] <= '9' {
		tone = string(word[len(word)-1])
		word = word[:len(word)-1]
	}
	segs := make([]Segment, len(word))
	toneAttached := false
	// Iterate reversed to find the last vowel.
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

func xjpPair(src, tgt string) FormPair {
	return FormPair{
		Src: Form{LectID: "src", Segments: xjpToned("src", src).Segments},
		Tgt: Form{LectID: "tgt", Segments: xjpToned("tgt", tgt).Segments},
	}
}

func xjpPairModel(t *testing.T, corpus []FormPair) *LearnedModel {
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

// ----- data model -------------------------------------------------------

func TestJointPredictorsCrossDimLinkJointFieldsDefaultNil(t *testing.T) {
	// COMMITMENT: single-predictor rules have both joint extension fields at nil.
	rule := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             40.0,
		SrcCount:          40.0,
		Confidence:        1.0,
	}
	if rule.SrcFeature2 != nil {
		t.Errorf("expected SrcFeature2 = nil, got %v", rule.SrcFeature2)
	}
	if rule.SrcPosition2 != "" {
		t.Errorf("expected SrcPosition2 = \"\", got %q", rule.SrcPosition2)
	}
}

func TestJointPredictorsCrossDimLinkJointFieldsSettable(t *testing.T) {
	fc2 := FeatureConstraint{Feature: "tone", Value: "2"}
	rule := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		SrcFeature2:       &fc2,
		SrcPosition2:      "relative_0",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             10.0,
		SrcCount:          12.0,
		Confidence:        10.0 / 12.0,
	}
	if rule.SrcFeature2 == nil {
		t.Fatal("expected SrcFeature2 != nil")
	}
	if rule.SrcFeature2.Feature != "tone" {
		t.Errorf("SrcFeature2.Feature = %q, want %q", rule.SrcFeature2.Feature, "tone")
	}
	if rule.SrcFeature2.Value != "2" {
		t.Errorf("SrcFeature2.Value = %q, want %q", rule.SrcFeature2.Value, "2")
	}
	if rule.SrcPosition2 != "relative_0" {
		t.Errorf("SrcPosition2 = %q, want %q", rule.SrcPosition2, "relative_0")
	}
}

// ----- scoring overlay --------------------------------------------------

func TestJointPredictorsJointRuleFiresOnlyWhenBothPredicatesHold(t *testing.T) {
	// COMMITMENT: a joint rule's adjustment applies only when both predicates
	// hold on the source side.
	fc2 := FeatureConstraint{Feature: "tone", Value: "2"}
	rule := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		SrcFeature2:       &fc2,
		SrcPosition2:      "relative_0",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             10.0,
		SrcCount:          12.0,
		Confidence:        10.0 / 12.0,
	}
	posAdj := -1.5
	negAdj := +0.8

	// "ba2": voiced initial (b), vowel a with tone 2.
	// Link position at the vowel (index 1):
	//   relative_-1 → index 0 (b, voiced) ✓
	//   relative_0  → index 1 (a, tone=2) ✓
	srcMatch := xjpToned("src", "ba2")
	tgtMatch := xjpToned("tgt", "ba4")
	cost := applyRuleToLink(rule, posAdj, negAdj, srcMatch, 1, tgtMatch, 1, "descriptive")
	if cost != posAdj {
		t.Errorf("both predicates hold + match: cost = %v, want %v", cost, posAdj)
	}

	// Miss on component 1: initial is voiceless (p).
	srcVL := xjpToned("src", "pa2")
	costVL := applyRuleToLink(rule, posAdj, negAdj, srcVL, 1, tgtMatch, 1, "descriptive")
	if costVL != 0.0 {
		t.Errorf("voiceless initial: cost = %v, want 0.0", costVL)
	}

	// Miss on component 2: wrong source tone (3 instead of 2).
	srcWrongTone := xjpToned("src", "ba3")
	costWT := applyRuleToLink(rule, posAdj, negAdj, srcWrongTone, 1, tgtMatch, 1, "descriptive")
	if costWT != 0.0 {
		t.Errorf("wrong source tone: cost = %v, want 0.0", costWT)
	}
}

// ----- commit loop ------------------------------------------------------

func xjpJointSignalCorpus() []FormPair {
	// A corpus where the rule is a JOINT predictor: target tone is 4 only
	// when the source initial is voiced AND the source vowel has tone 2.
	var pairs []FormPair

	// Voiced + tone 2: target becomes tone 4.  12 pairs.
	for _, init := range []string{"b", "d", "g", "m"} {
		for _, vowel := range []string{"a", "i", "u"} {
			pairs = append(pairs, xjpPair(init+vowel+"2", init+vowel+"4"))
		}
	}
	// Voiced + tone 1: target keeps tone 1.
	for _, init := range []string{"b", "d", "g"} {
		for _, vowel := range []string{"a", "i", "u", "o"} {
			pairs = append(pairs, xjpPair(init+vowel+"1", init+vowel+"1"))
		}
	}
	// Voiceless + tone 2: target keeps tone 2.
	for _, init := range []string{"p", "t", "k", "s"} {
		for _, vowel := range []string{"a", "i", "u"} {
			pairs = append(pairs, xjpPair(init+vowel+"2", init+vowel+"2"))
		}
	}
	// Voiceless + tone 1: target keeps tone 1.
	for _, init := range []string{"p", "t", "k"} {
		for _, vowel := range []string{"a", "i", "u", "o"} {
			pairs = append(pairs, xjpPair(init+vowel+"1", init+vowel+"1"))
		}
	}
	return pairs
}

func TestJointPredictorsCrossDimCommitsJointRuleOnDesignedJointSignal(t *testing.T) {
	// COMMITMENT: on a corpus designed so that the rule fires only when
	// (voicing AND source-tone) jointly hold, cross-dimensional discovery
	// commits at least one joint rule with SrcFeature2 set and high confidence.
	model := xjpPairModel(t, xjpJointSignalCorpus())
	entries := model.CrossDimensionalTable.Entries
	if len(entries) == 0 {
		t.Fatal("expected at least one committed rule, got none")
	}
	var jointRules []CrossDimensionalLink
	for _, e := range entries {
		if e.SrcFeature2 != nil {
			jointRules = append(jointRules, e)
		}
	}
	if len(jointRules) < 1 {
		t.Fatalf("no joint rule committed; entries: %v", entries)
	}
	// The committed joint should be voiced + tone=2 → tone=4 or equivalent.
	var matching []CrossDimensionalLink
	for _, r := range jointRules {
		if r.TgtValue != "4" || r.Confidence < 0.8 {
			continue
		}
		// Check for voiced×tone or tone×voiced framing.
		voicedTone := r.SrcFeature.Feature == "voiced" && r.SrcFeature2 != nil && r.SrcFeature2.Feature == "tone"
		toneVoiced := r.SrcFeature.Feature == "tone" && r.SrcFeature2 != nil && r.SrcFeature2.Feature == "voiced"
		if voicedTone || toneVoiced {
			matching = append(matching, r)
		}
	}
	if len(matching) < 1 {
		t.Errorf("no voiced × tone=2 → tone=4 joint committed; joint rules: %v", jointRules)
	}
}

func TestJointPredictorsCrossDimDoesNotCommitJointWhenSinglePredictorSuffices(t *testing.T) {
	// COMMITMENT: on a corpus where a single-predictor rule fully explains the
	// signal (voicing alone predicts target tone), no joint rule is committed.
	var pairs []FormPair
	for _, srcTone := range []string{"1", "2", "3", "4"} {
		for _, init := range []string{"p", "t", "k", "s"} {
			for _, vowel := range []string{"a", "i", "u"} {
				pairs = append(pairs, xjpPair(init+vowel+srcTone, init+vowel+"1"))
			}
		}
		for _, init := range []string{"b", "d", "g", "m"} {
			for _, vowel := range []string{"a", "i", "u"} {
				pairs = append(pairs, xjpPair(init+vowel+srcTone, init+vowel+"4"))
			}
		}
	}
	model := xjpPairModel(t, pairs)
	for _, e := range model.CrossDimensionalTable.Entries {
		if e.SrcFeature2 != nil {
			t.Errorf("unexpected joint rule on clean single-predictor signal: %v", e)
		}
	}
}

func TestJointPredictorsCrossDimJointRejectsPassengerPredictor(t *testing.T) {
	// COMMITMENT: the non-interaction guard filters "passenger" joints where
	// one predictor is tautologically implied by the other.
	var pairs []FormPair
	voiced := []string{"b", "d", "g", "m"}
	voiceless := []string{"p", "t", "k", "s"}
	for range [3]struct{}{} {
		for _, init := range voiced {
			for _, vowel := range []string{"a", "i"} {
				pairs = append(pairs, xjpPair(init+vowel+"2", init+vowel+"4"))
			}
		}
		for _, init := range voiceless {
			for _, vowel := range []string{"a", "i"} {
				pairs = append(pairs, xjpPair(init+vowel+"1", init+vowel+"1"))
			}
		}
	}
	model := xjpPairModel(t, pairs)
	for _, e := range model.CrossDimensionalTable.Entries {
		if e.SrcFeature2 != nil {
			t.Errorf("expected no joint rules on passenger-predictor signal; got %v", e)
		}
	}
}

// ----- dedup and signature -----------------------------------------------

func TestJointPredictorsJointRuleSignatureIncludesSecondFeature(t *testing.T) {
	// COMMITMENT: ruleSignature distinguishes joint rules from single-predictor
	// rules with the same first feature.
	single := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             20.0,
		SrcCount:          20.0,
		Confidence:        1.0,
	}
	fc2 := FeatureConstraint{Feature: "tone", Value: "2"}
	joint := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		SrcFeature2:       &fc2,
		SrcPosition2:      "relative_0",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             12.0,
		SrcCount:          14.0,
		Confidence:        12.0 / 14.0,
	}
	sigSingle := ruleSignature(single)
	sigJoint := ruleSignature(joint)
	if sigSingle == sigJoint {
		t.Errorf("single and joint rules have the same signature: %q", sigSingle)
	}
}
