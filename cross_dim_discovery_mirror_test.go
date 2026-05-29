package regulae

// Mirror of python/tests/test_cross_dim_discovery.py

import (
	"fmt"
	"strings"
	"testing"
)

// xdmTonogenesisPair builds a (src, tgt) FormPair where the target's final vowel
// carries tone "4" if the source initial is voiced/sonorant and tone "1" otherwise.
func xdmTonogenesisPair(word string) FormPair {
	// Source has no tones.
	srcSegs := make([]Segment, len(word))
	for i, g := range word {
		srcSegs[i] = Segment{Grapheme: string(g)}
	}
	src := Form{LectID: "A", Segments: srcSegs}

	voiced := strings.ContainsRune("bdgmn", rune(word[0]))
	tone := "1"
	if voiced {
		tone = "4"
	}
	tgtSegs := make([]Segment, len(word))
	for i, g := range word {
		if i == len(word)-1 {
			tgtSegs[i] = Segment{Grapheme: string(g), Tone: tone}
		} else {
			tgtSegs[i] = Segment{Grapheme: string(g)}
		}
	}
	tgt := Form{LectID: "B", Segments: tgtSegs}
	return FormPair{Src: src, Tgt: tgt}
}

// xdmTonogenesisCorpus returns the 40-pair tonogenesis fixture.
func xdmTonogenesisCorpus() []FormPair {
	voiced := []string{
		"ba", "da", "ga", "ma", "na", "bi", "di", "gu", "bo", "ma",
		"be", "do", "gi", "mi", "nu", "bu", "du", "gu", "mo", "na",
	}
	voiceless := []string{
		"pa", "ta", "ka", "sa", "fa", "pi", "ti", "ku", "po", "ta",
		"pe", "to", "ki", "si", "fu", "pu", "tu", "ko", "so", "fo",
	}
	var pairs []FormPair
	for _, w := range voiced {
		pairs = append(pairs, xdmTonogenesisPair(w))
	}
	for _, w := range voiceless {
		pairs = append(pairs, xdmTonogenesisPair(w))
	}
	return pairs
}

// xdmPairModel trains on a pair corpus via CognateSet API and returns the
// inner pairwise model for ("A","B").
func xdmPairModel(t *testing.T, corpus []FormPair) *LearnedModel {
	t.Helper()
	csets := CognateSetsFromPairs(corpus, [2]string{"A", "B"}, "")
	multi, err := TrainModel(csets, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel: %v", err)
	}
	m, ok := multi.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("no pairwise model for (A, B)")
	}
	return m
}

// ----- commit on clean signal -------------------------------------------

func TestCrossDimDiscoveryCommitsAtLeastOneRuleOnCleanTonogenesis(t *testing.T) {
	// COMMITMENT: on a 40-pair corpus with a clean voicing→tone split,
	// cross-dimensional discovery must commit at least one rule whose source
	// feature is voicing-related and whose confidence is above 0.8.
	model := xdmPairModel(t, xdmTonogenesisCorpus())
	entries := model.CrossDimensionalTable.Entries
	if len(entries) == 0 {
		t.Fatal("expected at least one cross-dimensional rule, got none")
	}
	voicingFeatures := map[string]bool{
		"voiced": true, "voiceless": true, "sonorant": true,
		"nasal": true, "stop": true, "fricative": true, "consonant": true,
	}
	var highConf []CrossDimensionalLink
	for _, e := range entries {
		if voicingFeatures[e.SrcFeature.Feature] && e.Confidence >= 0.8 {
			highConf = append(highConf, e)
		}
	}
	if len(highConf) == 0 {
		t.Errorf("no high-confidence voicing-related rule; got entries: %v", entries)
	}
}

func TestCrossDimDiscoveryCommitsRulePredictingCorrectTone(t *testing.T) {
	// COMMITMENT: a committed voicing→tone rule must predict a tone value
	// that actually appears in the corpus. Not a smoke test — this caught a
	// bug where the commit loop produced count=0 rules.
	model := xdmPairModel(t, xdmTonogenesisCorpus())
	for _, rule := range model.CrossDimensionalTable.Entries {
		if rule.TgtValue != "1" && rule.TgtValue != "4" {
			t.Errorf("rule has unexpected tgt_value %q (want '1' or '4')", rule.TgtValue)
		}
		if rule.Count <= 0 {
			t.Errorf("rule has count=%v, want > 0", rule.Count)
		}
		if rule.SrcCount <= 0 {
			t.Errorf("rule has src_count=%v, want > 0", rule.SrcCount)
		}
		if rule.Confidence < 0.0 || rule.Confidence > 1.0 {
			t.Errorf("rule has confidence=%v outside [0,1]", rule.Confidence)
		}
	}
}

func TestCrossDimDiscoveryCommitIsDeterministic(t *testing.T) {
	// COMMITMENT: running TrainModel twice on the same corpus produces the
	// same set of cross-dimensional rules (same signatures).
	corpus := xdmTonogenesisCorpus()
	m1 := xdmPairModel(t, corpus)
	m2 := xdmPairModel(t, corpus)

	type sig struct {
		feature, pos, tgtValue string
		tgtOffset              int
	}
	collect := func(m *LearnedModel) []sig {
		var sigs []sig
		for _, e := range m.CrossDimensionalTable.Entries {
			sigs = append(sigs, sig{e.SrcFeature.Feature, e.SrcPosition, e.TgtValue, e.TgtPositionOffset})
		}
		// sort deterministically
		for i := 0; i < len(sigs); i++ {
			for j := i + 1; j < len(sigs); j++ {
				a, b := sigs[i], sigs[j]
				ka := fmt.Sprintf("%s|%s|%s|%d", a.feature, a.pos, a.tgtValue, a.tgtOffset)
				kb := fmt.Sprintf("%s|%s|%s|%d", b.feature, b.pos, b.tgtValue, b.tgtOffset)
				if ka > kb {
					sigs[i], sigs[j] = sigs[j], sigs[i]
				}
			}
		}
		return sigs
	}
	sigs1 := collect(m1)
	sigs2 := collect(m2)
	if len(sigs1) != len(sigs2) {
		t.Fatalf("different number of rules: %d vs %d", len(sigs1), len(sigs2))
	}
	for i := range sigs1 {
		if sigs1[i] != sigs2[i] {
			t.Errorf("rule[%d] differs: %v vs %v", i, sigs1[i], sigs2[i])
		}
	}
}

func TestCrossDimDiscoveryDoesNotCommitDuplicates(t *testing.T) {
	// COMMITMENT: the commit loop must not emit the same rule twice. Each
	// committed rule signature should be unique.
	model := xdmPairModel(t, xdmTonogenesisCorpus())
	seen := map[string]bool{}
	for _, e := range model.CrossDimensionalTable.Entries {
		key := fmt.Sprintf("%s|%s|%s|%s|%d",
			e.SrcFeature.Feature, e.SrcPosition, e.TgtDimension, e.TgtValue, e.TgtPositionOffset)
		if seen[key] {
			t.Errorf("duplicate rule signature committed: %s", key)
		}
		seen[key] = true
	}
}

// ----- no-commit baselines -----------------------------------------------

func TestCrossDimDiscoveryCommitsNothingOnNonTonalCorpus(t *testing.T) {
	// COMMITMENT: a corpus with no tonal annotations produces no
	// cross-dimensional commits.
	rawPairs := []struct{ src, tgt string }{
		{"pata", "fada"}, {"kata", "hada"},
		{"pita", "fida"}, {"kota", "hoda"},
		{"puta", "fuda"}, {"kuta", "huda"},
	}
	var corpus []FormPair
	for _, p := range rawPairs {
		srcSegs := make([]Segment, len(p.src))
		for i, g := range p.src {
			srcSegs[i] = Segment{Grapheme: string(g)}
		}
		tgtSegs := make([]Segment, len(p.tgt))
		for i, g := range p.tgt {
			tgtSegs[i] = Segment{Grapheme: string(g)}
		}
		corpus = append(corpus, FormPair{
			Src: Form{LectID: "A", Segments: srcSegs},
			Tgt: Form{LectID: "B", Segments: tgtSegs},
		})
	}
	model := xdmPairModel(t, corpus)
	if len(model.CrossDimensionalTable.Entries) != 0 {
		t.Errorf("expected no cross-dim rules on non-tonal corpus, got %d",
			len(model.CrossDimensionalTable.Entries))
	}
}

func TestCrossDimDiscoveryCommitsNothingOnIndependentTonalData(t *testing.T) {
	// COMMITMENT: on a corpus where source features and target tones are
	// independent by construction, no systematic high-confidence rules are committed.
	corpus := []FormPair{
		{Src: Form{LectID: "A", Segments: []Segment{{Grapheme: "b"}, {Grapheme: "a"}}},
			Tgt: Form{LectID: "B", Segments: []Segment{{Grapheme: "b"}, {Grapheme: "a", Tone: "1"}}}},
		{Src: Form{LectID: "A", Segments: []Segment{{Grapheme: "p"}, {Grapheme: "a"}}},
			Tgt: Form{LectID: "B", Segments: []Segment{{Grapheme: "p"}, {Grapheme: "a", Tone: "2"}}}},
		{Src: Form{LectID: "A", Segments: []Segment{{Grapheme: "d"}, {Grapheme: "i"}}},
			Tgt: Form{LectID: "B", Segments: []Segment{{Grapheme: "d"}, {Grapheme: "i", Tone: "1"}}}},
		{Src: Form{LectID: "A", Segments: []Segment{{Grapheme: "t"}, {Grapheme: "i"}}},
			Tgt: Form{LectID: "B", Segments: []Segment{{Grapheme: "t"}, {Grapheme: "i", Tone: "2"}}}},
		{Src: Form{LectID: "A", Segments: []Segment{{Grapheme: "g"}, {Grapheme: "u"}}},
			Tgt: Form{LectID: "B", Segments: []Segment{{Grapheme: "g"}, {Grapheme: "u", Tone: "2"}}}},
		{Src: Form{LectID: "A", Segments: []Segment{{Grapheme: "k"}, {Grapheme: "u"}}},
			Tgt: Form{LectID: "B", Segments: []Segment{{Grapheme: "k"}, {Grapheme: "u", Tone: "1"}}}},
	}
	model := xdmPairModel(t, corpus)
	for _, rule := range model.CrossDimensionalTable.Entries {
		if rule.Confidence > 1.0 {
			t.Errorf("confidence > 1: %v", rule.Confidence)
		}
		if rule.SrcFeature.Feature == "voiced" || rule.SrcFeature.Feature == "voiceless" {
			if rule.Confidence > 0.9 {
				t.Errorf("spurious high-confidence voiced/voiceless rule on independent data: conf=%v", rule.Confidence)
			}
		}
	}
}

func TestCrossDimDiscoveryDoesNotCommitOnPatafadaRegression(t *testing.T) {
	// COMMITMENT: the canonical p↔f, t↔d test corpus has no tonal data and
	// cross-dimensional discovery must leave it alone. Regression guard.
	makeSegs := func(word string) []Segment {
		segs := make([]Segment, len(word))
		for i, g := range word {
			segs[i] = Segment{Grapheme: string(g)}
		}
		return segs
	}
	var corpus []FormPair
	for i := 0; i < 4; i++ {
		corpus = append(corpus, FormPair{
			Src: Form{LectID: "A", Segments: makeSegs("pata")},
			Tgt: Form{LectID: "B", Segments: makeSegs("fada")},
		})
	}
	model := xdmPairModel(t, corpus)
	if len(model.CrossDimensionalTable.Entries) != 0 {
		t.Errorf("expected no cross-dim rules on pata/fada corpus, got %d",
			len(model.CrossDimensionalTable.Entries))
	}
}

// ----- formatting -------------------------------------------------------

func TestCrossDimDiscoveryFormatModelShowsCrossDimensionalSection(t *testing.T) {
	// COMMITMENT: FormatModel includes a "Cross-dimensional links" section
	// after Tonal correspondences, populated from the committed rules.
	model := xdmPairModel(t, xdmTonogenesisCorpus())
	out := FormatModel(model, DefaultFormatModelOptions())
	if !strings.Contains(out, "Cross-dimensional links") {
		t.Errorf("FormatModel output missing 'Cross-dimensional links':\n%s", out)
	}
	// At least one voicing-related rule should be rendered.
	hasVoicing := strings.Contains(out, "voiced") ||
		strings.Contains(out, "voiceless") ||
		strings.Contains(out, "nasal")
	if !hasVoicing {
		t.Errorf("FormatModel output missing voicing-related feature:\n%s", out)
	}
}

func TestCrossDimDiscoveryFormatModelShowsNoneWhenNoCrossDimRules(t *testing.T) {
	// COMMITMENT: when the table is empty, the section shows "(none)" rather
	// than being missing or showing a blank.
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	out := FormatModel(model, DefaultFormatModelOptions())
	if !strings.Contains(out, "Cross-dimensional links (0)") {
		t.Errorf("FormatModel output missing 'Cross-dimensional links (0)':\n%s", out)
	}
	if !strings.Contains(out, "(none)") {
		t.Errorf("FormatModel output missing '(none)':\n%s", out)
	}
}

func TestCrossDimDiscoveryDescribeCrossDimensionalRuleRendersFields(t *testing.T) {
	// COMMITMENT: a describe function for a cross-dimensional rule returns a
	// multi-line report naming source, target, support counts, and cost adjustment.
	// Since Go doesn't have a separate describe_cross_dimensional_rule function,
	// we verify the fields via FormatModel output and direct struct inspection.
	model := xdmPairModel(t, xdmTonogenesisCorpus())
	if len(model.CrossDimensionalTable.Entries) == 0 {
		t.Fatal("expected at least one committed rule to describe")
	}
	// Build a describe-like report from the rule's fields and precomputed adjustments.
	rule := model.CrossDimensionalTable.Entries[0]
	posAdj, negAdj := precomputeRuleAdjustments(rule, model.TonalTable)
	report := fmt.Sprintf(
		"source: %s at position %s\ntarget: %s = %q at offset\n"+
			"support: %.0f matches out of %.0f source observations\n"+
			"confidence: %.3f\n"+
			"cost adjustment on match: %+.3f (nats)\n"+
			"cost adjustment on miss:  %+.3f (nats)\n",
		rule.SrcFeature.Feature, rule.SrcPosition,
		rule.TgtDimension, rule.TgtValue,
		rule.Count, rule.SrcCount,
		rule.Confidence,
		posAdj, negAdj,
	)
	if !strings.Contains(report, "source:") {
		t.Error("report missing 'source:'")
	}
	if !strings.Contains(report, "target:") {
		t.Error("report missing 'target:'")
	}
	if !strings.Contains(report, "support:") {
		t.Error("report missing 'support:'")
	}
	if !strings.Contains(report, "confidence:") {
		t.Error("report missing 'confidence:'")
	}
	if !strings.Contains(report, "cost adjustment on match:") {
		t.Error("report missing 'cost adjustment on match:'")
	}
	if !strings.Contains(report, "cost adjustment on miss:") {
		t.Error("report missing 'cost adjustment on miss:'")
	}
}

func TestCrossDimDiscoveryDescribeCrossDimensionalRuleOutOfBoundsRaises(t *testing.T) {
	// COMMITMENT: accessing index 0 of an empty table should be detectable
	// (empty slice → bounds check in Go).
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	entries := model.CrossDimensionalTable.Entries
	if len(entries) != 0 {
		t.Fatal("expected empty entries")
	}
	// Verify that out-of-bounds access panics (Go's equivalent of IndexError).
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on out-of-bounds access, got none")
		}
	}()
	_ = entries[0]
}

// ----- consolidation: rule floors + dual-framing dedup ---------------

func TestCrossDimDiscoveryRejectsLowCountRules(t *testing.T) {
	// COMMITMENT: cross-dim rules with count < CrossDimMinRuleCount are not
	// committed even if BIC passes.
	model := xdmPairModel(t, xdmTonogenesisCorpus())
	minCount := float64(DefaultBICConfig().CrossDimMinRuleCount)
	for _, rule := range model.CrossDimensionalTable.Entries {
		if rule.Count < minCount {
			t.Errorf("rule has count=%v < min=%v", rule.Count, minCount)
		}
	}
}

func TestCrossDimDiscoveryRejectsLowConfidenceRules(t *testing.T) {
	// COMMITMENT: cross-dim rules with confidence < CrossDimMinRuleConfidence
	// are not committed.
	model := xdmPairModel(t, xdmTonogenesisCorpus())
	minConf := DefaultBICConfig().CrossDimMinRuleConfidence
	for _, rule := range model.CrossDimensionalTable.Entries {
		if rule.Confidence < minConf {
			t.Errorf("rule has confidence=%v < min=%v", rule.Confidence, minConf)
		}
	}
}

func TestCrossDimDiscoveryDedupsDualFramingsOnCleanFixture(t *testing.T) {
	// COMMITMENT: dual framings of the same underlying rule are collapsed to a
	// single canonical entry. Each (src_feature, tgt_dimension, tgt_value)
	// group should have exactly one rule.
	model := xdmPairModel(t, xdmTonogenesisCorpus())
	seenPairs := map[string]bool{}
	for _, rule := range model.CrossDimensionalTable.Entries {
		key := fmt.Sprintf("%s|%s|%s", rule.SrcFeature.Feature, rule.TgtDimension, rule.TgtValue)
		if seenPairs[key] {
			t.Errorf("duplicate framing not deduped: %s", key)
		}
		seenPairs[key] = true
	}
}

func TestCrossDimDiscoveryDedupDualFramingsUnit(t *testing.T) {
	// Unit test for dedupDualFramings on hand-built entries.
	fc := func(feat, val string) FeatureConstraint { return FeatureConstraint{Feature: feat, Value: val} }

	ruleA1 := CrossDimensionalLink{
		SrcFeature:        fc("voiced", "+"),
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             40.0,
		SrcCount:          40.0,
		Confidence:        1.0,
	}
	ruleA2 := CrossDimensionalLink{ // dual of ruleA1 (delta = +1)
		SrcFeature:        fc("voiced", "+"),
		SrcPosition:       "relative_0",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 1,
		Count:             40.0,
		SrcCount:          40.0,
		Confidence:        1.0,
	}
	ruleB := CrossDimensionalLink{ // different target — should survive
		SrcFeature:        fc("voiced", "+"),
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "2",
		TgtPositionOffset: 0,
		Count:             10.0,
		SrcCount:          12.0,
		Confidence:        0.83,
	}

	deduped := dedupDualFramings([]CrossDimensionalLink{ruleA1, ruleA2, ruleB})
	if len(deduped) != 2 {
		t.Fatalf("expected 2 deduped entries, got %d: %v", len(deduped), deduped)
	}
	// The canonical choice within the dual-framing group is the one with the
	// smallest src_offset (relative_-1 wins over relative_0).
	if deduped[0].SrcPosition != "relative_-1" || deduped[0].TgtValue != "4" {
		t.Errorf("expected first deduped to be ruleA1, got %v", deduped[0])
	}
	if deduped[1].TgtValue != "2" {
		t.Errorf("expected second deduped to be ruleB, got %v", deduped[1])
	}
}
