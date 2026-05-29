package regulae

// Mirror of python/tests/test_multi_lect_cross_dim.py

import "testing"

// mltBuildTonogenesisSet builds one 3-way cognate set where each lect has one
// CV word whose vowel carries a lect-specific tone.
func mltBuildTonogenesisSet(
	cognateID, protoInitial, protoTone, daughterATone, daughterBTone, vowel string,
) CognateSet {
	mkForm := func(lect, initial, v, tone string) Form {
		segs := []Segment{
			{Grapheme: initial},
			{Grapheme: v, Tone: tone},
		}
		return Form{LectID: lect, Segments: segs}
	}
	return CognateSet{
		CognateID: cognateID,
		Forms: map[string]Form{
			"proto":      mkForm("proto", protoInitial, vowel, protoTone),
			"daughter_a": mkForm("daughter_a", protoInitial, vowel, daughterATone),
			"daughter_b": mkForm("daughter_b", protoInitial, vowel, daughterBTone),
		},
		FormsOrder: []string{"proto", "daughter_a", "daughter_b"},
		Confidence: 1.0,
	}
}

// mltTonogenesisCorpus builds the 80-set tonogenesis corpus from the Python test:
//
//	daughter_a: voiceless → tone 1, voiced → tone 4
//	daughter_b: voiceless → tone 2, voiced → tone 3
func mltTonogenesisCorpus() []CognateSet {
	var corpus []CognateSet
	count := 0
	for _, initial := range []string{"p", "t", "b", "d"} {
		voiced := initial == "b" || initial == "d"
		for _, vowel := range []string{"i", "a", "u", "o", "e"} {
			for _, srcTone := range []string{"1", "2", "3", "4"} {
				count++
				aTone := "1"
				bTone := "2"
				if voiced {
					aTone = "4"
					bTone = "3"
				}
				id := "cs-" + string(rune('0'+(count/100)%10)) + string(rune('0'+(count/10)%10)) + string(rune('0'+count%10))
				corpus = append(corpus, mltBuildTonogenesisSet(id, initial, srcTone, aTone, bTone, vowel))
			}
		}
	}
	return corpus
}

func TestMultiLectCrossDimTableLiftsPerPairCommits(t *testing.T) {
	// COMMITMENT: after training on a clean 3-way tonogenesis corpus, the
	// multi-lect model carries a non-empty cross_dimensional_table with lifted
	// entries that name both src_lect and tgt_lect explicitly.
	corpus := mltTonogenesisCorpus()
	m := mustTrainML(t, corpus)

	table := m.CrossDimensionalTable
	if len(table.Entries) == 0 {
		t.Fatal("expected non-empty CrossDimensionalTable.Entries")
	}

	lectSet := map[string]bool{}
	for _, id := range m.LectIDs {
		lectSet[id] = true
	}
	for _, rule := range table.Entries {
		if !lectSet[rule.SrcLect] {
			t.Errorf("rule.SrcLect %q not in LectIDs %v", rule.SrcLect, m.LectIDs)
		}
		if !lectSet[rule.TgtLect] {
			t.Errorf("rule.TgtLect %q not in LectIDs %v", rule.TgtLect, m.LectIDs)
		}
		if rule.SrcLect == rule.TgtLect {
			t.Errorf("rule has SrcLect == TgtLect == %q", rule.SrcLect)
		}
	}
}

func TestMultiLectCrossDimCoversProtoToBothDaughters(t *testing.T) {
	// COMMITMENT: the lift surfaces at least one rule for each directed pair
	// (proto → daughter_a) and (proto → daughter_b) predicting the expected
	// daughter-specific tonogenesis outcome.
	corpus := mltTonogenesisCorpus()
	m := mustTrainML(t, corpus)

	// daughter_a: voiced → tone 4 with confidence >= 0.8
	aHits := 0
	for _, r := range m.CrossDimensionalTable.Entries {
		if r.SrcLect == "proto" && r.TgtLect == "daughter_a" &&
			r.SrcFeature.Feature == "voiced" && r.TgtValue == "4" &&
			r.Confidence >= 0.8 {
			aHits++
		}
	}
	if aHits < 1 {
		t.Errorf("no proto→daughter_a voiced→tone4 rule with confidence>=0.8; got entries: %v",
			m.CrossDimensionalTable.Entries)
	}

	// daughter_b: voiced → tone 3 with confidence >= 0.8
	bHits := 0
	for _, r := range m.CrossDimensionalTable.Entries {
		if r.SrcLect == "proto" && r.TgtLect == "daughter_b" &&
			r.SrcFeature.Feature == "voiced" && r.TgtValue == "3" &&
			r.Confidence >= 0.8 {
			bHits++
		}
	}
	if bHits < 1 {
		t.Errorf("no proto→daughter_b voiced→tone3 rule with confidence>=0.8; got entries: %v",
			m.CrossDimensionalTable.Entries)
	}
}

func TestMultiLectCrossDimSilentOnNonTonalCorpus(t *testing.T) {
	// COMMITMENT: a 3-way non-tonal corpus produces an empty
	// cross_dimensional_table — no spurious lifted entries.
	words := []string{"pata", "kaka", "mama", "sasa", "dudu",
		"titi", "nana", "gogo", "bobo", "pipi"}
	var corpus []CognateSet
	for i, word := range words {
		corpus = append(corpus, csOf(
			"id-"+string(rune('0'+i)),
			[]string{"A", "B", "C"},
			map[string]string{"A": word, "B": word, "C": word},
		))
	}
	m := mustTrainML(t, corpus)
	if len(m.CrossDimensionalTable.Entries) != 0 {
		t.Errorf("expected empty cross-dimensional table on non-tonal corpus, got %d entries",
			len(m.CrossDimensionalTable.Entries))
	}
}

func TestMultiLectCrossDimLiftCrossDimensionalRulesHelper(t *testing.T) {
	// Unit test for liftCrossDimensionalRules on a hand-built pair of models.
	rule := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             20.0,
		SrcCount:          20.0,
		Confidence:        1.0,
	}
	pairModel := EmptyLearnedModel("descriptive", 1.0, 5.0)
	pairModel.CrossDimensionalTable = CrossDimensionalLinkTable{Entries: []CrossDimensionalLink{rule}}

	pairwiseModels := map[string]*LearnedModel{
		lectPairKey("A", "B"): pairModel,
	}
	table := liftCrossDimensionalRules(pairwiseModels, []string{"A", "B"})

	if len(table.Entries) != 1 {
		t.Fatalf("expected 1 lifted entry, got %d", len(table.Entries))
	}
	lifted := table.Entries[0]
	if lifted.SrcLect != "A" {
		t.Errorf("SrcLect = %q, want %q", lifted.SrcLect, "A")
	}
	if lifted.TgtLect != "B" {
		t.Errorf("TgtLect = %q, want %q", lifted.TgtLect, "B")
	}
	if lifted.SrcFeature != rule.SrcFeature {
		t.Errorf("SrcFeature = %v, want %v", lifted.SrcFeature, rule.SrcFeature)
	}
	if lifted.TgtValue != "4" {
		t.Errorf("TgtValue = %q, want %q", lifted.TgtValue, "4")
	}
	if lifted.Count != 20.0 {
		t.Errorf("Count = %v, want 20.0", lifted.Count)
	}
	if lifted.Confidence != 1.0 {
		t.Errorf("Confidence = %v, want 1.0", lifted.Confidence)
	}
}

func TestMultiLectCrossDimLiftCrossDimensionalRulesRespectsLectOrdering(t *testing.T) {
	// COMMITMENT: the lifted src_lect/tgt_lect labels follow the canonical
	// lect_ids ordering (first-seen), not an arbitrary order from the key.
	rule := CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: "voiced", Value: "+"},
		SrcPosition:       "relative_-1",
		TgtDimension:      "tone",
		TgtValue:          "4",
		TgtPositionOffset: 0,
		Count:             10.0,
		SrcCount:          10.0,
		Confidence:        1.0,
	}
	pairModel := EmptyLearnedModel("descriptive", 1.0, 5.0)
	pairModel.CrossDimensionalTable = CrossDimensionalLinkTable{Entries: []CrossDimensionalLink{rule}}

	pairwiseModels := map[string]*LearnedModel{
		lectPairKey("X", "Y"): pairModel,
	}

	// With lect_ids = ("X", "Y"), the lifted rule should be X → Y.
	tableXY := liftCrossDimensionalRules(pairwiseModels, []string{"X", "Y"})
	if len(tableXY.Entries) == 0 {
		t.Fatal("expected at least one entry with lect_ids X,Y")
	}
	if tableXY.Entries[0].SrcLect != "X" {
		t.Errorf("SrcLect = %q, want %q", tableXY.Entries[0].SrcLect, "X")
	}
	if tableXY.Entries[0].TgtLect != "Y" {
		t.Errorf("TgtLect = %q, want %q", tableXY.Entries[0].TgtLect, "Y")
	}

	// With lect_ids = ("Y", "X"), the lifted rule should be Y → X.
	tableYX := liftCrossDimensionalRules(pairwiseModels, []string{"Y", "X"})
	if len(tableYX.Entries) == 0 {
		t.Fatal("expected at least one entry with lect_ids Y,X")
	}
	if tableYX.Entries[0].SrcLect != "Y" {
		t.Errorf("SrcLect = %q, want %q", tableYX.Entries[0].SrcLect, "Y")
	}
	if tableYX.Entries[0].TgtLect != "X" {
		t.Errorf("TgtLect = %q, want %q", tableYX.Entries[0].TgtLect, "X")
	}
}
