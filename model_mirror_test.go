package regulae

import "testing"

// Mirror of python/tests/test_model.py

// ----- SegmentCorrespondenceTable ----------------------------------------

func TestModelSegmentCorrespondenceTableDefaultsToEmpty(t *testing.T) {
	// COMMITMENT: a default-constructed table has no counts and no prior.
	table := NewSegmentCorrespondenceTable()
	if len(table.Counts) != 0 {
		t.Errorf("expected empty counts, got %d entries", len(table.Counts))
	}
	if len(table.PriorPseudoCounts) != 0 {
		t.Errorf("expected empty prior_pseudo_counts, got %d entries", len(table.PriorPseudoCounts))
	}
	if len(table.SrcTotals) != 0 {
		t.Errorf("expected empty src_totals, got %d entries", len(table.SrcTotals))
	}
}

func TestModelSegmentCorrespondenceTableHoldsCounts(t *testing.T) {
	table := NewSegmentCorrespondenceTable()
	ccPF := ConditionedCorrespondence{Src: "p", Tgt: "f", Context: Context{}}
	ccPP := ConditionedCorrespondence{Src: "p", Tgt: "p", Context: Context{}}
	table.setCount(ccPF, 5.0)
	table.setCount(ccPP, 1.0)
	table.setPrior(ccPF, 0.5)
	table.setPrior(ccPP, 2.0)
	table.SrcTotals["p"] = 6.0

	if table.Counts[ccPF.key()] != 5.0 {
		t.Errorf("counts[p->f] = %v, want 5.0", table.Counts[ccPF.key()])
	}
	if table.PriorPseudoCounts[ccPP.key()] != 2.0 {
		t.Errorf("prior[p->p] = %v, want 2.0", table.PriorPseudoCounts[ccPP.key()])
	}
	if table.SrcTotals["p"] != 6.0 {
		t.Errorf("src_totals[p] = %v, want 6.0", table.SrcTotals["p"])
	}
}

// ----- DisplacementDistribution ------------------------------------------

func TestModelDisplacementDistributionDefaultsToEmpty(t *testing.T) {
	d := NewDisplacementDistribution()
	if len(d.Counts) != 0 {
		t.Errorf("expected empty counts, got %d entries", len(d.Counts))
	}
	if d.Total != 0.0 {
		t.Errorf("expected Total = 0.0, got %v", d.Total)
	}
	if d.PriorPseudoCount != 1.0 {
		t.Errorf("expected PriorPseudoCount = 1.0, got %v", d.PriorPseudoCount)
	}
}

func TestModelDisplacementDistributionHoldsCountsKeyedByVector(t *testing.T) {
	d := NewDisplacementDistribution()
	d1 := []FeatureDisplacement{{Feature: "continuant", FromValue: "-", ToValue: "+"}}
	d2 := []FeatureDisplacement{{Feature: "voice", FromValue: "-", ToValue: "+"}}
	d.setCount(d1, 8.0)
	d.setCount(d2, 3.0)
	d.Total = 11.0

	k1 := displacementKey(d1)
	if d.Counts[k1] != 8.0 {
		t.Errorf("counts[continuant -/+] = %v, want 8.0", d.Counts[k1])
	}
	if d.Total != 11.0 {
		t.Errorf("total = %v, want 11.0", d.Total)
	}
}

// ----- ChunkPhraseTable --------------------------------------------------

func TestModelChunkPhraseTableDefaultsToEmpty(t *testing.T) {
	table := NewChunkPhraseTable()
	if len(table.Entries) != 0 {
		t.Errorf("expected empty entries, got %d", len(table.Entries))
	}
}

func TestModelChunkPhraseTableHoldsEntries(t *testing.T) {
	table := NewChunkPhraseTable()
	kt := []Segment{{Grapheme: "k"}, {Grapheme: "t"}}
	tt := []Segment{{Grapheme: "t"}, {Grapheme: "t"}}
	table.setEntry(kt, tt, 0.1)
	k := chunkPairKey(kt, tt)
	if table.Entries[k] != 0.1 {
		t.Errorf("entries[(k,t)/(t,t)] = %v, want 0.1", table.Entries[k])
	}
}

// ----- LearnedModel ------------------------------------------------------

func TestModelLearnedModelHoldsAllThreeLayers(t *testing.T) {
	m := EmptyLearnedModel("descriptive", 1.0, 5.0)
	// Just verify the model fields exist and are non-nil maps.
	if m.SegmentTable.Counts == nil {
		t.Error("expected non-nil SegmentTable.Counts")
	}
	if m.DisplacementDist.Counts == nil {
		t.Error("expected non-nil DisplacementDist.Counts")
	}
	if m.ChunkTable.Entries == nil {
		t.Error("expected non-nil ChunkTable.Entries")
	}
}

func TestModelLearnedModelHasDefaultHyperparameters(t *testing.T) {
	// COMMITMENT: default hyperparameters are documented constants.
	m := EmptyLearnedModel("", 1.0, 5.0)
	if m.FeatureSystem != "descriptive" {
		t.Errorf("feature_system = %q, want %q", m.FeatureSystem, "descriptive")
	}
	if m.Temperature != 1.0 {
		t.Errorf("temperature = %v, want 1.0", m.Temperature)
	}
	if m.Concentration != 5.0 {
		t.Errorf("concentration = %v, want 5.0", m.Concentration)
	}
	if m.SegmentWeight != 0.7 {
		t.Errorf("segment_weight = %v, want 0.7", m.SegmentWeight)
	}
	if m.DisplacementWeight != 0.3 {
		t.Errorf("displacement_weight = %v, want 0.3", m.DisplacementWeight)
	}
}

func TestModelLearnedModelEmptyFactory(t *testing.T) {
	// COMMITMENT: empty() creates a model with no learned data.
	m := EmptyLearnedModel("", 1.0, 5.0)
	if len(m.SegmentTable.Counts) != 0 {
		t.Error("expected empty segment counts")
	}
	if len(m.SegmentTable.PriorPseudoCounts) != 0 {
		t.Error("expected empty prior pseudo counts")
	}
	if len(m.DisplacementDist.Counts) != 0 {
		t.Error("expected empty displacement counts")
	}
	if len(m.ChunkTable.Entries) != 0 {
		t.Error("expected empty chunk entries")
	}
}

func TestModelLearnedModelEmptyFactoryPreservesHyperparameters(t *testing.T) {
	m := EmptyLearnedModel("distinctive", 2.0, 10.0)
	if m.FeatureSystem != "distinctive" {
		t.Errorf("feature_system = %q, want %q", m.FeatureSystem, "distinctive")
	}
	if m.Temperature != 2.0 {
		t.Errorf("temperature = %v, want 2.0", m.Temperature)
	}
	if m.Concentration != 10.0 {
		t.Errorf("concentration = %v, want 10.0", m.Concentration)
	}
}

// (skipped: Python-internal test_learned_model_is_frozen — Go structs are not
// frozen; immutability is a convention and cannot be tested the same way.)

// ----- ConditionedCorrespondence -----------------------------------------

func TestModelConditionedCorrespondenceDefaultIsEmptyContext(t *testing.T) {
	// COMMITMENT: a conditioned correspondence without an explicit context uses
	// the empty context.
	cc := ConditionedCorrespondence{Src: "p", Tgt: "f"}
	if cc.Context.ConstraintCount() != 0 {
		t.Error("expected empty context")
	}
}

func TestModelConditionedCorrespondenceIsHashableForDictKeys(t *testing.T) {
	// COMMITMENT: ConditionedCorrespondence can be used as a dict key via key().
	cc := ConditionedCorrespondence{Src: "p", Tgt: "f"}
	d := map[string]float64{}
	d[cc.key()] = 1.0
	if d[cc.key()] != 1.0 {
		t.Error("key lookup failed")
	}
}

func TestModelConditionedCorrespondenceDistinctByContext(t *testing.T) {
	// COMMITMENT: same (src, tgt) with different contexts are distinct keys.
	k1 := ConditionedCorrespondence{Src: "p", Tgt: "f", Context: Context{}}
	k2 := ConditionedCorrespondence{
		Src: "p", Tgt: "f",
		Context: Context{Following: []FeatureConstraint{{Feature: "vowel", Value: "+"}}},
	}
	if k1.key() == k2.key() {
		t.Error("contexts differ but keys are equal")
	}
}

// ----- TonalCorrespondence and TonalCorrespondenceTable ------------------

func TestModelTonalCorrespondenceConstruction(t *testing.T) {
	tc := TonalCorrespondence{SrcTone: "high", TgtTone: "low"}
	if tc.SrcTone != "high" {
		t.Errorf("SrcTone = %q, want %q", tc.SrcTone, "high")
	}
	if tc.TgtTone != "low" {
		t.Errorf("TgtTone = %q, want %q", tc.TgtTone, "low")
	}
}

func TestModelTonalCorrespondenceEmptyStringRepresentsUntoned(t *testing.T) {
	// COMMITMENT: empty string tones represent untoned segments (Python's None).
	tc := TonalCorrespondence{SrcTone: "", TgtTone: ""}
	if tc.SrcTone != "" {
		t.Errorf("expected empty SrcTone, got %q", tc.SrcTone)
	}
	if tc.TgtTone != "" {
		t.Errorf("expected empty TgtTone, got %q", tc.TgtTone)
	}
}

func TestModelTonalCorrespondenceIsHashable(t *testing.T) {
	key := TonalCorrespondence{SrcTone: "55", TgtTone: "33"}
	d := map[TonalCorrespondence]float64{}
	d[key] = 5.0
	if d[key] != 5.0 {
		t.Error("tonal key lookup failed")
	}
}

func TestModelTonalCorrespondenceTableDefaultsToEmpty(t *testing.T) {
	table := NewTonalCorrespondenceTable()
	if len(table.Counts) != 0 {
		t.Errorf("expected empty counts, got %d", len(table.Counts))
	}
	if len(table.PriorPseudoCounts) != 0 {
		t.Errorf("expected empty prior pseudo counts, got %d", len(table.PriorPseudoCounts))
	}
	if len(table.SrcTotals) != 0 {
		t.Errorf("expected empty src_totals, got %d", len(table.SrcTotals))
	}
}

func TestModelTonalCorrespondenceTableHoldsCounts(t *testing.T) {
	key := TonalCorrespondence{SrcTone: "H", TgtTone: "L"}
	table := NewTonalCorrespondenceTable()
	table.Counts[key] = 12.0
	table.PriorPseudoCounts[key] = 1.0
	table.SrcTotals["H"] = 12.0
	if table.Counts[key] != 12.0 {
		t.Errorf("counts[H->L] = %v, want 12.0", table.Counts[key])
	}
}

// ----- LearnedModel with tonal table -------------------------------------

func TestModelLearnedModelHasTonalTableByDefault(t *testing.T) {
	// COMMITMENT: the learned model exposes an (empty) tonal table by default.
	m := EmptyLearnedModel("", 1.0, 5.0)
	if m.TonalTable.Counts == nil {
		t.Error("expected non-nil tonal table counts")
	}
	if len(m.TonalTable.Counts) != 0 {
		t.Errorf("expected empty tonal counts, got %d entries", len(m.TonalTable.Counts))
	}
}

func TestModelLearnedModelHasDefaultToneWeight(t *testing.T) {
	m := EmptyLearnedModel("", 1.0, 5.0)
	if m.ToneWeight != 1.0 {
		t.Errorf("tone_weight = %v, want 1.0", m.ToneWeight)
	}
}
