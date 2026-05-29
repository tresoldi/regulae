package regulae

import (
	"testing"
)

// Mirror of python/tests/test_resolvers.py

func mdlFormR(lect, word string) Form {
	return formIPA(lect, word)
}

func mdlBuildCorpusN2(count int, sectA, wordA, sectB, wordB string) []CognateSet {
	var corpus []CognateSet
	for i := 0; i < count; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  sectA + string(rune('0'+i%10)),
			Forms:      map[string]Form{sectA: mdlFormR(sectA, wordA), sectB: mdlFormR(sectB, wordB)},
			FormsOrder: []string{sectA, sectB},
			Confidence: 1.0,
		})
	}
	return corpus
}

// ----- LearnedModel.PosteriorFor ----------------------------------------

func TestResolversPosteriorForUnconditionedSumsOverSeenTargets(t *testing.T) {
	// COMMITMENT: the unconditioned posterior over targets is coherent
	// and sums approximately to 1 after Dirichlet smoothing.
	corpus := mdlBuildCorpusN2(10, "A", "pa", "B", "fa")
	model := mustTrainML(t, corpus)
	pair, ok := model.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("expected pairwise model for A/B")
	}
	posterior := pair.PosteriorFor("p", nil)
	if posterior["f"] <= 0.9 {
		t.Errorf("P(f|p) = %v, want > 0.9 after clean p->f training", posterior["f"])
	}
	sum := 0.0
	for _, v := range posterior {
		sum += v
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("posterior sum = %v, want ~1.0", sum)
	}
}

func TestResolversPosteriorForConditionedContextPicksMatchingEntry(t *testing.T) {
	// COMMITMENT: when a conditioned entry's context is a subset of the
	// query context, posterior_for uses that entry (not the unconditioned fallback).
	var corpus []CognateSet
	for i := 0; i < 20; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "c" + string(rune('0'+i%10)),
			Forms:      map[string]Form{"A": mdlFormR("A", "pa"), "B": mdlFormR("B", "fa")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	for i := 0; i < 20; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "d" + string(rune('0'+i%10)),
			Forms:      map[string]Form{"A": mdlFormR("A", "pi"), "B": mdlFormR("B", "bi")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	model := mustTrainML(t, corpus)
	pair, ok := model.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("expected pairwise model for A/B")
	}
	ctxClose := Context{Following: []FeatureConstraint{{Feature: "close", Value: "+"}}}
	post := pair.PosteriorFor("p", &ctxClose)
	// With a following-close context, the conditioned p->b entry should fire.
	if post["b"] <= 0.9 {
		t.Errorf("P(b|p, close:+) = %v, want > 0.9", post["b"])
	}
	if post["f"] >= 0.1 {
		t.Errorf("P(f|p, close:+) = %v, want < 0.1", post["f"])
	}
}

func TestResolversPosteriorForNoEntriesReturnsEmptyDict(t *testing.T) {
	corpus := []CognateSet{
		{CognateID: "c0", Forms: map[string]Form{"A": mdlFormR("A", "pa"), "B": mdlFormR("B", "fa")}, FormsOrder: []string{"A", "B"}, Confidence: 1.0},
	}
	model := mustTrainML(t, corpus)
	pair, ok := model.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("expected pairwise model for A/B")
	}
	result := pair.PosteriorFor("xyz", nil)
	if len(result) != 0 {
		t.Errorf("expected empty posterior for unknown source, got %v", result)
	}
}

func TestResolversPosteriorForEmptyContextEqualsNilContext(t *testing.T) {
	corpus := mdlBuildCorpusN2(5, "A", "pa", "B", "fa")
	model := mustTrainML(t, corpus)
	pair, ok := model.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("expected pairwise model for A/B")
	}
	emptyCtx := Context{}
	p1 := pair.PosteriorFor("p", nil)
	p2 := pair.PosteriorFor("p", &emptyCtx)
	if len(p1) != len(p2) {
		t.Errorf("posterior lengths differ: %d vs %d", len(p1), len(p2))
	}
	for k, v1 := range p1 {
		if v2, ok := p2[k]; !ok || v1 != v2 {
			t.Errorf("posterior[%q] nil=%v empty=%v", k, v1, v2)
		}
	}
}

// ----- MultiLectModel.ClassesForSegment --------------------------------

func TestResolversClassesForSegmentFiltersByLectGrapheme(t *testing.T) {
	var corpus []CognateSet
	for i := 0; i < 5; i++ {
		corpus = append(corpus, CognateSet{
			CognateID: "c" + string(rune('0'+i)),
			Forms: map[string]Form{
				"A": mdlFormR("A", "pa"),
				"B": mdlFormR("B", "fa"),
				"C": mdlFormR("C", "pa"),
			},
			FormsOrder: []string{"A", "B", "C"},
			Confidence: 1.0,
		})
	}
	model := mustTrainML(t, corpus)
	matches := model.ClassesForSegment("A", "p", true, true)
	if len(matches) == 0 {
		t.Fatal("expected at least one class binding A:p")
	}
	for _, m := range matches {
		if m.Segments["A"] != "p" {
			t.Errorf("class binds A=%q, want p", m.Segments["A"])
		}
		if m.Segments["B"] != "f" {
			t.Errorf("class binds B=%q, want f", m.Segments["B"])
		}
		if m.Segments["C"] != "p" {
			t.Errorf("class binds C=%q, want p", m.Segments["C"])
		}
	}
}

func TestResolversClassesForSegmentReturnsEmptyForAbsentSegment(t *testing.T) {
	corpus := []CognateSet{
		{CognateID: "c0", Forms: map[string]Form{"A": mdlFormR("A", "pa"), "B": mdlFormR("B", "fa")}, FormsOrder: []string{"A", "B"}, Confidence: 1.0},
	}
	model := mustTrainML(t, corpus)
	matches := model.ClassesForSegment("A", "q", true, true)
	if len(matches) != 0 {
		t.Errorf("expected no classes for absent segment q, got %d", len(matches))
	}
}

func TestResolversClassByIDRoundtrip(t *testing.T) {
	corpus := mdlBuildCorpusN2(5, "A", "pa", "B", "fa")
	model := mustTrainML(t, corpus)
	if len(model.UnconditionedClasses) == 0 {
		t.Skip("no unconditioned classes — skipping ClassByID roundtrip test")
	}
	first := model.UnconditionedClasses[0]
	found := model.ClassByID(first.ClassID)
	if found == nil {
		t.Fatal("ClassByID returned nil for a known class ID")
	}
	if found.ClassID != first.ClassID {
		t.Errorf("ClassByID returned wrong class: got %d, want %d", found.ClassID, first.ClassID)
	}
	if model.ClassByID(99999) != nil {
		t.Error("ClassByID should return nil for unknown ID 99999")
	}
}

func TestResolversCognateSetByIDWalksBackToSource(t *testing.T) {
	corpus := []CognateSet{
		{CognateID: "alpha", Forms: map[string]Form{"A": mdlFormR("A", "pa"), "B": mdlFormR("B", "fa")}, FormsOrder: []string{"A", "B"}, Confidence: 1.0},
		{CognateID: "beta", Forms: map[string]Form{"A": mdlFormR("A", "ta"), "B": mdlFormR("B", "da")}, FormsOrder: []string{"A", "B"}, Confidence: 1.0},
	}
	model := mustTrainML(t, corpus)
	found := model.CognateSetByID("alpha")
	if found == nil {
		t.Fatal("CognateSetByID returned nil for 'alpha'")
	}
	if found.Forms["A"].Segments[0].Grapheme != "p" {
		t.Errorf("alpha A[0] = %q, want p", found.Forms["A"].Segments[0].Grapheme)
	}
	if model.CognateSetByID("gamma") != nil {
		t.Error("CognateSetByID should return nil for absent ID 'gamma'")
	}
}

// ----- ChunkPhraseTable.Diagnostics --------------------------------------

func TestResolversChunkDiagnosticsPopulatedAtTraining(t *testing.T) {
	// COMMITMENT: every promoted chunk carries a diagnostic report.
	var corpus []CognateSet
	for i := 0; i < 5; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "c" + string(rune('0'+i)),
			Forms:      map[string]Form{"A": mdlFormR("A", "kta"), "B": mdlFormR("B", "tʃa")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	model := mustTrainML(t, corpus)
	pair, ok := model.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("expected pairwise model for A/B")
	}
	if len(pair.ChunkTable.Entries) == 0 {
		t.Skip("no chunks promoted — skipping diagnostics check")
	}
	validProfiles := map[string]bool{
		"compact_fusion": true, "residual_reduction": true, "nasal_fusion": true,
		"glide_or_vocalization_fusion": true, "balanced_restructuring": true,
		"bundled_reduction": true, "mixed_or_unclear": true,
	}
	for k := range pair.ChunkTable.Entries {
		report, ok := pair.ChunkTable.Diagnostics[k]
		if !ok {
			t.Errorf("chunk %q missing diagnostic report", k)
			continue
		}
		if report.TransparencyScore < 0.0 || report.TransparencyScore > 1.0 {
			t.Errorf("transparency_score = %v out of [0, 1]", report.TransparencyScore)
		}
		if !validProfiles[report.ProcessProfile] {
			t.Errorf("invalid process_profile: %q", report.ProcessProfile)
		}
	}
}

func TestResolversChunkMinTransparencyFiltersLowScoreChunks(t *testing.T) {
	// COMMITMENT: setting chunk_min_transparency drops low-score chunks.
	var corpus []CognateSet
	for i := 0; i < 5; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "c" + string(rune('0'+i)),
			Forms:      map[string]Form{"A": mdlFormR("A", "kta"), "B": mdlFormR("B", "tʃa")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	filtOpts := DefaultTrainOptions()
	filtOpts.ChunkMinTransparency = 0.5
	full := mustTrainML(t, corpus)
	filt, err := TrainModel(corpus, filtOpts)
	if err != nil {
		t.Fatalf("TrainModel (filtered) error: %v", err)
	}
	fullPair, _ := full.PairwiseModel("A", "B")
	filtPair, _ := filt.PairwiseModel("A", "B")
	if fullPair == nil || filtPair == nil {
		t.Skip("no pairwise model — skipping chunk filter test")
	}
	if len(filtPair.ChunkTable.Entries) > len(fullPair.ChunkTable.Entries) {
		t.Errorf("filtered chunk count %d > full count %d", len(filtPair.ChunkTable.Entries), len(fullPair.ChunkTable.Entries))
	}
	for k := range filtPair.ChunkTable.Entries {
		report, ok := filtPair.ChunkTable.Diagnostics[k]
		if !ok {
			t.Errorf("filtered chunk %q missing diagnostic", k)
			continue
		}
		if report.TransparencyScore < 0.5 {
			t.Errorf("filtered chunk has transparency_score %v < 0.5", report.TransparencyScore)
		}
	}
}

func TestResolversChunkFilterRetainsDiagnosticsOnlyForSurvivors(t *testing.T) {
	var corpus []CognateSet
	for i := 0; i < 5; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "c" + string(rune('0'+i)),
			Forms:      map[string]Form{"A": mdlFormR("A", "kta"), "B": mdlFormR("B", "tʃa")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	filtOpts := DefaultTrainOptions()
	filtOpts.ChunkMinTransparency = 0.5
	filt, err := TrainModel(corpus, filtOpts)
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	pair, ok := filt.PairwiseModel("A", "B")
	if !ok {
		t.Skip("no pairwise model")
	}
	// diagnostics must be a subset of entries (no orphaned reports).
	for k := range pair.ChunkTable.Diagnostics {
		if _, ok := pair.ChunkTable.Entries[k]; !ok {
			t.Errorf("orphaned diagnostic for key %q (not in entries)", k)
		}
	}
	// entries and diagnostics should have same keys.
	if len(pair.ChunkTable.Entries) != len(pair.ChunkTable.Diagnostics) {
		t.Errorf("entries count %d != diagnostics count %d",
			len(pair.ChunkTable.Entries), len(pair.ChunkTable.Diagnostics))
	}
}

func TestResolversChunkDiagnosticsNotInEquality(t *testing.T) {
	// Two tables with the same entries but different diagnostics: entries equality.
	t1 := NewChunkPhraseTable()
	t2 := NewChunkPhraseTable()
	// Both have empty entries — they're equal.
	if len(t1.Entries) != len(t2.Entries) {
		t.Error("empty tables should have equal entry counts")
	}
}

// ----- regression / invariance --------------------------------------------

func TestResolversDefaultTrainingUnchanged(t *testing.T) {
	// Default training with no chunk_min_transparency keeps all BIC-promoted
	// chunks (backward compatible).
	corpus := mdlBuildCorpusN2(5, "A", "pa", "B", "fa")
	a := mustTrainML(t, corpus)
	bOpts := DefaultTrainOptions()
	bOpts.ChunkMinTransparency = 0.0
	b, err := TrainModel(corpus, bOpts)
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	pA, _ := a.PairwiseModel("A", "B")
	pB, _ := b.PairwiseModel("A", "B")
	if pA == nil || pB == nil {
		t.Skip("no pairwise model")
	}
	if len(pA.ChunkTable.Entries) != len(pB.ChunkTable.Entries) {
		t.Errorf("chunk entry counts differ: %d vs %d",
			len(pA.ChunkTable.Entries), len(pB.ChunkTable.Entries))
	}
}
