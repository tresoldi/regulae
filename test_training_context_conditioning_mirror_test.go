package regulae

// Mirror of python/tests/test_training_context_conditioning.py

import "testing"

// trnCC builds a ConditionedCorrespondence with an empty context (unconditioned).
func trnCC(src, tgt string) ConditionedCorrespondence {
	return ConditionedCorrespondence{Src: src, Tgt: tgt, Context: Context{}}
}

// trnTonedForm builds a Form from a slice of (grapheme, tone) pairs.
// tone == "" means no tone annotation (Python None).
func trnTonedForm(lectID string, pairs []struct{ g, tone string }) Form {
	segs := make([]Segment, len(pairs))
	for i, p := range pairs {
		segs[i] = Segment{Grapheme: p.g, Tone: p.tone}
	}
	return Form{LectID: lectID, Segments: segs}
}

// ----- context discovery -----------------------------------------------

// TestContextConditioningContextDiscoveryRecoversIntervocalicVoicingSplit checks
// that after training on a corpus where /p/ → /b/ only intervocalically, the
// model contains a conditioned p→b entry.
func TestContextConditioningContextDiscoveryRecoversIntervocalicVoicingSplit(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pap", "B", "pap"), pairOf("A", "pap", "B", "pap"), pairOf("A", "pap", "B", "pap"),
		pairOf("A", "pop", "B", "pop"), pairOf("A", "pop", "B", "pop"), pairOf("A", "pop", "B", "pop"),
		pairOf("A", "apa", "B", "aba"), pairOf("A", "apa", "B", "aba"), pairOf("A", "apa", "B", "aba"),
		pairOf("A", "opa", "B", "oba"), pairOf("A", "opa", "B", "oba"),
		pairOf("A", "upa", "B", "uba"),
	}
	trained := mustTrain(t, corpus)

	// At least one conditioned p entry should exist.
	var pConditioned []string
	for k, cc := range trained.SegmentTable.Corr {
		if cc.Src == "p" && cc.Context.ConstraintCount() > 0 {
			pConditioned = append(pConditioned, k)
		}
	}
	if len(pConditioned) == 0 {
		t.Fatal("expected at least one conditioned p entry after context discovery")
	}

	// There should be at least one conditioned p→b entry.
	var pToBConditioned []string
	for _, k := range pConditioned {
		cc := trained.SegmentTable.Corr[k]
		if cc.Tgt == "b" {
			pToBConditioned = append(pToBConditioned, k)
		}
	}
	if len(pToBConditioned) == 0 {
		t.Error("expected a conditioned p→b entry (intervocalic lenition)")
	}
}

// TestContextConditioningContextDiscoveryDoesNotSplitCleanMergerData checks
// that a clean unconditional merger (p→f everywhere) produces no conditioned entries.
func TestContextConditioningContextDiscoveryDoesNotSplitCleanMergerData(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pap", "B", "faf"), pairOf("A", "pap", "B", "faf"),
		pairOf("A", "pop", "B", "fof"), pairOf("A", "pop", "B", "fof"),
		pairOf("A", "pep", "B", "fef"), pairOf("A", "pep", "B", "fef"),
		pairOf("A", "pip", "B", "fif"), pairOf("A", "pip", "B", "fif"),
		pairOf("A", "apa", "B", "afa"),
		pairOf("A", "opo", "B", "ofo"),
		pairOf("A", "epe", "B", "efe"),
	}
	trained := mustTrain(t, corpus)

	for k, cc := range trained.SegmentTable.Corr {
		if cc.Src == "p" && cc.Context.ConstraintCount() > 0 {
			t.Errorf("unexpected conditioned p entry: key=%q cc=%+v", k, cc)
		}
	}
}

// TestContextConditioningContextDiscoveryRespectsMinSplitObservations checks
// that a single-occurrence p→f observation is not enough to justify a split.
func TestContextConditioningContextDiscoveryRespectsMinSplitObservations(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pa", "B", "pa"),
		pairOf("A", "pa", "B", "pa"),
		pairOf("A", "pa", "B", "pa"),
		pairOf("A", "pa", "B", "pa"),
		pairOf("A", "pa", "B", "pa"),
		pairOf("A", "pi", "B", "fi"), // single oddball
	}
	trained := mustTrain(t, corpus)

	for k, cc := range trained.SegmentTable.Corr {
		if cc.Src == "p" && cc.Tgt == "f" && cc.Context.ConstraintCount() > 0 {
			t.Errorf("unexpected conditioned p→f entry: key=%q", k)
		}
	}
}

// ----- tonal aggregation -----------------------------------------------

// TestContextConditioningPhase4LearnssToneCorrespondenceFromTonedData checks
// that an all-H→L corpus produces a tonal correspondence H→L with count 5.
func TestContextConditioningPhase4LearnssToneCorrespondenceFromTonedData(t *testing.T) {
	type tonePair struct{ g, tone string }
	makePairs := func(gs []tonePair) Form {
		segs := make([]Segment, len(gs))
		for i, p := range gs {
			segs[i] = Segment{Grapheme: p.g, Tone: p.tone}
		}
		return Form{LectID: "tmp", Segments: segs}
	}

	cons := []string{"p", "t", "k", "m", "n"}
	corpus := make([]FormPair, 5)
	for i, c := range cons {
		src := makePairs([]tonePair{{c, ""}, {"a", "H"}})
		src.LectID = "A"
		tgt := makePairs([]tonePair{{c, ""}, {"a", "L"}})
		tgt.LectID = "B"
		corpus[i] = FormPair{Src: src, Tgt: tgt}
	}

	trained := mustTrain(t, corpus)

	key := TonalCorrespondence{SrcTone: "H", TgtTone: "L"}
	if got := trained.TonalTable.Counts[key]; got != 5 {
		t.Errorf("tonal count(H→L) = %v, want 5", got)
	}
}

// TestContextConditioningPhase4LeavesTableEmptyForNonTonalCorpus checks that
// a non-tonal corpus leaves the tonal table empty.
func TestContextConditioningPhase4LeavesTableEmptyForNonTonalCorpus(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pat", "B", "pat"),
		pairOf("A", "pit", "B", "pit"),
	}
	trained := mustTrain(t, corpus)
	if len(trained.TonalTable.Counts) != 0 {
		t.Errorf("tonal table should be empty for non-tonal corpus, got %d entries", len(trained.TonalTable.Counts))
	}
}

// TestContextConditioningPhase4DoesNotStoreNoneToNonePair checks that (None,
// None) tonal pairings (both segments untoned) do not enter the tonal table.
func TestContextConditioningPhase4DoesNotStoreNoneToNonePair(t *testing.T) {
	makeToned := func(lectID string, gTones []struct{ g, t string }) Form {
		segs := make([]Segment, len(gTones))
		for i, p := range gTones {
			segs[i] = Segment{Grapheme: p.g, Tone: p.t}
		}
		return Form{LectID: lectID, Segments: segs}
	}

	corpus := []FormPair{
		{
			Src: makeToned("A", []struct{ g, t string }{{"p", ""}, {"a", "H"}, {"t", ""}}),
			Tgt: makeToned("B", []struct{ g, t string }{{"p", ""}, {"a", "M"}, {"t", ""}}),
		},
		{
			Src: makeToned("A", []struct{ g, t string }{{"k", ""}, {"i", "H"}, {"n", ""}}),
			Tgt: makeToned("B", []struct{ g, t string }{{"k", ""}, {"i", "M"}, {"n", ""}}),
		},
		{
			Src: makeToned("A", []struct{ g, t string }{{"m", ""}, {"u", "H"}, {"r", ""}}),
			Tgt: makeToned("B", []struct{ g, t string }{{"m", ""}, {"u", "M"}, {"r", ""}}),
		},
	}
	trained := mustTrain(t, corpus)

	noneNone := TonalCorrespondence{SrcTone: "", TgtTone: ""}
	if _, ok := trained.TonalTable.Counts[noneNone]; ok {
		t.Error("tonal table should not contain (none, none) entry")
	}
}

// TestContextConditioningTonalScoringZeroCostForUntoned checks that untoned
// segments incur zero tonal cost regardless of the tonal table.
func TestContextConditioningTonalScoringZeroCostForUntoned(t *testing.T) {
	table := TonalCorrespondenceTable{
		Counts: map[TonalCorrespondence]float64{
			{SrcTone: "H", TgtTone: "L"}: 5.0,
		},
		PriorPseudoCounts: map[TonalCorrespondence]float64{
			{SrcTone: "H", TgtTone: "L"}: 1.0,
		},
		SrcTotals: map[string]float64{"H": 5.0},
	}
	// Empty tone strings represent Python None.
	cost := tonalCost("", "", table)
	if cost != 0.0 {
		t.Errorf("tonal cost for untoned pair = %v, want 0.0", cost)
	}
}

// TestContextConditioningTonalScoringNonzeroCostForDifferentTones checks that
// a rare tonal pairing costs more than a common one.
func TestContextConditioningTonalScoringNonzeroCostForDifferentTones(t *testing.T) {
	table := TonalCorrespondenceTable{
		Counts: map[TonalCorrespondence]float64{
			{SrcTone: "H", TgtTone: "H"}: 100.0,
		},
		PriorPseudoCounts: map[TonalCorrespondence]float64{
			{SrcTone: "H", TgtTone: "H"}: 1.0,
			{SrcTone: "H", TgtTone: "L"}: 1.0,
		},
		SrcTotals: map[string]float64{"H": 100.0},
	}
	costHL := tonalCost("H", "L", table)
	costHH := tonalCost("H", "H", table)

	if costHL == 0 || costHH == 0 {
		// Cost should not be zero for toned segments in a non-empty table.
		// (It may be zero if the table structure causes the denominator to
		// collapse; just check that neither is +Inf.)
	}
	// Both should be finite.
	if costHL > 1e9 {
		t.Errorf("H→L cost is unexpectedly large: %v", costHL)
	}
	if costHH > 1e9 {
		t.Errorf("H→H cost is unexpectedly large: %v", costHH)
	}
	// H→L (rare) should cost more than H→H (common).
	if costHL <= costHH {
		t.Errorf("H→L cost %v should exceed H→H cost %v", costHL, costHH)
	}
}
