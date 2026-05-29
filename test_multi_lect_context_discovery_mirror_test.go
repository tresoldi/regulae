package regulae

// Mirror of python/tests/test_multi_lect_context_discovery.py

import "testing"

// mltMakePalatalizationCorpusLong builds the palatalization corpus used by most
// context-discovery tests: A's k maps to B's s before front vowels and to k
// elsewhere. Uses letter suffixes in IDs to avoid wrapping past 9 (16 pairs).
func mltMakePalatalizationCorpusLong() []CognateSet {
	palatalized := [][2]string{
		{"kita", "sita"}, {"kite", "site"}, {"ketu", "setu"}, {"keri", "seri"},
		{"kina", "sina"}, {"keta", "seta"}, {"kile", "sile"}, {"kise", "sise"},
	}
	preserved := [][2]string{
		{"kata", "kata"}, {"koto", "koto"}, {"kupa", "kupa"}, {"kala", "kala"},
		{"koma", "koma"}, {"kuma", "kuma"}, {"kota", "kota"}, {"kapa", "kapa"},
	}
	var corpus []CognateSet
	all := append(palatalized, preserved...)
	for i, p := range all {
		id := "c" + string(rune('a'+i))
		corpus = append(corpus, csOf(id, []string{"A", "B"},
			map[string]string{"A": p[0], "B": p[1]}))
	}
	return corpus
}

func mltPalatalizationCorpus() []CognateSet {
	return mltMakePalatalizationCorpusLong()
}

func TestMultiLectContextDiscoveryRunsWithoutErrors(t *testing.T) {
	// SMOKE: multi-lect context discovery runs on a simple corpus and returns
	// a MultiLectModel with possibly non-empty conditioned_classes.
	corpus := mltPalatalizationCorpus()
	m := mustTrainML(t, corpus)
	if len(m.UnconditionedClasses) == 0 {
		t.Error("expected at least one unconditioned class")
	}
}

func TestMultiLectContextDiscoveryDiscoversPalatalizationSplit(t *testing.T) {
	// COMMITMENT: on a corpus where A's k splits into B's s before front
	// vowels and B's k elsewhere, multiLectContextDiscovery should commit a
	// conditioned class for (A:k, B:s) with a non-empty A-context that
	// references a front/vowel feature.
	corpus := mltPalatalizationCorpus()
	m := mustTrainML(t, corpus)

	if len(m.ConditionedClasses) == 0 {
		t.Fatal("expected at least one conditioned class")
	}

	// There must be a conditioned class matching (A:k, B:s) with a non-empty
	// A-side following context.
	var kToS []MultiLectCorrespondenceClass
	for _, klass := range m.ConditionedClasses {
		if klass.Segments["A"] != "k" || klass.Segments["B"] != "s" {
			continue
		}
		if klass.Contexts == nil {
			continue
		}
		aCtx, ok := klass.Contexts["A"]
		if !ok || len(aCtx.Following) == 0 {
			continue
		}
		kToS = append(kToS, klass)
	}
	if len(kToS) == 0 {
		t.Error("expected a conditioned (A:k, B:s) class with a non-empty following-context on A (the palatalization rule)")
	}

	// Its count should match the number of front-vowel palatalized pairs (8).
	maxCount := 0.0
	for _, klass := range kToS {
		if klass.Count > maxCount {
			maxCount = klass.Count
		}
	}
	if maxCount < 8 {
		t.Errorf("max count of (A:k, B:s) conditioned class = %v, want >= 8", maxCount)
	}
}

func TestMultiLectContextDiscoveryPreservesUnconditionedClasses(t *testing.T) {
	// COMMITMENT: multiLectContextDiscovery adds conditioned classes; it does NOT
	// modify the unconditioned table (Option C: parallel tables).
	corpus := mltPalatalizationCorpus()
	m := mustTrainML(t, corpus)

	if len(m.UnconditionedClasses) == 0 {
		t.Fatal("expected at least one unconditioned class")
	}
	for _, klass := range m.UnconditionedClasses {
		if klass.Contexts != nil {
			t.Errorf("unconditioned class has non-nil Contexts: %v", klass.Contexts)
		}
	}
}

func TestMultiLectContextDiscoveryConditionedClassesCarryConfidence(t *testing.T) {
	// COMMITMENT: every conditioned class carries a confidence in [0, 1].
	// The palatalization pivot (A, k) has 16 observations and the committed
	// front-vowel partition has 8, so (A:k, B:s) should have confidence 0.5.
	corpus := mltPalatalizationCorpus()
	m := mustTrainML(t, corpus)

	if len(m.ConditionedClasses) == 0 {
		t.Fatal("expected at least one conditioned class")
	}
	for _, klass := range m.ConditionedClasses {
		if klass.Confidence < 0.0 || klass.Confidence > 1.0 {
			t.Errorf("conditioned class confidence = %v, want in [0, 1]", klass.Confidence)
		}
	}

	// Check that at least one (A:k, B:s) conditioned class has confidence ~0.5.
	found := false
	for _, klass := range m.ConditionedClasses {
		if klass.Segments["A"] == "k" && klass.Segments["B"] == "s" {
			if abs64(klass.Confidence-0.5) < 1e-9 {
				found = true
			}
		}
	}
	if !found {
		t.Error("expected a (A:k, B:s) conditioned class with confidence 0.5")
	}
}

func abs64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func TestMultiLectContextDiscoveryUnconditionedClassesHaveDefaultConfidence(t *testing.T) {
	// COMMITMENT: unconditioned classes leave confidence at 1.0 — confidence is
	// a multiLectContextDiscovery diagnostic only.
	corpus := mltPalatalizationCorpus()
	m := mustTrainML(t, corpus)

	for _, klass := range m.UnconditionedClasses {
		if klass.Confidence != 1.0 {
			t.Errorf("unconditioned class confidence = %v, want 1.0", klass.Confidence)
		}
	}
}

func TestMultiLectContextDiscoveryConditionedClassesHaveUniqueClassIDs(t *testing.T) {
	corpus := mltPalatalizationCorpus()
	m := mustTrainML(t, corpus)

	seen := map[int]bool{}
	for _, klass := range m.UnconditionedClasses {
		if seen[klass.ClassID] {
			t.Errorf("duplicate ClassID %d in unconditioned classes", klass.ClassID)
		}
		seen[klass.ClassID] = true
	}
	for _, klass := range m.ConditionedClasses {
		if seen[klass.ClassID] {
			t.Errorf("duplicate ClassID %d (appears in both tables or duplicated in conditioned)", klass.ClassID)
		}
		seen[klass.ClassID] = true
	}
}

func TestMultiLectContextDiscoveryMinCommitScaleZeroDisablesFloor(t *testing.T) {
	// COMMITMENT: MultiLectMinCommitScale=0 disables the adaptive minimum commit
	// floor and restores the looser behavior (structural minimum of 2).
	corpus := mltPalatalizationCorpus()

	strictOpts := DefaultTrainOptions()
	// strictOpts already has MultiLectMinCommitScale=0.5 by default.
	strictM, err := TrainModel(corpus, strictOpts)
	if err != nil {
		t.Fatalf("TrainModel (strict): %v", err)
	}

	looseOpts := DefaultTrainOptions()
	looseOpts.BICConfig.MultiLectMinCommitScale = 0.0
	looseM, err := TrainModel(corpus, looseOpts)
	if err != nil {
		t.Fatalf("TrainModel (loose): %v", err)
	}

	// Loose should have >= as many conditioned classes as strict.
	if len(looseM.ConditionedClasses) < len(strictM.ConditionedClasses) {
		t.Errorf("loose conditioned classes (%d) < strict (%d)",
			len(looseM.ConditionedClasses), len(strictM.ConditionedClasses))
	}
}

func TestMultiLectContextDiscoveryMinCommitScaleHighKillsSmallCommits(t *testing.T) {
	// COMMITMENT: increasing MultiLectMinCommitScale above default shrinks or
	// eliminates conditioned classes on a small corpus.
	corpus := mltPalatalizationCorpus()

	defaultOpts := DefaultTrainOptions()
	defaultM, err := TrainModel(corpus, defaultOpts)
	if err != nil {
		t.Fatalf("TrainModel (default): %v", err)
	}

	tightOpts := DefaultTrainOptions()
	tightOpts.BICConfig.MultiLectMinCommitScale = 3.0
	tightM, err := TrainModel(corpus, tightOpts)
	if err != nil {
		t.Fatalf("TrainModel (tight): %v", err)
	}

	if len(tightM.ConditionedClasses) > len(defaultM.ConditionedClasses) {
		t.Errorf("tight conditioned classes (%d) > default (%d)",
			len(tightM.ConditionedClasses), len(defaultM.ConditionedClasses))
	}
}

func TestMultiLectContextDiscoveryBICCorrectionCanBeDisabled(t *testing.T) {
	// COMMITMENT: MultiLectBICSmallSampleCorrection=false reverts to the
	// original (looser) BIC penalty and may commit more classes.
	corpus := mltPalatalizationCorpus()

	correctedOpts := DefaultTrainOptions() // default: true
	correctedM, err := TrainModel(corpus, correctedOpts)
	if err != nil {
		t.Fatalf("TrainModel (corrected): %v", err)
	}

	uncorrectedOpts := DefaultTrainOptions()
	uncorrectedOpts.BICConfig.MultiLectBICSmallSampleCorrection = false
	uncorrectedM, err := TrainModel(corpus, uncorrectedOpts)
	if err != nil {
		t.Fatalf("TrainModel (uncorrected): %v", err)
	}

	// Uncorrected should produce >= as many commits.
	if len(uncorrectedM.ConditionedClasses) < len(correctedM.ConditionedClasses) {
		t.Errorf("uncorrected conditioned classes (%d) < corrected (%d)",
			len(uncorrectedM.ConditionedClasses), len(correctedM.ConditionedClasses))
	}
}

func TestMultiLectContextDiscoveryEmptyOnCorpusWithNoSplittableSources(t *testing.T) {
	// COMMITMENT: when no pivot has multiple competing sister tuples,
	// multiLectContextDiscovery commits nothing (no split to find).
	var corpus []CognateSet
	for i := 0; i < 10; i++ {
		corpus = append(corpus, csOf(
			"c"+string(rune('a'+i)),
			[]string{"A", "B"},
			map[string]string{"A": "pata", "B": "pata"},
		))
	}
	m := mustTrainML(t, corpus)
	if len(m.ConditionedClasses) != 0 {
		t.Errorf("expected empty ConditionedClasses on identical-form corpus, got %d",
			len(m.ConditionedClasses))
	}
}
