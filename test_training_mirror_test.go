package regulae

// Mirror of python/tests/test_training.py
//
// SKIPPED (already covered in training_test.go):
//   test_train_model_on_empty_corpus_returns_empty_model → TestTrainEmptyCorpusReturnsEmptyModel
//   test_regular_correspondence_recovery_on_synthetic_corpus → TestRegularCorrespondenceRecovery
//   test_regular_correspondence_makes_alignment_cheaper → TestRegularCorrespondenceMakesAlignmentCheaper
//   test_displacement_layer_generalizes_across_natural_class → TestDisplacementLayerPopulated
//   test_chunk_promotion_for_recurring_cluster → TestChunkPromotionForRecurringCluster
//   test_chunk_promotion_rejects_singletons → TestChunkPromotionRejectsSingletons
//   test_train_model_is_deterministic → TestTrainDeterministic

import "testing"

// trnForm is a local alias for the test-local form builder (segments from IPA string).
func trnForm(lectID, ipa string) Form {
	return formIPA(lectID, ipa)
}

// TestTrainingPriorOnlyModelReproducesM2Alignment checks that prior-only
// scoring gives the same link structure and cost as M2.
func TestTrainingPriorOnlyModelReproducesM2Alignment(t *testing.T) {
	src := trnForm("A", "pater")
	tgt := trnForm("B", "fadar")
	// Use an empty LearnedModel as the initial model stand-in.
	initial := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)

	aM2, err := AlignForms(src, tgt, "descriptive", 0, nil)
	if err != nil {
		t.Fatalf("AlignForms M2: %v", err)
	}
	aPrior, err := AlignForms(src, tgt, "descriptive", 0, initial)
	if err != nil {
		t.Fatalf("AlignForms prior: %v", err)
	}

	if len(aM2.Links) != len(aPrior.Links) {
		t.Fatalf("link count differs: M2=%d prior=%d", len(aM2.Links), len(aPrior.Links))
	}
	for i := range aM2.Links {
		l1 := aM2.Links[i]
		l2 := aPrior.Links[i]
		if !segmentSlicesEqual(l1.SourceChunk, l2.SourceChunk) {
			t.Errorf("link[%d] source chunk differs", i)
		}
		if !segmentSlicesEqual(l1.TargetChunk, l2.TargetChunk) {
			t.Errorf("link[%d] target chunk differs", i)
		}
	}

	costM2, err := AlignmentCost(aM2, "descriptive", nil)
	if err != nil {
		t.Fatalf("AlignmentCost M2: %v", err)
	}
	costPrior, err := AlignmentCost(aPrior, "descriptive", initial)
	if err != nil {
		t.Fatalf("AlignmentCost prior: %v", err)
	}
	if costM2 != costPrior {
		t.Errorf("costs differ: M2=%v prior=%v", costM2, costPrior)
	}
}

// TestTrainingAlignCorpusReturnsOneAlignmentPerPair checks the AlignCorpus API.
func TestTrainingAlignCorpusReturnsOneAlignmentPerPair(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pa", "B", "fa"),
		pairOf("A", "ta", "B", "da"),
		pairOf("A", "ka", "B", "xa"),
	}
	trained := mustTrain(t, corpus)
	alignments, err := AlignCorpus(corpus, trained, 0)
	if err != nil {
		t.Fatalf("AlignCorpus: %v", err)
	}
	if len(alignments) != 3 {
		t.Fatalf("want 3 alignments, got %d", len(alignments))
	}
	for i, fp := range corpus {
		al := alignments[i]
		if !segmentSlicesEqual(al.SourceForm.Segments, fp.Src.Segments) {
			t.Errorf("alignment[%d] source form mismatch", i)
		}
		if !segmentSlicesEqual(al.TargetForm.Segments, fp.Tgt.Segments) {
			t.Errorf("alignment[%d] target form mismatch", i)
		}
	}
}

// TestTrainingAlignCorpusRespectsEmptyCorpus checks that empty corpus returns empty slice.
func TestTrainingAlignCorpusRespectsEmptyCorpus(t *testing.T) {
	trained := mustTrain(t, nil)
	result, err := AlignCorpus(nil, trained, 0)
	if err != nil {
		t.Fatalf("AlignCorpus on empty corpus: %v", err)
	}
	if len(result) != 0 {
		t.Errorf("expected empty result, got %d alignments", len(result))
	}
}

// TestTrainingTrainModelRejectsSingleFormInMultiLectCorpus checks that a
// CognateSet with only 1 form in a multi-lect corpus is rejected.
func TestTrainingTrainModelRejectsSingleFormInMultiLectCorpus(t *testing.T) {
	corpus := []CognateSet{
		{
			CognateID:  "good",
			Forms:      map[string]Form{"A": trnForm("A", "p"), "B": trnForm("B", "f")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		},
		{
			CognateID:  "bad",
			Forms:      map[string]Form{"A": trnForm("A", "k")},
			FormsOrder: []string{"A"},
			Confidence: 1.0,
		},
	}
	_, err := TrainModel(corpus, DefaultTrainOptions())
	if err == nil {
		t.Fatal("expected error for single-form CognateSet in multi-lect corpus")
	}
	// The error should mention "at least 2".
	if errMsg := err.Error(); !containsStr(errMsg, "at least 2") {
		t.Errorf("error message %q does not contain 'at least 2'", errMsg)
	}
}

// TestTrainingTrainModelRejectsEmptyForm checks that a CognateSet with a
// zero-segment form is rejected.
func TestTrainingTrainModelRejectsEmptyForm(t *testing.T) {
	corpus := []CognateSet{
		{
			CognateID: "bad",
			Forms: map[string]Form{
				"A": {LectID: "A", Segments: []Segment{}},
				"B": trnForm("B", "p"),
			},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		},
	}
	_, err := TrainModel(corpus, DefaultTrainOptions())
	if err == nil {
		t.Fatal("expected error for empty-segment form")
	}
	if errMsg := err.Error(); !containsStr(errMsg, "no segments") {
		t.Errorf("error message %q does not contain 'no segments'", errMsg)
	}
}

// containsStr is a simple substring check helper (not already defined elsewhere).
func containsStr(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(sub) == 0 ||
		func() bool {
			for i := 0; i <= len(s)-len(sub); i++ {
				if s[i:i+len(sub)] == sub {
					return true
				}
			}
			return false
		}())
}
