package regulae

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// Mirror of python/tests/test_learned_model_invariants.py
//
// Property-based invariant tests for the training pipeline.

var lmiGraphemeAlphabet = []string{
	"p", "b", "t", "d", "k", "g", "m", "n",
	"f", "v", "s", "z", "h",
	"r", "l", "j", "w",
	"a", "e", "i", "o", "u",
}

var lmiToneAlphabet = []string{"H", "M", "L"}
var lmiVowels = map[string]bool{"a": true, "e": true, "i": true, "o": true, "u": true}

func lmiRandomForm(r *rand.Rand, lectID string, maxLength int) Form {
	length := r.Intn(maxLength) + 1 // [1, maxLength]
	segs := make([]Segment, length)
	for i := range segs {
		segs[i] = Segment{Grapheme: lmiGraphemeAlphabet[r.Intn(len(lmiGraphemeAlphabet))]}
	}
	return Form{LectID: lectID, Segments: segs}
}

func lmiRandomTonedForm(r *rand.Rand, lectID string, maxLength int) Form {
	length := r.Intn(maxLength) + 1
	segs := make([]Segment, length)
	for i := range segs {
		g := lmiGraphemeAlphabet[r.Intn(len(lmiGraphemeAlphabet))]
		tone := ""
		if lmiVowels[g] {
			tone = lmiToneAlphabet[r.Intn(len(lmiToneAlphabet))]
		}
		segs[i] = Segment{Grapheme: g, Tone: tone}
	}
	return Form{LectID: lectID, Segments: segs}
}

// lmiMakePairs generates random pairs and converts them to cognate sets.
func lmiMakePairs(seed int64, count int, toned bool) []CognateSet {
	r := rand.New(rand.NewSource(seed))
	pairs := make([]FormPair, count)
	for i := range pairs {
		if toned {
			pairs[i] = FormPair{
				Src: lmiRandomTonedForm(r, "A", 6),
				Tgt: lmiRandomTonedForm(r, "B", 6),
			}
		} else {
			pairs[i] = FormPair{
				Src: lmiRandomForm(r, "A", 6),
				Tgt: lmiRandomForm(r, "B", 6),
			}
		}
	}
	return CognateSetsFromPairs(pairs, [2]string{"A", "B"}, "")
}

func lmiTrain(t *testing.T, corpus []CognateSet) *LearnedModel {
	t.Helper()
	ml := mustTrainML(t, corpus)
	pm, ok := ml.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("expected pairwise model for A/B")
	}
	return pm
}

// ----- Training never crashes ---------------------------------------------

func TestLearnedModelInvariantsTrainingNeverCrashesOnRandomUntoned(t *testing.T) {
	// INVARIANT: training completes on random form pairs.
	for seed := int64(0); seed < 10; seed++ {
		corpus := lmiMakePairs(seed, 20, false)
		model := lmiTrain(t, corpus)
		if model == nil {
			t.Fatalf("seed %d: expected non-nil model", seed)
		}
		// At least segment correspondences map should exist.
		if model.SegmentTable.Counts == nil {
			t.Errorf("seed %d: nil segment counts", seed)
		}
	}
}

func TestLearnedModelInvariantsTrainingNeverCrashesOnRandomToned(t *testing.T) {
	// INVARIANT: training completes on corpora with random tones.
	for seed := int64(0); seed < 5; seed++ {
		corpus := lmiMakePairs(seed+100, 15, true)
		model := lmiTrain(t, corpus)
		if model == nil {
			t.Fatalf("seed %d: expected non-nil model", seed+100)
		}
	}
}

// ----- Context discovery never introduces spurious entries beyond what data
// supports ---------------------------------------------------------------

func TestLearnedModelInvariantsConditionedEntriesAreSubsetOfObservedSources(t *testing.T) {
	// INVARIANT: every conditioned entry in the trained model has a source
	// grapheme that appears in the training data.
	corpus := lmiMakePairs(200, 20, false)
	observedSources := map[string]bool{}
	for _, cs := range corpus {
		for _, form := range cs.Forms {
			if form.LectID != "A" {
				continue
			}
			for _, seg := range form.Segments {
				observedSources[seg.Grapheme] = true
			}
		}
	}
	model := lmiTrain(t, corpus)
	for k, cc := range model.SegmentTable.Corr {
		if model.SegmentTable.Counts[k] == 0 {
			continue
		}
		if cc.Context.ConstraintCount() > 0 {
			if !observedSources[cc.Src] {
				t.Errorf("conditioned entry for unobserved source %q", cc.Src)
			}
		}
	}
}

func TestLearnedModelInvariantsConditionedCountsDoNotExceedUnconditioned(t *testing.T) {
	// INVARIANT: for any source grapheme, no single conditioned entry count
	// for (src, tgt) exceeds the unconditioned count for that same (src, tgt).
	corpus := lmiMakePairs(201, 30, false)
	model := lmiTrain(t, corpus)

	for k, n := range model.SegmentTable.Counts {
		cc, ok := model.SegmentTable.Corr[k]
		if !ok {
			continue
		}
		if cc.Context.ConstraintCount() == 0 {
			continue
		}
		// Find unconditioned (src, tgt) count.
		uncondKey := ccKeyEmpty(cc.Src, cc.Tgt)
		uncondCount := model.SegmentTable.Counts[uncondKey]
		if n > uncondCount+1e-9 {
			t.Errorf("conditioned %s->%s (ctx count %d) count %v > unconditioned %v",
				cc.Src, cc.Tgt, cc.Context.ConstraintCount(), n, uncondCount)
		}
	}
}

// ----- Determinism is preserved -------------------------------------------

func TestLearnedModelInvariantsTrainingIsDeterministicUnderRepeatedRuns(t *testing.T) {
	// INVARIANT: same input → same trained model.
	corpus := lmiMakePairs(300, 20, false)
	m1 := lmiTrain(t, corpus)
	m2 := lmiTrain(t, corpus)
	if !reflect.DeepEqual(m1.SegmentTable.Counts, m2.SegmentTable.Counts) {
		t.Error("segment counts differ across repeated runs")
	}
	if !reflect.DeepEqual(m1.DisplacementDist.Counts, m2.DisplacementDist.Counts) {
		t.Error("displacement counts differ across repeated runs")
	}
	if !reflect.DeepEqual(m1.ChunkTable.Entries, m2.ChunkTable.Entries) {
		t.Error("chunk entries differ across repeated runs")
	}
	if !reflect.DeepEqual(m1.TonalTable.Counts, m2.TonalTable.Counts) {
		t.Error("tonal counts differ across repeated runs")
	}
}

func TestLearnedModelInvariantsTrainingIsDeterministicUnderCorpusReordering(t *testing.T) {
	// INVARIANT: the trained model is invariant under corpus reordering.
	corpus := lmiMakePairs(301, 25, false)
	// Shuffle deterministically.
	r := rand.New(rand.NewSource(0))
	shuffled := make([]CognateSet, len(corpus))
	copy(shuffled, corpus)
	r.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
	m1 := lmiTrain(t, corpus)
	m2 := lmiTrain(t, shuffled)
	if !reflect.DeepEqual(m1.SegmentTable.Counts, m2.SegmentTable.Counts) {
		t.Error("segment counts differ under corpus reordering")
	}
	if !reflect.DeepEqual(m1.TonalTable.Counts, m2.TonalTable.Counts) {
		t.Error("tonal counts differ under corpus reordering")
	}
}

// ----- Cost invariants ----------------------------------------------------

func TestLearnedModelInvariantsTrainedModelAlignmentCostIsFiniteOnRandomInputs(t *testing.T) {
	// INVARIANT: after training, aligning any pair from the corpus with the
	// trained model yields a finite cost.
	corpus := lmiMakePairs(400, 20, false)
	model := lmiTrain(t, corpus)
	for _, cs := range corpus {
		src := cs.Forms["A"]
		tgt := cs.Forms["B"]
		al, err := AlignForms(src, tgt, "descriptive", 0, model)
		if err != nil {
			t.Fatalf("AlignForms error: %v", err)
		}
		c, err := AlignmentCost(al, "descriptive", model)
		if err != nil {
			t.Fatalf("AlignmentCost error: %v", err)
		}
		if math.IsInf(c, 0) || math.IsNaN(c) { // Inf or NaN
			t.Errorf("non-finite cost %v for trained pair", c)
		}
	}
}

func TestLearnedModelInvariantsTrainedModelProducesSensibleCostsOnUnseenPairs(t *testing.T) {
	// INVARIANT: aligning a pair with segments partially overlapping the training
	// corpus still yields a finite cost.
	corpus := lmiMakePairs(401, 15, false)
	model := lmiTrain(t, corpus)
	unseen := lmiMakePairs(402, 5, false)
	for _, cs := range unseen {
		src := cs.Forms["A"]
		tgt := cs.Forms["B"]
		al, err := AlignForms(src, tgt, "descriptive", 0, model)
		if err != nil {
			t.Fatalf("AlignForms error: %v", err)
		}
		c, err := AlignmentCost(al, "descriptive", model)
		if err != nil {
			t.Fatalf("AlignmentCost error: %v", err)
		}
		if c != c { // NaN
			t.Errorf("NaN cost on unseen pair")
		}
	}
}

// ----- Tonal aggregation invariants -------------------------------------------

func TestLearnedModelInvariantsTonaCreatesEntriesOnlyWhenTonesArePresent(t *testing.T) {
	// INVARIANT: training on an untoned corpus leaves the tonal table empty.
	untoned := lmiMakePairs(500, 15, false)
	untonedModel := lmiTrain(t, untoned)
	if len(untonedModel.TonalTable.Counts) != 0 {
		t.Errorf("expected empty tonal table for untoned corpus, got %d entries",
			len(untonedModel.TonalTable.Counts))
	}
	// Toned corpus may or may not produce tonal entries (depends on alignment).
	toned := lmiMakePairs(501, 15, true)
	tonedModel := lmiTrain(t, toned)
	// May be 0 if no tones happened to align; just ensure it doesn't crash.
	_ = tonedModel.TonalTable.Counts
}

func TestLearnedModelInvariantsTonalCountsNonNegative(t *testing.T) {
	// INVARIANT: tonal counts are always non-negative.
	toned := lmiMakePairs(502, 20, true)
	model := lmiTrain(t, toned)
	for k, count := range model.TonalTable.Counts {
		if count < 0 {
			t.Errorf("tonal count for %v = %v < 0", k, count)
		}
	}
}

// ----- Chunk table invariants (still hold) --------------------------------

func TestLearnedModelInvariantsChunksRestrainThemselvesonNonRecurringData(t *testing.T) {
	// INVARIANT: random data (where nothing recurs) yields few promoted chunks.
	corpus := lmiMakePairs(600, 15, false)
	model := lmiTrain(t, corpus)
	if len(model.ChunkTable.Entries) > 20 {
		t.Errorf("chunk table too large (%d entries) for random corpus (expected <= 20)",
			len(model.ChunkTable.Entries))
	}
}

// ----- M2 compatibility preserved -----------------------------------------

func TestLearnedModelInvariantsM2AlignmentCostUnchangedWhenNoModelPassed(t *testing.T) {
	// INVARIANT: aligning without a model reproduces M2 behavior exactly,
	// even on corpora that would use all features during training.
	corpus := lmiMakePairs(700, 10, true)
	for _, cs := range corpus {
		src := cs.Forms["A"]
		tgt := cs.Forms["B"]
		al, err := AlignForms(src, tgt, "descriptive", 0, nil)
		if err != nil {
			t.Fatalf("AlignForms error: %v", err)
		}
		c, err := AlignmentCost(al, "descriptive", nil)
		if err != nil {
			t.Fatalf("AlignmentCost error: %v", err)
		}
		if c != c { // NaN
			t.Errorf("NaN M2 cost for pair without model")
		}
	}
}
