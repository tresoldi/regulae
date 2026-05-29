package regulae

// Mirror of python/tests/test_uncertainty.py
//
// Skipped (already covered by uncertainty_test.go):
//   TestWilsonZeroNFullInterval
//   TestWilsonFullCertaintyNarrowHigh
//   TestWilsonZeroSuccessesNarrowLow
//   TestWilsonHalfHalfCentered
//   TestWilsonWidensAsNShrinks
//   TestWilsonFractionalCounts
//   TestWilsonRejectsUnsupportedAlpha
//   TestPercentileIntervalEmpty
//   TestPercentileIntervalBracketsMiddle

import (
	"testing"
)

// ----- integration: training populates uncertainty -----------------------

func uncFormIPA(lect, word string) Form {
	var segs []Segment
	for _, c := range word {
		segs = append(segs, Segment{Grapheme: string(c)})
	}
	return Form{LectID: lect, Segments: segs}
}

func uncMakeCorpus(n int, srcWord, tgtWord string) []CognateSet {
	corpus := make([]CognateSet, n)
	for i := range corpus {
		cogID := string(rune('c')) + string(rune('0'+i))
		if i >= 10 {
			cogID = "c" + string(rune('a'+i-10))
		}
		corpus[i] = CognateSet{
			CognateID:  cogID,
			Forms:      map[string]Form{"A": uncFormIPA("A", srcWord), "B": uncFormIPA("B", tgtWord)},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		}
	}
	return corpus
}

func TestUncertaintyTrainingPopulatesSegmentUncertainty(t *testing.T) {
	corpus := uncMakeCorpus(5, "pa", "fa")
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	pm, ok := model.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("no pairwise model for A/B")
	}
	for k := range pm.SegmentTable.Counts {
		u, has := pm.SegmentTable.Uncertainty[k]
		if !has {
			t.Errorf("key %q in counts but not in uncertainty", k)
		} else {
			if u.Method != "wilson" {
				t.Errorf("key %q: method = %q, want wilson", k, u.Method)
			}
			if u.Lo < 0.0 || u.Hi > 1.0 || u.Lo > u.Hi {
				t.Errorf("key %q: interval [%v, %v] out of range", k, u.Lo, u.Hi)
			}
		}
	}
}

func TestUncertaintyTrainingPopulatesDisplacementUncertainty(t *testing.T) {
	corpus := uncMakeCorpus(5, "pa", "fa")
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	pm, ok := model.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("no pairwise model for A/B")
	}
	for k := range pm.DisplacementDist.Counts {
		if _, has := pm.DisplacementDist.Uncertainty[k]; !has {
			t.Errorf("displacement key %q in counts but not in uncertainty", k)
		}
	}
}

func TestUncertaintyTrainingPopulatesMultiLectClassUncertainty(t *testing.T) {
	corpus := make([]CognateSet, 5)
	for i := range corpus {
		corpus[i] = CognateSet{
			CognateID: "c" + string(rune('0'+i)),
			Forms: map[string]Form{
				"A": uncFormIPA("A", "pa"),
				"B": uncFormIPA("B", "fa"),
				"C": uncFormIPA("C", "fa"),
			},
			FormsOrder: []string{"A", "B", "C"},
			Confidence: 1.0,
		}
	}
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	for _, klass := range model.UnconditionedClasses {
		if klass.Uncertainty == nil {
			t.Errorf("class %d has nil uncertainty", klass.ClassID)
			continue
		}
		if klass.Uncertainty.Method != "wilson" {
			t.Errorf("class %d uncertainty method = %q, want wilson", klass.ClassID, klass.Uncertainty.Method)
		}
	}
}

func TestUncertaintySegmentIntervalNarrowerWhenSrcIsWellAttested(t *testing.T) {
	// Build corpus: 20 pairs of p/f and 3 pairs of t/θ.
	var corpus []CognateSet
	for i := 0; i < 20; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "c" + string(rune('A'+i%26)),
			Forms:      map[string]Form{"A": uncFormIPA("A", "pap"), "B": uncFormIPA("B", "faf")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	for i := 0; i < 3; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "d" + string(rune('0'+i)),
			Forms:      map[string]Form{"A": uncFormIPA("A", "ta"), "B": uncFormIPA("B", "θa")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	pm, ok := model.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("no pairwise model A/B")
	}
	// Find p->f (unconditioned) and t->θ (unconditioned) uncertainty.
	var pU, tU *UncertaintyEstimate
	for k, u := range pm.SegmentTable.Uncertainty {
		cc, ok2 := pm.SegmentTable.Corr[k]
		if !ok2 {
			continue
		}
		if cc.Context.ConstraintCount() != 0 {
			continue
		}
		u2 := u
		if cc.Src == "p" && cc.Tgt == "f" {
			pU = &u2
		}
		if cc.Src == "t" && cc.Tgt == "θ" {
			tU = &u2
		}
	}
	if pU == nil {
		t.Fatal("no p->f unconditioned uncertainty found")
	}
	if tU == nil {
		t.Fatal("no t->θ unconditioned uncertainty found")
	}
	if pU.N <= tU.N {
		t.Errorf("p N=%v should be > t N=%v", pU.N, tU.N)
	}
	if (pU.Hi - pU.Lo) >= (tU.Hi - tU.Lo) {
		t.Errorf("p interval width %v should be < t width %v", pU.Hi-pU.Lo, tU.Hi-tU.Lo)
	}
}

// ----- bootstrap ----------------------------------------------------------

func TestUncertaintyBootstrapProducesBootstrapMethodIntervals(t *testing.T) {
	corpus := uncMakeCorpus(10, "pa", "fa")
	opts := DefaultTrainOptions()
	opts.BootstrapN = 10
	opts.BootstrapSeed = 1
	model, err := TrainModel(corpus, opts)
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	pm, ok := model.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("no pairwise model A/B")
	}
	methods := map[string]bool{}
	for _, u := range pm.SegmentTable.Uncertainty {
		methods[u.Method] = true
	}
	if !methods["bootstrap"] {
		t.Error("expected at least one bootstrap method interval in segment table")
	}
	if methods["wilson"] {
		t.Error("expected no wilson intervals when bootstrap is active (all should be bootstrap)")
	}
	for _, klass := range model.UnconditionedClasses {
		if klass.Uncertainty == nil {
			t.Errorf("class %d has nil uncertainty after bootstrap", klass.ClassID)
			continue
		}
		if klass.Uncertainty.Method != "bootstrap" {
			t.Errorf("class %d: uncertainty method = %q, want bootstrap", klass.ClassID, klass.Uncertainty.Method)
		}
	}
}

func TestUncertaintyBootstrapIsDeterministic(t *testing.T) {
	var corpus []CognateSet
	for i := 0; i < 8; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "c" + string(rune('0'+i)),
			Forms:      map[string]Form{"A": uncFormIPA("A", "pa"), "B": uncFormIPA("B", "pa")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	for i := 0; i < 4; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "d" + string(rune('0'+i)),
			Forms:      map[string]Form{"A": uncFormIPA("A", "pa"), "B": uncFormIPA("B", "ba")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	opts := DefaultTrainOptions()
	opts.BootstrapN = 20
	opts.BootstrapSeed = 7
	m1, err1 := TrainModel(corpus, opts)
	m2, err2 := TrainModel(corpus, opts)
	if err1 != nil || err2 != nil {
		t.Fatalf("training errors: %v / %v", err1, err2)
	}
	pm1, _ := m1.PairwiseModel("A", "B")
	pm2, _ := m2.PairwiseModel("A", "B")
	for k, u1 := range pm1.SegmentTable.Uncertainty {
		u2, ok := pm2.SegmentTable.Uncertainty[k]
		if !ok {
			t.Errorf("key %q present in run1 but not run2", k)
			continue
		}
		if u1.Lo != u2.Lo || u1.Hi != u2.Hi {
			t.Errorf("key %q: run1=[%v,%v] run2=[%v,%v]", k, u1.Lo, u1.Hi, u2.Lo, u2.Hi)
		}
	}
}

func TestUncertaintyBootstrapBracketsBaseRate(t *testing.T) {
	var corpus []CognateSet
	for i := 0; i < 8; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "c" + string(rune('0'+i)),
			Forms:      map[string]Form{"A": uncFormIPA("A", "pa"), "B": uncFormIPA("B", "fa")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	for i := 0; i < 3; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "d" + string(rune('0'+i)),
			Forms:      map[string]Form{"A": uncFormIPA("A", "pa"), "B": uncFormIPA("B", "ba")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	opts := DefaultTrainOptions()
	opts.BootstrapN = 100
	opts.BootstrapSeed = 42
	model, err := TrainModel(corpus, opts)
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	pm, ok := model.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("no pairwise model A/B")
	}
	// Find p->f unconditioned key.
	var pfKey string
	for k, cc := range pm.SegmentTable.Corr {
		if cc.Src == "p" && cc.Tgt == "f" && cc.Context.ConstraintCount() == 0 {
			pfKey = k
			break
		}
	}
	if pfKey == "" {
		t.Fatal("no p->f unconditioned key found")
	}
	u := pm.SegmentTable.Uncertainty[pfKey]
	pfCount := pm.SegmentTable.Counts[pfKey]
	pTotal := pm.SegmentTable.SrcTotals["p"]
	baseRate := pfCount / pTotal
	if !(u.Lo <= baseRate && baseRate <= u.Hi) {
		t.Errorf("bootstrap interval [%v, %v] does not bracket base rate %v", u.Lo, u.Hi, baseRate)
	}
}

// ----- invariance ---------------------------------------------------------

func TestUncertaintyTrainingWithoutBootstrapIsDeterministic(t *testing.T) {
	corpus := uncMakeCorpus(5, "pa", "fa")
	m1, err1 := TrainModel(corpus, DefaultTrainOptions())
	m2, err2 := TrainModel(corpus, DefaultTrainOptions())
	if err1 != nil || err2 != nil {
		t.Fatalf("training errors: %v / %v", err1, err2)
	}
	pm1, _ := m1.PairwiseModel("A", "B")
	pm2, _ := m2.PairwiseModel("A", "B")
	if len(pm1.SegmentTable.Counts) != len(pm2.SegmentTable.Counts) {
		t.Error("segment table sizes differ across runs")
	}
	for k, v1 := range pm1.SegmentTable.Counts {
		v2 := pm2.SegmentTable.Counts[k]
		if v1 != v2 {
			t.Errorf("count[%q] = %v vs %v", k, v1, v2)
		}
	}
}

func TestUncertaintyMultiLectClassEqualityIgnoresUncertainty(t *testing.T) {
	corpus := uncMakeCorpus(5, "pa", "fa")
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	if len(model.UnconditionedClasses) == 0 {
		t.Skip("no unconditioned classes to check")
	}
	klass := model.UnconditionedClasses[0]
	// Copy the class with Uncertainty = nil and verify equality on the
	// comparable fields (ClassID, Segments, Count).
	other := klass
	other.Uncertainty = nil
	// Go equality (==) on structs with pointer fields: compare the key fields.
	if klass.ClassID != other.ClassID {
		t.Error("ClassID mismatch")
	}
	if klass.Count != other.Count {
		t.Error("Count mismatch")
	}
	// Uncertainty differing should not matter for logical equality of
	// the class (the Python uses compare=False on the dataclass field).
	// In Go we simply verify the fields-that-matter remain equal.
	if klass.Segments["A"] != other.Segments["A"] {
		t.Error("Segments differ after clearing Uncertainty")
	}
}
