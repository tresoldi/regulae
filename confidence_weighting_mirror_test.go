package regulae

// Mirror of python/tests/test_confidence_weighting.py

import "testing"

func confFormIPA(lect, word string) Form {
	segs := make([]Segment, len([]rune(word)))
	for i, c := range []rune(word) {
		segs[i] = Segment{Grapheme: string(c)}
	}
	return Form{LectID: lect, Segments: segs}
}

func TestConfidenceWeightingPairwiseCountsScaleWithConfidence(t *testing.T) {
	corpus := []CognateSet{
		{
			CognateID:  "c1",
			Forms:      map[string]Form{"A": confFormIPA("A", "pa"), "B": confFormIPA("B", "fa")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		},
		{
			CognateID:  "c2",
			Forms:      map[string]Form{"A": confFormIPA("A", "pa"), "B": confFormIPA("B", "fa")},
			FormsOrder: []string{"A", "B"},
			Confidence: 0.25,
		},
	}
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	pm, ok := model.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("no pairwise model for A/B")
	}
	// Sum counts for p->f and a->a unconditioned.
	var pToF, aToA float64
	for k, count := range pm.SegmentTable.Counts {
		cc, has := pm.SegmentTable.Corr[k]
		if !has || cc.Context.ConstraintCount() != 0 {
			continue
		}
		if cc.Src == "p" && cc.Tgt == "f" {
			pToF += count
		}
		if cc.Src == "a" && cc.Tgt == "a" {
			aToA += count
		}
	}
	if pToF != 1.25 {
		t.Errorf("p->f count = %v, want 1.25", pToF)
	}
	if aToA != 1.25 {
		t.Errorf("a->a count = %v, want 1.25", aToA)
	}
}

func TestConfidenceWeightingMultilectUnconditionedCountsScaleWithConfidence(t *testing.T) {
	corpus := []CognateSet{
		{
			CognateID:  "c1",
			Forms:      map[string]Form{"A": confFormIPA("A", "ka"), "B": confFormIPA("B", "ta"), "C": confFormIPA("C", "ka")},
			FormsOrder: []string{"A", "B", "C"},
			Confidence: 1.0,
		},
		{
			CognateID:  "c2",
			Forms:      map[string]Form{"A": confFormIPA("A", "ka"), "B": confFormIPA("B", "ta"), "C": confFormIPA("C", "ka")},
			FormsOrder: []string{"A", "B", "C"},
			Confidence: 0.5,
		},
	}
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	// Build a key from sorted (lect, grapheme) pairs -> count.
	type segKey struct{ a, b, c string }
	counts := map[segKey]float64{}
	for _, klass := range model.UnconditionedClasses {
		k := segKey{klass.Segments["A"], klass.Segments["B"], klass.Segments["C"]}
		counts[k] = klass.Count
	}
	ktk := segKey{"k", "t", "k"}
	aaa := segKey{"a", "a", "a"}
	if counts[ktk] != 1.5 {
		t.Errorf("A:k/B:t/C:k count = %v, want 1.5", counts[ktk])
	}
	if counts[aaa] != 1.5 {
		t.Errorf("A:a/B:a/C:a count = %v, want 1.5", counts[aaa])
	}
}

func TestConfidenceWeightingZeroConfidenceSuppressesFalseMergerClass(t *testing.T) {
	// Build the clean corpus.
	var clean []CognateSet
	for i := 0; i < 6; i++ {
		clean = append(clean, CognateSet{
			CognateID:  "good_k." + string(rune('0'+i)),
			Forms:      map[string]Form{"M": confFormIPA("M", "ʔa"), "P": confFormIPA("P", "ka"), "C": confFormIPA("C", "ka")},
			FormsOrder: []string{"M", "P", "C"},
			Confidence: 1.0,
		})
	}
	for i := 0; i < 6; i++ {
		clean = append(clean, CognateSet{
			CognateID:  "good_glot." + string(rune('0'+i)),
			Forms:      map[string]Form{"M": confFormIPA("M", "ʔa"), "P": confFormIPA("P", "ʔa"), "C": confFormIPA("C", "ta")},
			FormsOrder: []string{"M", "P", "C"},
			Confidence: 1.0,
		})
	}

	contaminated := append(append([]CognateSet{}, clean...), func() []CognateSet {
		var out []CognateSet
		for i := 0; i < 4; i++ {
			out = append(out, CognateSet{
				CognateID:  "bad." + string(rune('0'+i)),
				Forms:      map[string]Form{"M": confFormIPA("M", "ʔa"), "P": confFormIPA("P", "ka"), "C": confFormIPA("C", "ta")},
				FormsOrder: []string{"M", "P", "C"},
				Confidence: 1.0,
			})
		}
		return out
	}()...)

	downweighted := append(append([]CognateSet{}, clean...), func() []CognateSet {
		var out []CognateSet
		for i := 0; i < 4; i++ {
			out = append(out, CognateSet{
				CognateID:  "bad0." + string(rune('0'+i)),
				Forms:      map[string]Form{"M": confFormIPA("M", "ʔa"), "P": confFormIPA("P", "ka"), "C": confFormIPA("C", "ta")},
				FormsOrder: []string{"M", "P", "C"},
				Confidence: 0.0,
			})
		}
		return out
	}()...)

	contamModel, err := TrainModel(contaminated, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("contaminated TrainModel error: %v", err)
	}
	downModel, err := TrainModel(downweighted, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("downweighted TrainModel error: %v", err)
	}

	// Find counts by segment-set for both models.
	type classKey struct{ M, P, C string }
	contamCounts := map[classKey]float64{}
	for _, klass := range contamModel.UnconditionedClasses {
		k := classKey{klass.Segments["M"], klass.Segments["P"], klass.Segments["C"]}
		contamCounts[k] = klass.Count
	}
	downCounts := map[classKey]float64{}
	for _, klass := range downModel.UnconditionedClasses {
		k := classKey{klass.Segments["M"], klass.Segments["P"], klass.Segments["C"]}
		downCounts[k] = klass.Count
	}

	// The bogus partial class in Python is keyed by (M:ʔ, P:k) — a 2-lect
	// partial class (C is absent from that key). We check for any class
	// that has M:ʔ and P:k but no C entry.
	// Actually in the multi-lect reconciliation, partial observations
	// (not all lects participate) are stored as partial classes.
	// We look for the class with M=ʔ, P=k, C=<any not-t and not-k, i.e. empty or mixed>.
	// Python's bogus_partial = (("M","ʔ"), ("P","k")) — 2-way class.
	// Go equivalent: a class with exactly M=ʔ and P=k (and no C segment).
	// We locate these by scanning for classes where C segment is absent.
	var contamBogusMPCount, downBogusMPCount float64
	for _, klass := range contamModel.UnconditionedClasses {
		if klass.Segments["M"] == "ʔ" && klass.Segments["P"] == "k" && klass.Segments["C"] == "" {
			contamBogusMPCount += klass.Count
		}
	}
	for _, klass := range downModel.UnconditionedClasses {
		if klass.Segments["M"] == "ʔ" && klass.Segments["P"] == "k" && klass.Segments["C"] == "" {
			downBogusMPCount += klass.Count
		}
	}

	if contamBogusMPCount != 4.0 {
		t.Errorf("contaminated bogus partial (M:ʔ, P:k) count = %v, want 4.0", contamBogusMPCount)
	}
	if downBogusMPCount != 0.0 {
		t.Errorf("downweighted bogus partial (M:ʔ, P:k) count = %v, want 0.0", downBogusMPCount)
	}

	// Clean class (M:ʔ, P:k, C:k) should be 6.0 in both.
	cleanK := classKey{"ʔ", "k", "k"}
	if contamCounts[cleanK] != 6.0 {
		t.Errorf("contaminated clean-k count = %v, want 6.0", contamCounts[cleanK])
	}
	if downCounts[cleanK] != 6.0 {
		t.Errorf("downweighted clean-k count = %v, want 6.0", downCounts[cleanK])
	}
}
