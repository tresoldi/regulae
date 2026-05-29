package regulae

// Mirror of python/tests/test_typological_prior.py
//
// Skipped (already covered by priors_test.go):
//   TestUniformPriorIsZero  (TestUniformPriorIsZero)
//   TestCombinePriorsSum    (TestCombinePriorsSum)
//
// Ported below: all other tests from test_typological_prior.py.

import "testing"

func typFormIPA(lect, word string) Form {
	segs := make([]Segment, len([]rune(word)))
	for i, c := range []rune(word) {
		segs[i] = Segment{Grapheme: string(c)}
	}
	return Form{LectID: lect, Segments: segs}
}

func typMixedCorpus() []CognateSet {
	var corpus []CognateSet
	for i := 0; i < 3; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "c" + string(rune('0'+i)),
			Forms:      map[string]Form{"A": typFormIPA("A", "pa"), "B": typFormIPA("B", "fa")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	for i := 0; i < 3; i++ {
		corpus = append(corpus, CognateSet{
			CognateID:  "d" + string(rune('0'+i)),
			Forms:      map[string]Form{"A": typFormIPA("A", "pe"), "B": typFormIPA("B", "be")},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		})
	}
	return corpus
}

// ----- uniform prior is a no-op -------------------------------------------

func TestTypologicalPriorUniformPriorMatchesNoPrior(t *testing.T) {
	corpus := typMixedCorpus()
	mNone, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel (no prior) error: %v", err)
	}
	optsUni := DefaultTrainOptions()
	optsUni.TypologicalPrior = Uniform()
	mUni, err := TrainModel(corpus, optsUni)
	if err != nil {
		t.Fatalf("TrainModel (uniform) error: %v", err)
	}
	pmNone, _ := mNone.PairwiseModel("A", "B")
	pmUni, _ := mUni.PairwiseModel("A", "B")

	// Prior pseudo counts should be identical.
	if len(pmNone.SegmentTable.PriorPseudoCounts) != len(pmUni.SegmentTable.PriorPseudoCounts) {
		t.Errorf("prior_pseudo_counts size: no-prior=%d uniform=%d",
			len(pmNone.SegmentTable.PriorPseudoCounts), len(pmUni.SegmentTable.PriorPseudoCounts))
	}
	for k, v := range pmNone.SegmentTable.PriorPseudoCounts {
		if pmUni.SegmentTable.PriorPseudoCounts[k] != v {
			t.Errorf("prior[%q] = no-prior:%v uniform:%v", k, v, pmUni.SegmentTable.PriorPseudoCounts[k])
		}
	}
	// Counts should be identical.
	for k, v := range pmNone.SegmentTable.Counts {
		if pmUni.SegmentTable.Counts[k] != v {
			t.Errorf("count[%q] = no-prior:%v uniform:%v", k, v, pmUni.SegmentTable.Counts[k])
		}
	}
}

// ----- custom prior shifts the posterior ----------------------------------

func TestTypologicalPriorCustomPriorBoostsTargetedCorrespondence(t *testing.T) {
	corpus := typMixedCorpus()

	boostPToB := func(s, tg string) float64 {
		if s == "p" && tg == "b" {
			return 3.0
		}
		return 0.0
	}

	mNone, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("no-prior error: %v", err)
	}
	optsBoost := DefaultTrainOptions()
	optsBoost.TypologicalPrior = boostPToB
	mBoost, err := TrainModel(corpus, optsBoost)
	if err != nil {
		t.Fatalf("boost error: %v", err)
	}

	pmNone, _ := mNone.PairwiseModel("A", "B")
	pmBoost, _ := mBoost.PairwiseModel("A", "B")

	postNone := pmNone.PosteriorFor("p", nil)
	postBoost := pmBoost.PosteriorFor("p", nil)

	if postBoost["b"] <= postNone["b"] {
		t.Errorf("P(b|p) with boost (%v) should be > without boost (%v)", postBoost["b"], postNone["b"])
	}
	if postBoost["f"] >= postNone["f"] {
		t.Errorf("P(f|p) with boost (%v) should be < without boost (%v)", postBoost["f"], postNone["f"])
	}
}

func TestTypologicalPriorCustomPriorSuppressesTargetedCorrespondence(t *testing.T) {
	corpus := typMixedCorpus()

	suppressPToB := func(s, tg string) float64 {
		if s == "p" && tg == "b" {
			return -3.0
		}
		return 0.0
	}

	mNone, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("no-prior error: %v", err)
	}
	optsSupp := DefaultTrainOptions()
	optsSupp.TypologicalPrior = suppressPToB
	mSupp, err := TrainModel(corpus, optsSupp)
	if err != nil {
		t.Fatalf("suppress error: %v", err)
	}

	pmNone, _ := mNone.PairwiseModel("A", "B")
	pmSupp, _ := mSupp.PairwiseModel("A", "B")

	postNone := pmNone.PosteriorFor("p", nil)
	postSupp := pmSupp.PosteriorFor("p", nil)

	if postSupp["b"] >= postNone["b"] {
		t.Errorf("P(b|p) suppressed (%v) should be < no-prior (%v)", postSupp["b"], postNone["b"])
	}
}

// ----- composition --------------------------------------------------------

func TestTypologicalPriorCombineSumsPriors(t *testing.T) {
	corpus := typMixedCorpus()

	boost := func(s, tg string) float64 {
		if s == "p" && tg == "b" {
			return 1.5
		}
		return 0.0
	}

	optsSolo := DefaultTrainOptions()
	optsSolo.TypologicalPrior = boost
	mSolo, err := TrainModel(corpus, optsSolo)
	if err != nil {
		t.Fatalf("solo error: %v", err)
	}

	optsCombined := DefaultTrainOptions()
	optsCombined.TypologicalPrior = Combine(Uniform(), boost)
	mCombined, err := TrainModel(corpus, optsCombined)
	if err != nil {
		t.Fatalf("combined error: %v", err)
	}

	pmSolo, _ := mSolo.PairwiseModel("A", "B")
	pmCombined, _ := mCombined.PairwiseModel("A", "B")

	// prior_pseudo_counts should be identical (Uniform adds 0).
	if len(pmSolo.SegmentTable.PriorPseudoCounts) != len(pmCombined.SegmentTable.PriorPseudoCounts) {
		t.Errorf("prior size: solo=%d combined=%d",
			len(pmSolo.SegmentTable.PriorPseudoCounts), len(pmCombined.SegmentTable.PriorPseudoCounts))
	}
	for k, v := range pmSolo.SegmentTable.PriorPseudoCounts {
		vc := pmCombined.SegmentTable.PriorPseudoCounts[k]
		if v != vc {
			t.Errorf("prior[%q] solo=%v combined=%v", k, v, vc)
		}
	}
}

func TestTypologicalPriorCombineTwoCustomPriors(t *testing.T) {
	corpus := typMixedCorpus()

	boostPB := func(s, tg string) float64 {
		if s == "p" && tg == "b" {
			return 1.5
		}
		return 0.0
	}
	boostPF := func(s, tg string) float64 {
		if s == "p" && tg == "f" {
			return 1.5
		}
		return 0.0
	}

	optsCombined := DefaultTrainOptions()
	optsCombined.TypologicalPrior = Combine(boostPB, boostPF)
	mCombined, err := TrainModel(corpus, optsCombined)
	if err != nil {
		t.Fatalf("combined error: %v", err)
	}

	pm, _ := mCombined.PairwiseModel("A", "B")
	pcs := pm.SegmentTable.PriorPseudoCounts

	var pToB, pToF, pToE float64
	for k, c := range pcs {
		cc, ok := pm.SegmentTable.Corr[k]
		if !ok {
			continue
		}
		if cc.Src == "p" && cc.Tgt == "b" {
			pToB = c
		}
		if cc.Src == "p" && cc.Tgt == "f" {
			pToF = c
		}
		if cc.Src == "p" && cc.Tgt == "e" {
			pToE = c
		}
	}
	if pToB <= pToE {
		t.Errorf("p->b prior (%v) should be > p->e prior (%v)", pToB, pToE)
	}
	if pToF <= pToE {
		t.Errorf("p->f prior (%v) should be > p->e prior (%v)", pToF, pToE)
	}
}

// ----- regression: determinism --------------------------------------------

func TestTypologicalPriorTrainingWithPriorIsDeterministic(t *testing.T) {
	corpus := typMixedCorpus()

	boost := func(s, tg string) float64 {
		if s == "p" && tg == "b" {
			return 2.0
		}
		return 0.0
	}

	opts := DefaultTrainOptions()
	opts.TypologicalPrior = boost

	m1, err := TrainModel(corpus, opts)
	if err != nil {
		t.Fatalf("run1 error: %v", err)
	}
	m2, err := TrainModel(corpus, opts)
	if err != nil {
		t.Fatalf("run2 error: %v", err)
	}

	pa1, _ := m1.PairwiseModel("A", "B")
	pa2, _ := m2.PairwiseModel("A", "B")

	for k, v := range pa1.SegmentTable.PriorPseudoCounts {
		if pa2.SegmentTable.PriorPseudoCounts[k] != v {
			t.Errorf("prior[%q] run1=%v run2=%v", k, v, pa2.SegmentTable.PriorPseudoCounts[k])
		}
	}
	for k, v := range pa1.SegmentTable.Counts {
		if pa2.SegmentTable.Counts[k] != v {
			t.Errorf("count[%q] run1=%v run2=%v", k, v, pa2.SegmentTable.Counts[k])
		}
	}
}
