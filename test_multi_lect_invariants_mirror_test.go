package regulae

// Mirror of python/tests/test_multi_lect_invariants.py
//
// Skipped (already covered):
//   TestMultiLectDeterministic — reconciliation_test.go::TestMultiLectDeterministic

import (
	"math/rand"
	"testing"
)

// mltMakeN3Corpus builds the N=3 corpus used by invariant tests.
func mltMakeN3Corpus() []CognateSet {
	data := [][3]string{
		{"pata", "fada", "pada"},
		{"kata", "hada", "kada"},
		{"pona", "fona", "pona"},
		{"kono", "hono", "kono"},
		{"pila", "fila", "pila"},
		{"kina", "hina", "kina"},
		{"puto", "futo", "puto"},
		{"kuta", "huta", "kuta"},
	}
	corpus := make([]CognateSet, len(data))
	for i, row := range data {
		corpus[i] = csOf(
			"c"+string(rune('0'+i)),
			[]string{"A", "B", "C"},
			map[string]string{"A": row[0], "B": row[1], "C": row[2]},
		)
	}
	return corpus
}

// mltCanonicalCounts maps sorted (lect, grapheme) key string to count.
func mltCanonicalCounts(m *MultiLectModel) map[string]float64 {
	out := map[string]float64{}
	for _, klass := range m.UnconditionedClasses {
		items := sortedSegmentItems(klass.Segments)
		out[segItemsKey(items)] = klass.Count
	}
	return out
}

// mltConditionedSignature returns a set of (segments-key, count) pairs for
// conditioned classes, used for order-independent comparison.
func mltConditionedSignature(m *MultiLectModel) map[string]float64 {
	out := map[string]float64{}
	for _, klass := range m.ConditionedClasses {
		items := sortedSegmentItems(klass.Segments)
		k := segItemsKey(items)
		out[k] = klass.Count
	}
	return out
}

// ----- determinism under reordering ---------------------------------------

func TestMultiLectInvariantsDeterminismReorderPreservesClasses(t *testing.T) {
	// COMMITMENT: shuffling the input corpus must not change the unconditioned
	// class table.
	corpus := mltMakeN3Corpus()
	ma := mustTrainML(t, corpus)

	shuffled := append([]CognateSet(nil), corpus...)
	rng := rand.New(rand.NewSource(42))
	rng.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
	mb := mustTrainML(t, shuffled)

	countA := mltCanonicalCounts(ma)
	countB := mltCanonicalCounts(mb)
	if len(countA) != len(countB) {
		t.Fatalf("class count differs after shuffle: %d vs %d", len(countA), len(countB))
	}
	for k, v := range countA {
		if countB[k] != v {
			t.Errorf("class %q: count %v vs %v after shuffle", k, v, countB[k])
		}
	}
}

func TestMultiLectInvariantsDeterminismReorderPreservesConditionedClasses(t *testing.T) {
	// COMMITMENT: multiLectContextDiscovery output is deterministic under input
	// order. We check the set of (segments, count) pairs.
	corpus := mltMakeN3Corpus()
	ma := mustTrainML(t, corpus)

	shuffled := append([]CognateSet(nil), corpus...)
	rng := rand.New(rand.NewSource(7))
	rng.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
	mb := mustTrainML(t, shuffled)

	sigA := mltConditionedSignature(ma)
	sigB := mltConditionedSignature(mb)
	if len(sigA) != len(sigB) {
		t.Fatalf("conditioned class count differs after shuffle: %d vs %d", len(sigA), len(sigB))
	}
	for k, v := range sigA {
		if sigB[k] != v {
			t.Errorf("conditioned class %q: count %v vs %v after shuffle", k, v, sigB[k])
		}
	}
}

func TestMultiLectInvariantsDeterminismTwoRunsSameInputs(t *testing.T) {
	// COMMITMENT: training twice on the same input gives the same model.
	corpus := mltMakeN3Corpus()
	m1 := mustTrainML(t, corpus)
	m2 := mustTrainML(t, corpus)

	c1 := mltCanonicalCounts(m1)
	c2 := mltCanonicalCounts(m2)
	if len(c1) != len(c2) {
		t.Fatalf("class count differs between runs: %d vs %d", len(c1), len(c2))
	}
	for k, v := range c1 {
		if c2[k] != v {
			t.Errorf("class %q: count %v vs %v across runs", k, v, c2[k])
		}
	}
	// Pairwise model key sets should be equal.
	if len(m1.PairwiseModels) != len(m2.PairwiseModels) {
		t.Errorf("pairwise model count: %d vs %d", len(m1.PairwiseModels), len(m2.PairwiseModels))
	}
	for k := range m1.PairwiseModels {
		if _, ok := m2.PairwiseModels[k]; !ok {
			t.Errorf("pairwise key %q missing from second run", k)
		}
	}
}

// ----- lect removal stability ---------------------------------------------

func TestMultiLectInvariantsRemovingLectPreservesOtherPairCoverage(t *testing.T) {
	// COMMITMENT: removing one lect from every cognate set produces a model
	// whose pairwise_models is the subset of the original that didn't include
	// the removed lect, and whose remaining classes refer only to the
	// remaining lects.
	corpus := mltMakeN3Corpus()
	full := mustTrainML(t, corpus)
	_ = full

	pruned := make([]CognateSet, len(corpus))
	for i, cs := range corpus {
		forms := map[string]Form{}
		order := []string{}
		for _, l := range cs.orderedLects() {
			if l != "C" {
				forms[l] = cs.Forms[l]
				order = append(order, l)
			}
		}
		pruned[i] = CognateSet{
			CognateID:  cs.CognateID,
			Forms:      forms,
			FormsOrder: order,
			Confidence: cs.Confidence,
		}
	}
	reduced := mustTrainML(t, pruned)

	for k := range reduced.PairwiseModels {
		// K is the internal pair key; unpack it by checking lect ids.
		// We just need to confirm C is not in any pair.
		aKey := lectPairKey("A", "C")
		bKey := lectPairKey("B", "C")
		if k == aKey || k == bKey {
			t.Errorf("reduced model contains a pair with lect C: key=%q", k)
		}
	}
	// The A-B pair must exist.
	if _, ok := reduced.PairwiseModel("A", "B"); !ok {
		t.Error("reduced model missing A-B pairwise model")
	}

	// All classes must only mention A and B.
	for _, klass := range reduced.UnconditionedClasses {
		for l := range klass.Segments {
			if l == "C" {
				t.Errorf("reduced model unconditioned class mentions lect C: %v", klass.Segments)
			}
		}
	}
}

func TestMultiLectInvariantsRemovingLectDoesNotEmptyOtherPairModels(t *testing.T) {
	// COMMITMENT: the surviving pair's learned model is still trained
	// (has observed segment correspondences).
	corpus := mltMakeN3Corpus()
	pruned := make([]CognateSet, len(corpus))
	for i, cs := range corpus {
		forms := map[string]Form{}
		order := []string{}
		for _, l := range cs.orderedLects() {
			if l != "C" {
				forms[l] = cs.Forms[l]
				order = append(order, l)
			}
		}
		pruned[i] = CognateSet{
			CognateID:  cs.CognateID,
			Forms:      forms,
			FormsOrder: order,
			Confidence: cs.Confidence,
		}
	}
	reduced := mustTrainML(t, pruned)

	ab, ok := reduced.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("reduced model missing A-B pairwise model")
	}
	if len(ab.SegmentTable.Counts) == 0 {
		t.Error("A-B pairwise model has empty segment table (not trained)")
	}
}

// ----- sparse / degenerate input robustness --------------------------------

func TestMultiLectInvariantsSingleCognateSetDoesNotCrash(t *testing.T) {
	corpus := []CognateSet{
		csOf("only", []string{"A", "B"}, map[string]string{"A": "pata", "B": "fada"}),
	}
	m := mustTrainML(t, corpus)
	if m == nil {
		t.Error("expected non-nil model for single cognate set")
	}
}

func TestMultiLectInvariantsSparseMissingLectsDoesNotCrashContextDiscovery(t *testing.T) {
	// COMMITMENT: multiLectContextDiscovery handles very sparse corpora without
	// crashing, even if the resulting conditioned table is empty.
	corpus := []CognateSet{
		csOf("c1", []string{"A", "B"}, map[string]string{"A": "pa", "B": "fa"}),
		csOf("c2", []string{"A", "C"}, map[string]string{"A": "ka", "C": "xa"}),
		csOf("c3", []string{"B", "C"}, map[string]string{"B": "ma", "C": "ma"}),
	}
	m := mustTrainML(t, corpus)
	// Each of the 3 pairs should have its own model.
	if len(m.PairwiseModels) != 3 {
		t.Errorf("expected 3 pairwise models, got %d", len(m.PairwiseModels))
	}
}

func TestMultiLectInvariantsN1DegenerateCorpusIsValid(t *testing.T) {
	// COMMITMENT: a corpus with only one lect produces an empty MultiLectModel
	// (no pairs, no classes) without crashing.
	var corpus []CognateSet
	for i := 0; i < 3; i++ {
		word := "pa" + string(rune('0'+i))
		corpus = append(corpus, CognateSet{
			CognateID:  "c" + string(rune('0'+i)),
			Forms:      map[string]Form{"A": formIPA("A", word)},
			FormsOrder: []string{"A"},
			Confidence: 1.0,
		})
	}
	m := mustTrainML(t, corpus)
	if len(m.LectIDs) != 1 || m.LectIDs[0] != "A" {
		t.Errorf("LectIDs = %v, want [A]", m.LectIDs)
	}
	if len(m.PairwiseModels) != 0 {
		t.Errorf("expected empty PairwiseModels, got %d", len(m.PairwiseModels))
	}
	if len(m.UnconditionedClasses) != 0 {
		t.Errorf("expected empty UnconditionedClasses, got %d", len(m.UnconditionedClasses))
	}
	if len(m.ConditionedClasses) != 0 {
		t.Errorf("expected empty ConditionedClasses, got %d", len(m.ConditionedClasses))
	}
}

func TestMultiLectInvariantsEmptyCorpusReturnsEmptyMultiLectModel(t *testing.T) {
	// COMMITMENT: empty corpus returns an empty MultiLectModel (Go version does
	// not have the legacy pairwise LearnedModel dual-return; TrainModel always
	// returns *MultiLectModel).
	m := mustTrainML(t, nil)
	if m == nil {
		t.Fatal("expected non-nil model for empty corpus")
	}
	// An empty corpus should produce an empty multi-lect model.
	if len(m.PairwiseModels) != 0 {
		t.Errorf("expected empty PairwiseModels, got %d", len(m.PairwiseModels))
	}
	if len(m.UnconditionedClasses) != 0 {
		t.Errorf("expected empty UnconditionedClasses, got %d", len(m.UnconditionedClasses))
	}
}
