package regulae

// Mirror of python/tests/test_multi_lect_reconciliation.py

import (
	"sort"
	"testing"
)

// mltClassesByKey returns a map keyed by sorted (lect, grapheme) pair slices
// encoded as a string to the count, matching the Python _classes_by_key helper.
func mltClassesByKey(model *MultiLectModel) map[string]float64 {
	out := map[string]float64{}
	for _, klass := range model.UnconditionedClasses {
		items := sortedSegmentItems(klass.Segments)
		key := segItemsKey(items)
		out[key] = klass.Count
	}
	return out
}

// mltSegKey encodes a list of (lect, grapheme) pairs as the canonical key used
// in mltClassesByKey, matching Python's tuple(sorted(k.segments.items())).
func mltSegKey(pairs ...string) string {
	// pairs is alternating lect, grapheme
	items := make([]lectGrapheme, 0, len(pairs)/2)
	for i := 0; i+1 < len(pairs); i += 2 {
		items = append(items, lectGrapheme{Lect: pairs[i], Grapheme: pairs[i+1]})
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].Lect != items[j].Lect {
			return items[i].Lect < items[j].Lect
		}
		return items[i].Grapheme < items[j].Grapheme
	})
	return segItemsKey(items)
}

// ----- N=2 sanity ---------------------------------------------------------

func TestMultiLectReconciliationN2MatchesPairwiseCounts(t *testing.T) {
	// COMMITMENT: for N=2 with a simple 1-to-1 corpus, reconciled
	// classes match the raw pairwise correspondence counts.
	pairs := []FormPair{
		pairOf("A", "pata", "B", "fada"),
		pairOf("A", "pata", "B", "fada"),
	}
	corpus := CognateSetsFromPairs(pairs, [2]string{"A", "B"}, "")
	m := mustTrainML(t, corpus)

	byKey := mltClassesByKey(m)

	pf := mltSegKey("A", "p", "B", "f")
	aa := mltSegKey("A", "a", "B", "a")
	td := mltSegKey("A", "t", "B", "d")

	if byKey[pf] != 2 {
		t.Errorf("A:p↔B:f count = %v, want 2", byKey[pf])
	}
	if byKey[td] != 2 {
		t.Errorf("A:t↔B:d count = %v, want 2", byKey[td])
	}
	if byKey[aa] != 4 {
		t.Errorf("A:a↔B:a count = %v, want 4 (two positions × two pairs)", byKey[aa])
	}
}

func TestMultiLectReconciliationClassesAreSortedByCountDesc(t *testing.T) {
	// COMMITMENT: OQ4 — classes ordered by count descending, with
	// lexicographic tiebreak.
	pairs := []FormPair{pairOf("A", "aaab", "B", "aaac")}
	corpus := CognateSetsFromPairs(pairs, [2]string{"A", "B"}, "")
	m := mustTrainML(t, corpus)

	counts := make([]float64, len(m.UnconditionedClasses))
	for i, klass := range m.UnconditionedClasses {
		counts[i] = klass.Count
	}
	for i := 1; i < len(counts); i++ {
		if counts[i] > counts[i-1] {
			t.Errorf("classes not sorted by count desc: counts[%d]=%v > counts[%d]=%v",
				i, counts[i], i-1, counts[i-1])
		}
	}
}

func TestMultiLectReconciliationSupportingCognatesTracked(t *testing.T) {
	pairs := []FormPair{
		pairOf("A", "pa", "B", "fa"),
		pairOf("A", "pa", "B", "fa"),
	}
	corpus := CognateSetsFromPairs(pairs, [2]string{"A", "B"}, "")
	m := mustTrainML(t, corpus)

	// Every class in this micro-corpus should list both cognate IDs.
	for _, klass := range m.UnconditionedClasses {
		sc := map[string]bool{}
		for _, id := range klass.SupportingCognates {
			sc[id] = true
		}
		if !sc["pair.00000"] || !sc["pair.00001"] {
			t.Errorf("class %v: supporting_cognates = %v, want {pair.00000, pair.00001}",
				klass.Segments, klass.SupportingCognates)
		}
	}
}

// ----- N=3: merger disambiguation -----------------------------------------

func TestMultiLectReconciliationN3ProducesThreeLectClasses(t *testing.T) {
	// COMMITMENT: with three lects and consistent pairwise alignments,
	// reconciliation yields three-lect classes.
	corpus := []CognateSet{
		csOf("c1", []string{"A", "B", "C"}, map[string]string{"A": "pa", "B": "fa", "C": "pa"}),
		csOf("c2", []string{"A", "B", "C"}, map[string]string{"A": "pa", "B": "fa", "C": "pa"}),
	}
	m := mustTrainML(t, corpus)

	if len(m.UnconditionedClasses) < 1 {
		t.Fatal("expected at least one unconditioned class")
	}
	// There should be a class with all three lects participating.
	threeLect := 0
	for _, klass := range m.UnconditionedClasses {
		if len(klass.Segments) == 3 {
			threeLect++
		}
	}
	if threeLect < 1 {
		t.Error("expected at least one 3-lect class")
	}

	byKey := mltClassesByKey(m)
	pfp := mltSegKey("A", "p", "B", "f", "C", "p")
	aaa := mltSegKey("A", "a", "B", "a", "C", "a")
	if byKey[pfp] != 2 {
		t.Errorf("A:p↔B:f↔C:p count = %v, want 2", byKey[pfp])
	}
	if byKey[aaa] != 2 {
		t.Errorf("A:a↔B:a↔C:a count = %v, want 2", byKey[aaa])
	}
}

func TestMultiLectReconciliationMergerDisambiguationVisible(t *testing.T) {
	// COMMITMENT: when lect A has merged two proto-segments but lect B
	// preserves the distinction, the A↔B correspondence should yield two
	// distinct classes — one where A's merged segment aligns to B's
	// preserved reflex 1, and one where it aligns to reflex 2.
	corpus := []CognateSet{
		csOf("c1", []string{"A", "B"}, map[string]string{"A": "ka", "B": "ta"}),
		csOf("c2", []string{"A", "B"}, map[string]string{"A": "ka", "B": "ka"}),
	}
	m := mustTrainML(t, corpus)

	byKey := mltClassesByKey(m)
	kt := mltSegKey("A", "k", "B", "t")
	kk := mltSegKey("A", "k", "B", "k")
	if byKey[kt] < 1 {
		t.Errorf("A:k↔B:t class missing or count < 1: %v", byKey[kt])
	}
	if byKey[kk] < 1 {
		t.Errorf("A:k↔B:k class missing or count < 1: %v", byKey[kk])
	}
}

// ----- sparse data --------------------------------------------------------

func TestMultiLectReconciliationRecoversDataFromUnequalLengthChunks(t *testing.T) {
	// COMMITMENT: when training promotes an unequal-length chunk (e.g.
	// 2-to-1 like sk→ʃ), reconciliation decomposes the chunk via a
	// sub-alignment and recovers position-level observations rather
	// than dropping the whole alignment as inconsistent.
	pairs := []FormPair{
		pairOf("A", "ska", "B", "ʃa"),
		pairOf("A", "ski", "B", "ʃi"),
		pairOf("A", "sku", "B", "ʃu"),
		pairOf("A", "sko", "B", "ʃo"),
		pairOf("A", "ske", "B", "ʃe"),
		pairOf("A", "aska", "B", "aʃa"),
	}
	corpus := CognateSetsFromPairs(pairs, [2]string{"A", "B"}, "")
	m := mustTrainML(t, corpus)

	byKey := mltClassesByKey(m)
	// After sub-alignment decomposition we should see A:s↔B:ʃ as a class.
	sS := mltSegKey("A", "s", "B", "ʃ")
	if byKey[sS] < 1 {
		t.Errorf("expected a (A:s, B:ʃ) class from sub-alignment decomposition, "+
			"got keys: %v", byKey)
	}
}

func TestMultiLectReconciliationSparseMissingLectsOnlyAffectsThatSet(t *testing.T) {
	// COMMITMENT: a cognate set missing one of several lects contributes
	// classes only for the lects that are present.
	corpus := []CognateSet{
		csOf("c1", []string{"A", "B", "C"}, map[string]string{"A": "pa", "B": "fa", "C": "pa"}),
		csOf("c2", []string{"A", "B"}, map[string]string{"A": "pa", "B": "fa"}),
	}
	m := mustTrainML(t, corpus)

	hasThree := false
	hasTwo := false
	for _, klass := range m.UnconditionedClasses {
		if len(klass.Segments) == 3 {
			hasThree = true
		}
		if len(klass.Segments) == 2 {
			hasTwo = true
		}
	}
	if !hasThree {
		t.Error("expected at least one 3-lect class")
	}
	if !hasTwo {
		t.Error("expected at least one 2-lect class")
	}
}
