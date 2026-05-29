package regulae

// Mirror of python/tests/test_morpheme_boundaries.py

import "testing"

// trnMorphForm builds a Form with optional morpheme breaks.
func trnMorphForm(lect, ipa string, breaks []int) Form {
	segs := make([]Segment, len(ipa))
	for i, c := range ipa {
		segs[i] = Segment{Grapheme: string(c)}
	}
	return Form{LectID: lect, Segments: segs, MorphemeBreaks: breaks}
}

// trnMorphSyntheticPairs builds 16 stem+suffix gemination pairs + 8 controls,
// optionally with morpheme boundaries at position 3.
func trnMorphSyntheticPairs(withBoundaries bool) []FormPair {
	stems := []string{"pat", "pat", "pat", "pat", "tat", "tat", "kat", "kat",
		"mat", "mat", "nat", "nat", "rat", "rat", "bat", "dat"}
	suffixes := []string{"a", "i", "u", "o", "a", "i", "a", "u",
		"a", "o", "a", "e", "a", "u", "a", "a"}

	var pairs []FormPair
	for idx, stem := range stems {
		sfx := suffixes[idx]
		proto := stem + sfx
		derived := stem + string(stem[len(stem)-1]) + sfx // gemination at boundary
		var srcBreaks, tgtBreaks []int
		if withBoundaries {
			srcBreaks = []int{3}
			tgtBreaks = []int{3}
		}
		pairs = append(pairs, FormPair{
			Src: trnMorphForm("proto", proto, srcBreaks),
			Tgt: trnMorphForm("derived", derived, tgtBreaks),
		})
	}
	// controls (monomorphemic CVC, no transformation)
	controls := []string{"pat", "tap", "kap", "map", "pak", "tak", "mak", "nat"}
	for _, w := range controls {
		pairs = append(pairs, FormPair{
			Src: trnMorphForm("proto", w, nil),
			Tgt: trnMorphForm("derived", w, nil),
		})
	}
	return pairs
}

// TestMorphemeBoundariesSyntheticFiltersChunks checks that with boundaries
// supplied, boundary-crossing chunks are rejected, while without boundaries
// they are promoted.
func TestMorphemeBoundariesSyntheticFiltersChunks(t *testing.T) {
	pairsNo := trnMorphSyntheticPairs(false)
	pairsYes := trnMorphSyntheticPairs(true)

	setsNo := CognateSetsFromPairs(pairsNo, [2]string{"proto", "derived"}, "")
	setsYes := CognateSetsFromPairs(pairsYes, [2]string{"proto", "derived"}, "")

	mNo := mustTrainML(t, setsNo)
	mYes := mustTrainML(t, setsYes)

	pairKey := lectPairKey("proto", "derived")
	modelNo, ok1 := mNo.PairwiseModels[pairKey]
	modelYes, ok2 := mYes.PairwiseModels[pairKey]
	if !ok1 || !ok2 {
		t.Fatal("missing pairwise model for proto/derived")
	}

	// Compute dropped = chunksNo - chunksYes (by key).
	chunksNo := modelNo.ChunkTable.Entries
	chunksYes := modelYes.ChunkTable.Entries

	var dropped []string
	for k := range chunksNo {
		if _, inYes := chunksYes[k]; !inYes {
			dropped = append(dropped, k)
		}
	}

	if len(dropped) == 0 {
		t.Error("expected at least one boundary-crossing chunk to be rejected")
	}

	// Each dropped chunk should have src or tgt length >= 2.
	for _, k := range dropped {
		cp := modelNo.ChunkTable.Pairs[k]
		if len(cp.Src) < 2 && len(cp.Tgt) < 2 {
			t.Errorf("dropped chunk %q has both src and tgt length 1 (not boundary-crossing?): src=%v tgt=%v",
				k, cp.Src, cp.Tgt)
		}
	}
}

// TestMorphemeBoundariesFormBreaksPropagatesViaCognateSetsFromPairs checks
// that CognateSetsFromPairs preserves MorphemeBreaks.
func TestMorphemeBoundariesFormBreaksPropagatesViaCognateSetsFromPairs(t *testing.T) {
	src := trnMorphForm("p", "pata", []int{3})
	tgt := trnMorphForm("d", "patta", []int{3})
	sets := CognateSetsFromPairs([]FormPair{{Src: src, Tgt: tgt}}, [2]string{"p", "d"}, "")

	if len(sets) == 0 {
		t.Fatal("no cognate sets returned")
	}
	pForm, ok := sets[0].Forms["p"]
	if !ok {
		t.Fatal("no form for lect 'p'")
	}
	dForm, ok := sets[0].Forms["d"]
	if !ok {
		t.Fatal("no form for lect 'd'")
	}
	if len(pForm.MorphemeBreaks) != 1 || pForm.MorphemeBreaks[0] != 3 {
		t.Errorf("proto MorphemeBreaks = %v, want [3]", pForm.MorphemeBreaks)
	}
	if len(dForm.MorphemeBreaks) != 1 || dForm.MorphemeBreaks[0] != 3 {
		t.Errorf("derived MorphemeBreaks = %v, want [3]", dForm.MorphemeBreaks)
	}
}

// TestMorphemeBoundariesLegacyCognateSetMorphemeBoundariesStillWorks checks
// that CognateSet.MorphemeBoundaries (legacy field) flows into chunk promotion.
func TestMorphemeBoundariesLegacyCognateSetMorphemeBoundariesStillWorks(t *testing.T) {
	makeCS := func(id string, morphBounds map[string][]int) CognateSet {
		return CognateSet{
			CognateID: id,
			Forms: map[string]Form{
				"p": {LectID: "p", Segments: []Segment{
					{Grapheme: "p"}, {Grapheme: "a"}, {Grapheme: "t"}, {Grapheme: "a"},
				}},
				"d": {LectID: "d", Segments: []Segment{
					{Grapheme: "p"}, {Grapheme: "a"}, {Grapheme: "t"}, {Grapheme: "t"}, {Grapheme: "a"},
				}},
			},
			MorphemeBoundaries: morphBounds,
			FormsOrder:         []string{"p", "d"},
			Confidence:         1.0,
		}
	}
	makeCS2 := func(id string, morphBounds map[string][]int) CognateSet {
		return CognateSet{
			CognateID: id,
			Forms: map[string]Form{
				"p": {LectID: "p", Segments: []Segment{
					{Grapheme: "t"}, {Grapheme: "a"}, {Grapheme: "t"}, {Grapheme: "a"},
				}},
				"d": {LectID: "d", Segments: []Segment{
					{Grapheme: "t"}, {Grapheme: "a"}, {Grapheme: "t"}, {Grapheme: "t"}, {Grapheme: "a"},
				}},
			},
			MorphemeBoundaries: morphBounds,
			FormsOrder:         []string{"p", "d"},
			Confidence:         1.0,
		}
	}
	bounds := map[string][]int{"p": {3}, "d": {3}}
	var corpus []CognateSet
	for i := 0; i < 8; i++ {
		corpus = append(corpus, makeCS("c"+string(rune('0'+i)), bounds))
	}
	for i := 0; i < 8; i++ {
		corpus = append(corpus, makeCS2("d"+string(rune('0'+i)), bounds))
	}
	multi := mustTrainML(t, corpus)

	pairKey := lectPairKey("p", "d")
	model, ok := multi.PairwiseModels[pairKey]
	if !ok {
		t.Fatal("no pairwise model for p/d")
	}

	// Build the bad chunks as segment slices matching the chunkPairKey format.
	// Boundary-spanning chunks: (at→att) and (ta→tta).
	badAt := chunkPairKey(
		[]Segment{{Grapheme: "a"}, {Grapheme: "t"}},
		[]Segment{{Grapheme: "a"}, {Grapheme: "t"}, {Grapheme: "t"}},
	)
	badTa := chunkPairKey(
		[]Segment{{Grapheme: "t"}, {Grapheme: "a"}},
		[]Segment{{Grapheme: "t"}, {Grapheme: "t"}, {Grapheme: "a"}},
	)
	if _, ok := model.ChunkTable.Entries[badAt]; ok {
		t.Error("boundary-crossing chunk (at→att) should not be promoted")
	}
	if _, ok := model.ChunkTable.Entries[badTa]; ok {
		t.Error("boundary-crossing chunk (ta→tta) should not be promoted")
	}
}

// TestMorphemeBoundariesUnannotatedCorpusUnaffected checks that a corpus
// without morpheme_breaks behaves deterministically (retrain = same result).
func TestMorphemeBoundariesUnannotatedCorpusUnaffected(t *testing.T) {
	pairsNo := trnMorphSyntheticPairs(false)
	setsNo := CognateSetsFromPairs(pairsNo, [2]string{"proto", "derived"}, "")

	mA := mustTrainML(t, setsNo)
	mB := mustTrainML(t, setsNo)

	pairKey := lectPairKey("proto", "derived")
	a, ok1 := mA.PairwiseModels[pairKey]
	b, ok2 := mB.PairwiseModels[pairKey]
	if !ok1 || !ok2 {
		t.Fatal("missing pairwise model")
	}

	for k, vA := range a.ChunkTable.Entries {
		vB := b.ChunkTable.Entries[k]
		if vA != vB {
			t.Errorf("chunk[%q]: %v vs %v", k, vA, vB)
		}
	}
	for k, vB := range b.ChunkTable.Entries {
		vA := a.ChunkTable.Entries[k]
		if vA != vB {
			t.Errorf("chunk[%q] missing in a: %v", k, vB)
		}
	}
}

// TestMorphemeBoundariesBoundaryAtChunkEdgeDoesNotReject checks that a
// morpheme break at exactly the chunk start or end does not cause rejection.
func TestMorphemeBoundariesBoundaryAtChunkEdgeDoesNotReject(t *testing.T) {
	// spansBreak uses a map[int]struct{}.
	breaks := map[int]struct{}{3: {}}

	// chunk [3, 5): break at 3 is at start, not inside → should NOT span
	if spansBreak(3, 5, breaks) {
		t.Error("break at start of chunk [3,5) should not count as spanning")
	}
	// chunk [1, 3): break at 3 is at end, not inside → should NOT span
	if spansBreak(1, 3, breaks) {
		t.Error("break at end of chunk [1,3) should not count as spanning")
	}
	// chunk [2, 4): break at 3 is strictly inside → SHOULD span
	if !spansBreak(2, 4, breaks) {
		t.Error("break at 3 strictly inside [2,4) should count as spanning")
	}
}
