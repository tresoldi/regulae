package regulae

import "testing"

// Mirror of python/tests/test_cognate_sets_from_pairs.py

func mdlFormFromWord(lect, word string) Form {
	segs := make([]Segment, len([]rune(word)))
	for i, r := range []rune(word) {
		segs[i] = Segment{Grapheme: string(r)}
	}
	return Form{LectID: lect, Segments: segs}
}

func TestCognateSetsFromPairsPairsToConateSetsBasic(t *testing.T) {
	// COMMITMENT: each pair becomes one CognateSet with the two forms
	// keyed by the given lect IDs.
	pairs := []FormPair{
		{Src: mdlFormFromWord("x", "pata"), Tgt: mdlFormFromWord("x", "fada")},
		{Src: mdlFormFromWord("x", "kat"), Tgt: mdlFormFromWord("x", "gat")},
	}
	sets := CognateSetsFromPairs(pairs, [2]string{"latin", "spanish"}, "")
	if len(sets) != 2 {
		t.Fatalf("expected 2 cognate sets, got %d", len(sets))
	}
	for _, s := range sets {
		if _, ok := s.Forms["latin"]; !ok {
			t.Error("missing latin form")
		}
		if _, ok := s.Forms["spanish"]; !ok {
			t.Error("missing spanish form")
		}
	}
	if sets[0].Forms["latin"].Segments[0].Grapheme != "p" {
		t.Errorf("first latin segment = %q, want %q", sets[0].Forms["latin"].Segments[0].Grapheme, "p")
	}
	if sets[0].Forms["spanish"].Segments[0].Grapheme != "f" {
		t.Errorf("first spanish segment = %q, want %q", sets[0].Forms["spanish"].Segments[0].Grapheme, "f")
	}
}

func TestCognateSetsFromPairsPairsToConateSetsRewritesLectIDsOnForms(t *testing.T) {
	// COMMITMENT: the Form.lect_id is overwritten with the requested
	// lect_ids so downstream code doesn't see a stale label.
	pairs := []FormPair{
		{Src: mdlFormFromWord("x", "pa"), Tgt: mdlFormFromWord("y", "fa")},
	}
	sets := CognateSetsFromPairs(pairs, [2]string{"A", "B"}, "")
	if sets[0].Forms["A"].LectID != "A" {
		t.Errorf("lect_id(A) = %q, want %q", sets[0].Forms["A"].LectID, "A")
	}
	if sets[0].Forms["B"].LectID != "B" {
		t.Errorf("lect_id(B) = %q, want %q", sets[0].Forms["B"].LectID, "B")
	}
}

func TestCognateSetsFromPairsPairsToConateSetsGeneratesUniqueCognateIDs(t *testing.T) {
	var pairs []FormPair
	for i := 0; i < 5; i++ {
		pairs = append(pairs, FormPair{
			Src: mdlFormFromWord("x", "a"),
			Tgt: mdlFormFromWord("x", "a"),
		})
	}
	sets := CognateSetsFromPairs(pairs, [2]string{"src", "tgt"}, "")
	ids := map[string]bool{}
	for _, s := range sets {
		ids[s.CognateID] = true
	}
	if len(ids) != 5 {
		t.Errorf("expected 5 unique IDs, got %d", len(ids))
	}
}

func TestCognateSetsFromPairsPairsToConateSetsDefaultsToSrcTgtLectIDs(t *testing.T) {
	pairs := []FormPair{
		{Src: mdlFormFromWord("x", "a"), Tgt: mdlFormFromWord("x", "b")},
	}
	// Default lect IDs are whatever the caller passes; pass "src"/"tgt".
	sets := CognateSetsFromPairs(pairs, [2]string{"src", "tgt"}, "")
	if _, ok := sets[0].Forms["src"]; !ok {
		t.Error("missing src form")
	}
	if _, ok := sets[0].Forms["tgt"]; !ok {
		t.Error("missing tgt form")
	}
}

func TestCognateSetsFromPairsEmptyInput(t *testing.T) {
	sets := CognateSetsFromPairs(nil, [2]string{"src", "tgt"}, "")
	if len(sets) != 0 {
		t.Errorf("expected empty result, got %d sets", len(sets))
	}
}

func TestCognateSetsFromPairsPreservesSyllableBreaks(t *testing.T) {
	srcSegs := []Segment{{Grapheme: "p"}, {Grapheme: "a"}, {Grapheme: "t"}, {Grapheme: "a"}}
	tgtSegs := []Segment{{Grapheme: "f"}, {Grapheme: "a"}, {Grapheme: "d"}, {Grapheme: "a"}}
	src := Form{LectID: "x", Segments: srcSegs, SyllableBreaks: []int{2}}
	tgt := Form{LectID: "y", Segments: tgtSegs, SyllableBreaks: []int{2}}
	pairs := []FormPair{{Src: src, Tgt: tgt}}
	sets := CognateSetsFromPairs(pairs, [2]string{"A", "B"}, "")
	aBreaks := sets[0].Forms["A"].SyllableBreaks
	bBreaks := sets[0].Forms["B"].SyllableBreaks
	if len(aBreaks) != 1 || aBreaks[0] != 2 {
		t.Errorf("A syllable_breaks = %v, want [2]", aBreaks)
	}
	if len(bBreaks) != 1 || bBreaks[0] != 2 {
		t.Errorf("B syllable_breaks = %v, want [2]", bBreaks)
	}
}
