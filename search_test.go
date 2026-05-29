package regulae

import (
	"math"
	"reflect"
	"testing"
)

func formIPA(lectID, ipa string) Form {
	runes := []rune(ipa)
	segs := make([]Segment, len(runes))
	for i, r := range runes {
		segs[i] = Segment{Grapheme: string(r)}
	}
	return Form{LectID: lectID, Segments: segs}
}

func mustAlign(t *testing.T, source, target Form, maxChunk int) Alignment {
	t.Helper()
	al, err := AlignForms(source, target, "descriptive", maxChunk, nil)
	if err != nil {
		t.Fatalf("AlignForms error: %v", err)
	}
	return al
}

func mustCost(t *testing.T, al Alignment, featureSystem string) float64 {
	t.Helper()
	c, err := AlignmentCost(al, featureSystem, nil)
	if err != nil {
		t.Fatalf("AlignmentCost error: %v", err)
	}
	return c
}

func assertCoversBothForms(t *testing.T, al Alignment) {
	t.Helper()
	var srcRec, tgtRec []Segment
	for _, link := range al.Links {
		srcRec = append(srcRec, link.SourceChunk...)
		tgtRec = append(tgtRec, link.TargetChunk...)
	}
	if !reflect.DeepEqual(srcRec, nonNil(al.SourceForm.Segments)) && !segmentSlicesEqual(srcRec, al.SourceForm.Segments) {
		t.Errorf("source coverage failed: recovered %v vs %v", srcRec, al.SourceForm.Segments)
	}
	if !segmentSlicesEqual(tgtRec, al.TargetForm.Segments) {
		t.Errorf("target coverage failed: recovered %v vs %v", tgtRec, al.TargetForm.Segments)
	}
}

func nonNil(s []Segment) []Segment {
	if s == nil {
		return []Segment{}
	}
	return s
}

func segmentSlicesEqual(a, b []Segment) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestAlignEmptyForms(t *testing.T) {
	src := Form{LectID: "A"}
	tgt := Form{LectID: "B"}
	al := mustAlign(t, src, tgt, 0)
	if len(al.Links) != 0 {
		t.Errorf("expected no links, got %d", len(al.Links))
	}
	if c := mustCost(t, al, "descriptive"); c != 0.0 {
		t.Errorf("cost = %v, want 0", c)
	}
}

func TestAlignIdenticalSingleSegment(t *testing.T) {
	al := mustAlign(t, formIPA("A", "p"), formIPA("B", "p"), 0)
	if len(al.Links) != 1 {
		t.Fatalf("expected 1 link, got %d", len(al.Links))
	}
	if c := mustCost(t, al, "descriptive"); c != 0.0 {
		t.Errorf("identity cost = %v, want 0", c)
	}
}

func TestAlignIdenticalFormsAllIdentityLinks(t *testing.T) {
	al := mustAlign(t, formIPA("A", "pater"), formIPA("B", "pater"), 0)
	assertCoversBothForms(t, al)
	if c := mustCost(t, al, "descriptive"); c != 0.0 {
		t.Errorf("cost = %v, want 0", c)
	}
	if len(al.Links) != 5 {
		t.Fatalf("expected 5 links, got %d", len(al.Links))
	}
	for _, link := range al.Links {
		if len(link.SourceChunk) != 1 || len(link.TargetChunk) != 1 {
			t.Errorf("non 1-to-1 link: %v", link)
		}
		if link.SourceChunk[0].Grapheme != link.TargetChunk[0].Grapheme {
			t.Errorf("non-identity link: %v", link)
		}
	}
}

func TestAlignEmptySourceIsPureInsertion(t *testing.T) {
	al := mustAlign(t, Form{LectID: "A"}, formIPA("B", "abc"), 0)
	assertCoversBothForms(t, al)
	for _, link := range al.Links {
		if len(link.SourceChunk) != 0 {
			t.Errorf("expected empty source chunk, got %v", link.SourceChunk)
		}
	}
}

func TestAlignEmptyTargetIsPureDeletion(t *testing.T) {
	al := mustAlign(t, formIPA("A", "abc"), Form{LectID: "B"}, 0)
	assertCoversBothForms(t, al)
	for _, link := range al.Links {
		if len(link.TargetChunk) != 0 {
			t.Errorf("expected empty target chunk, got %v", link.TargetChunk)
		}
	}
}

func TestCoverageInvariantVaried(t *testing.T) {
	cases := [][2]string{
		{"p", "f"}, {"pa", "fa"}, {"pater", "fadar"}, {"noktem", "notte"},
		{"a", "we"}, {"abc", "xy"}, {"", "xy"}, {"abc", ""},
	}
	for _, c := range cases {
		al := mustAlign(t, formIPA("A", c[0]), formIPA("B", c[1]), 0)
		assertCoversBothForms(t, al)
	}
}

// bruteForceMinCost enumerates all alignments and returns the minimum total
// cost, mirroring the Python brute-force reference (model-free scoring).
func bruteForceMinCost(t *testing.T, source, target Form, maxChunkSize int) float64 {
	t.Helper()
	n := len(source.Segments)
	m := len(target.Segments)
	memo := map[[2]int]float64{}
	var rec func(i, j int) float64
	rec = func(i, j int) float64 {
		if i == 0 && j == 0 {
			return 0.0
		}
		if v, ok := memo[[2]int{i, j}]; ok {
			return v
		}
		best := math.Inf(1)
		kMax := maxChunkSize
		if i < kMax {
			kMax = i
		}
		lMax := maxChunkSize
		if j < lMax {
			lMax = j
		}
		for k := 0; k <= kMax; k++ {
			for l := 0; l <= lMax; l++ {
				if k == 0 && l == 0 {
					continue
				}
				link := Link{SourceChunk: source.Segments[i-k : i], TargetChunk: target.Segments[j-l : j], Confidence: 1.0}
				lc, err := ScoreLink(link, "descriptive", nil)
				if err != nil {
					t.Fatalf("ScoreLink error: %v", err)
				}
				total := rec(i-k, j-l) + lc
				if total < best {
					best = total
				}
			}
		}
		memo[[2]int{i, j}] = best
		return best
	}
	return rec(n, m)
}

func TestSearchFindsMinimumCost(t *testing.T) {
	cases := [][2]string{
		{"p", "p"}, {"p", "f"}, {"pa", "fa"}, {"pat", "fad"},
		{"pater", "fadar"}, {"noktem", "notte"}, {"a", "we"}, {"kt", "tt"},
	}
	for _, c := range cases {
		src := formIPA("A", c[0])
		tgt := formIPA("B", c[1])
		al := mustAlign(t, src, tgt, 0)
		dp := mustCost(t, al, "descriptive")
		brute := bruteForceMinCost(t, src, tgt, DefaultMaxChunkSize)
		if math.Abs(dp-brute) > 1e-9 {
			t.Errorf("%v: DP cost %v != brute %v", c, dp, brute)
		}
	}
}

func TestAlignmentCostSymmetric(t *testing.T) {
	cases := [][2]string{
		{"pater", "fadar"}, {"noktem", "notte"}, {"abc", "xyz"}, {"", "xyz"},
	}
	for _, c := range cases {
		fwd := mustCost(t, mustAlign(t, formIPA("A", c[0]), formIPA("B", c[1]), 0), "descriptive")
		bwd := mustCost(t, mustAlign(t, formIPA("B", c[1]), formIPA("A", c[0]), 0), "descriptive")
		if math.Abs(fwd-bwd) > 1e-9 {
			t.Errorf("%v: forward %v != backward %v", c, fwd, bwd)
		}
	}
}

func TestSearchDeterministic(t *testing.T) {
	a1 := mustAlign(t, formIPA("A", "pater"), formIPA("B", "fadar"), 0)
	a2 := mustAlign(t, formIPA("A", "pater"), formIPA("B", "fadar"), 0)
	if !reflect.DeepEqual(a1, a2) {
		t.Error("alignment not deterministic")
	}
}

func TestOneToOneLinksHaveDisplacement(t *testing.T) {
	al := mustAlign(t, formIPA("A", "pater"), formIPA("B", "fadar"), 0)
	for _, link := range al.Links {
		if len(link.SourceChunk) != 1 || len(link.TargetChunk) != 1 {
			t.Fatalf("expected all 1-to-1 links")
		}
		if link.SourceChunk[0] != link.TargetChunk[0] {
			if len(link.FeatureDisplacement) == 0 {
				t.Errorf("substitution link missing displacement: %v", link)
			}
		}
	}
}

func TestIdentityLinksEmptyDisplacement(t *testing.T) {
	al := mustAlign(t, formIPA("A", "pater"), formIPA("B", "pater"), 0)
	for _, link := range al.Links {
		if len(link.FeatureDisplacement) != 0 {
			t.Errorf("identity link has displacement: %v", link)
		}
	}
}

func TestPaterFadarSegmentBySegment(t *testing.T) {
	al := mustAlign(t, formIPA("latin", "pater"), formIPA("gothic", "fadar"), 0)
	if len(al.Links) != 5 {
		t.Fatalf("expected 5 links, got %d", len(al.Links))
	}
	expected := [][2]string{{"p", "f"}, {"a", "a"}, {"t", "d"}, {"e", "a"}, {"r", "r"}}
	for i, link := range al.Links {
		if len(link.SourceChunk) != 1 || len(link.TargetChunk) != 1 {
			t.Fatalf("link %d not 1-to-1", i)
		}
		if link.SourceChunk[0].Grapheme != expected[i][0] || link.TargetChunk[0].Grapheme != expected[i][1] {
			t.Errorf("link %d = (%s, %s), want (%s, %s)", i,
				link.SourceChunk[0].Grapheme, link.TargetChunk[0].Grapheme, expected[i][0], expected[i][1])
		}
	}
}

func TestMaxChunkSizeOneIsNeedlemanWunsch(t *testing.T) {
	al := mustAlign(t, formIPA("A", "kt"), formIPA("B", "tt"), 1)
	assertCoversBothForms(t, al)
	for _, link := range al.Links {
		if len(link.SourceChunk) > 1 || len(link.TargetChunk) > 1 {
			t.Errorf("chunk size > 1 with max_chunk_size=1: %v", link)
		}
	}
}

func TestMaxChunkSizeZeroDefault(t *testing.T) {
	// 0 is treated as "use default" in the Go API; negative is rejected.
	_, err := AlignForms(formIPA("A", "p"), formIPA("B", "p"), "descriptive", -1, nil)
	if err == nil {
		t.Error("expected error for negative max_chunk_size")
	}
}

func TestLargerMaxChunkSizeCannotIncreaseCost(t *testing.T) {
	src := formIPA("A", "nokt")
	tgt := formIPA("B", "nott")
	c1 := mustCost(t, mustAlign(t, src, tgt, 1), "descriptive")
	c2 := mustCost(t, mustAlign(t, src, tgt, 2), "descriptive")
	c3 := mustCost(t, mustAlign(t, src, tgt, 3), "descriptive")
	if c2 > c1+1e-9 {
		t.Errorf("c2 %v > c1 %v", c2, c1)
	}
	if c3 > c2+1e-9 {
		t.Errorf("c3 %v > c2 %v", c3, c2)
	}
}

func TestAlignmentCostEqualsSumOfLinkScores(t *testing.T) {
	al := mustAlign(t, formIPA("A", "pater"), formIPA("B", "fadar"), 0)
	expected := 0.0
	for _, link := range al.Links {
		c, err := ScoreLink(link, "descriptive", nil)
		if err != nil {
			t.Fatal(err)
		}
		expected += c
	}
	if math.Abs(mustCost(t, al, "descriptive")-expected) > 1e-9 {
		t.Errorf("alignment cost != sum of link scores")
	}
}

func TestDPTiebreakerPrefersAtomicLinks(t *testing.T) {
	al := mustAlign(t, formIPA("A", "pata"), formIPA("B", "fada"), 0)
	for _, link := range al.Links {
		if len(link.SourceChunk) != 1 || len(link.TargetChunk) != 1 {
			t.Errorf("expected all 1-to-1 links, got chunk %v/%v", link.SourceChunk, link.TargetChunk)
		}
	}
}

func TestDPTiebreakerDoesNotFlipSubstantive(t *testing.T) {
	al := mustAlign(t, formIPA("A", "papa"), formIPA("B", "papa"), 0)
	for _, link := range al.Links {
		if !segmentSlicesEqual(link.SourceChunk, link.TargetChunk) {
			t.Errorf("non-identity link: %v", link)
		}
		if len(link.SourceChunk) != 1 {
			t.Errorf("non-atomic link: %v", link)
		}
	}
}
