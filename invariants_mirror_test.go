package regulae

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// Mirror of python/tests/test_invariants.py
//
// Property-based alignment invariant tests. All random inputs are seeded
// for determinism.

// A reasonably-covering set of graphemes known to be in merkmal's
// descriptive system.
var invGraphemeAlphabet = []string{
	"p", "b", "t", "d", "k", "ɡ", "m", "n", "ŋ",
	"f", "v", "s", "z", "ʃ", "ʒ", "h", "x",
	"r", "l", "j", "w",
	"a", "e", "i", "o", "u", "ɛ", "ɔ", "ə",
}

func invRandomForm(r *rand.Rand, lectID string, maxLength int) Form {
	length := r.Intn(maxLength + 1) // [0, maxLength]
	segs := make([]Segment, length)
	for i := range segs {
		segs[i] = Segment{Grapheme: invGraphemeAlphabet[r.Intn(len(invGraphemeAlphabet))]}
	}
	return Form{LectID: lectID, Segments: segs}
}

func invRandomPairs(seed int64, count, maxLength int) [][2]Form {
	r := rand.New(rand.NewSource(seed))
	pairs := make([][2]Form, count)
	for i := range pairs {
		pairs[i] = [2]Form{invRandomForm(r, "A", maxLength), invRandomForm(r, "B", maxLength)}
	}
	return pairs
}

func invAlign(t *testing.T, src, tgt Form) Alignment {
	t.Helper()
	al, err := AlignForms(src, tgt, "descriptive", 0, nil)
	if err != nil {
		t.Fatalf("AlignForms error: %v", err)
	}
	return al
}

func invCost(t *testing.T, al Alignment) float64 {
	t.Helper()
	c, err := AlignmentCost(al, "descriptive", nil)
	if err != nil {
		t.Fatalf("AlignmentCost error: %v", err)
	}
	return c
}

// ----- invariant: coverage -----------------------------------------------

func TestInvariantsCoverageHoldsForRandomInputs(t *testing.T) {
	// INVARIANT: for any form pair, the alignment links exhaustively
	// cover both forms.
	pairs := invRandomPairs(42, 50, 8)
	for _, p := range pairs {
		src, tgt := p[0], p[1]
		al := invAlign(t, src, tgt)
		var recoveredSrc, recoveredTgt []Segment
		for _, link := range al.Links {
			recoveredSrc = append(recoveredSrc, link.SourceChunk...)
			recoveredTgt = append(recoveredTgt, link.TargetChunk...)
		}
		if !segmentSlicesEqual(recoveredSrc, src.Segments) {
			t.Errorf("source coverage failed for %v ~ %v", invGraphemes(src), invGraphemes(tgt))
		}
		if !segmentSlicesEqual(recoveredTgt, tgt.Segments) {
			t.Errorf("target coverage failed for %v ~ %v", invGraphemes(src), invGraphemes(tgt))
		}
	}
}

// ----- invariant: cost non-negative --------------------------------------

func TestInvariantsCostIsNonNegativeForRandomInputs(t *testing.T) {
	// INVARIANT: alignment cost is always >= 0.
	pairs := invRandomPairs(43, 50, 8)
	for _, p := range pairs {
		al := invAlign(t, p[0], p[1])
		c := invCost(t, al)
		if c < 0.0 {
			t.Errorf("cost = %v < 0 for %v ~ %v", c, invGraphemes(p[0]), invGraphemes(p[1]))
		}
	}
}

// ----- invariant: symmetry -----------------------------------------------

func TestInvariantsSymmetryHoldsForRandomInputs(t *testing.T) {
	// INVARIANT: swapping source and target preserves the total cost.
	pairs := invRandomPairs(44, 50, 8)
	for _, p := range pairs {
		src, tgt := p[0], p[1]
		fwd := invCost(t, invAlign(t, src, tgt))
		bwd := invCost(t, invAlign(t, tgt, src))
		if math.Abs(fwd-bwd) > 1e-9 {
			t.Errorf("symmetry failed for %v ~ %v: fwd=%v bwd=%v",
				invGraphemes(src), invGraphemes(tgt), fwd, bwd)
		}
	}
}

// ----- invariant: identity -----------------------------------------------

func TestInvariantsIdentityHasZeroCostForRandomInputs(t *testing.T) {
	// INVARIANT: aligning a form with itself has cost 0.
	r := rand.New(rand.NewSource(45))
	for i := 0; i < 50; i++ {
		form := invRandomForm(r, "A", 8)
		al := invAlign(t, form, form)
		c := invCost(t, al)
		if c != 0.0 {
			t.Errorf("identity cost = %v for %v", c, invGraphemes(form))
		}
	}
}

// ----- invariant: determinism --------------------------------------------

func TestInvariantsDeterminismHoldsForRandomInputs(t *testing.T) {
	// INVARIANT: the same input always yields the same alignment.
	pairs := invRandomPairs(46, 30, 8)
	for _, p := range pairs {
		a1 := invAlign(t, p[0], p[1])
		a2 := invAlign(t, p[0], p[1])
		if !reflect.DeepEqual(a1, a2) {
			t.Errorf("non-deterministic alignment for %v ~ %v", invGraphemes(p[0]), invGraphemes(p[1]))
		}
	}
}

// ----- invariant: monotonicity in max_chunk_size -------------------------

func TestInvariantsLargerMaxChunkSizeNeverWorsensCostForRandomInputs(t *testing.T) {
	// INVARIANT: enlarging the search space cannot increase the optimum.
	pairs := invRandomPairs(47, 20, 6)
	for _, p := range pairs {
		al1, err := AlignForms(p[0], p[1], "descriptive", 1, nil)
		if err != nil {
			t.Fatalf("AlignForms error: %v", err)
		}
		al2, err := AlignForms(p[0], p[1], "descriptive", 2, nil)
		if err != nil {
			t.Fatalf("AlignForms error: %v", err)
		}
		al3, err := AlignForms(p[0], p[1], "descriptive", 3, nil)
		if err != nil {
			t.Fatalf("AlignForms error: %v", err)
		}
		c1, _ := AlignmentCost(al1, "descriptive", nil)
		c2, _ := AlignmentCost(al2, "descriptive", nil)
		c3, _ := AlignmentCost(al3, "descriptive", nil)
		if c2 > c1+1e-9 {
			t.Errorf("c2 %v > c1 %v for %v ~ %v", c2, c1, invGraphemes(p[0]), invGraphemes(p[1]))
		}
		if c3 > c2+1e-9 {
			t.Errorf("c3 %v > c2 %v for %v ~ %v", c3, c2, invGraphemes(p[0]), invGraphemes(p[1]))
		}
	}
}

// ----- invariant: link count bounds --------------------------------------

func TestInvariantsLinkCountIsBoundedForRandomInputs(t *testing.T) {
	// INVARIANT: the number of links is between 1 (both in one chunk)
	// and len(src)+len(tgt) (all 0-to-1 or 1-to-0).
	pairs := invRandomPairs(48, 50, 8)
	for _, p := range pairs {
		src, tgt := p[0], p[1]
		al := invAlign(t, src, tgt)
		n := len(src.Segments)
		m := len(tgt.Segments)
		if n == 0 && m == 0 {
			if len(al.Links) != 0 {
				t.Errorf("expected 0 links for empty/empty, got %d", len(al.Links))
			}
		} else {
			if len(al.Links) < 1 {
				t.Errorf("expected >= 1 links for non-empty pair, got %d", len(al.Links))
			}
			if len(al.Links) > n+m {
				t.Errorf("expected <= %d links, got %d", n+m, len(al.Links))
			}
		}
	}
}

// invGraphemes returns a short string representation of the form's graphemes.
func invGraphemes(f Form) []string {
	gs := make([]string, len(f.Segments))
	for i, s := range f.Segments {
		gs[i] = s.Grapheme
	}
	return gs
}
