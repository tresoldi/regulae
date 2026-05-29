package regulae

import (
	"math"
	"testing"
)

func seg(g string) Segment { return Segment{Grapheme: g} }

func scoreNoModel(t *testing.T, src, tgt []Segment) float64 {
	t.Helper()
	c, err := ScoreLink(Link{SourceChunk: src, TargetChunk: tgt, Confidence: 1.0}, "descriptive", nil)
	if err != nil {
		t.Fatalf("ScoreLink error: %v", err)
	}
	return c
}

func TestIdentityLinkZeroCost(t *testing.T) {
	if c := scoreNoModel(t, []Segment{seg("p")}, []Segment{seg("p")}); c != 0.0 {
		t.Errorf("identity cost = %v, want 0", c)
	}
}

func TestSubstitutionCostsMoreThanIdentity(t *testing.T) {
	identity := scoreNoModel(t, []Segment{seg("p")}, []Segment{seg("p")})
	sub := scoreNoModel(t, []Segment{seg("p")}, []Segment{seg("k")})
	if sub <= identity {
		t.Errorf("substitution %v not greater than identity %v", sub, identity)
	}
}

func TestScoreSymmetricUnderChunkSwap(t *testing.T) {
	fwd := scoreNoModel(t, []Segment{seg("p")}, []Segment{seg("b")})
	bwd := scoreNoModel(t, []Segment{seg("b")}, []Segment{seg("p")})
	if math.Abs(fwd-bwd) > 1e-12 {
		t.Errorf("not symmetric: %v vs %v", fwd, bwd)
	}
}

func TestCloserSegmentsLowerCost(t *testing.T) {
	// p->b (voicing only) should cost less than p->a (stop vs vowel).
	close := scoreNoModel(t, []Segment{seg("p")}, []Segment{seg("b")})
	far := scoreNoModel(t, []Segment{seg("p")}, []Segment{seg("a")})
	if close >= far {
		t.Errorf("close %v not less than far %v", close, far)
	}
}

func TestGapCosts(t *testing.T) {
	del := scoreNoModel(t, []Segment{seg("p")}, nil)
	ins := scoreNoModel(t, nil, []Segment{seg("p")})
	if del != defaultGapCost {
		t.Errorf("deletion cost = %v, want %v", del, defaultGapCost)
	}
	if ins != del {
		t.Errorf("insertion %v != deletion %v", ins, del)
	}
	if c := scoreNoModel(t, nil, nil); c != 0.0 {
		t.Errorf("empty-to-empty cost = %v, want 0", c)
	}
}

func TestGapCostScalesLinearly(t *testing.T) {
	one := scoreNoModel(t, []Segment{seg("p")}, nil)
	three := scoreNoModel(t, []Segment{seg("p"), seg("a"), seg("t")}, nil)
	if math.Abs(three-3*one) > 1e-12 {
		t.Errorf("gap cost not linear: %v vs 3*%v", three, one)
	}
}

func TestAsymmetricChunkHasExtraPenalty(t *testing.T) {
	// 2-to-1 link pays gap + chunk penalty for the asymmetry.
	symmetric := scoreNoModel(t, []Segment{seg("k"), seg("t")}, []Segment{seg("t"), seg("t")})
	asymmetric := scoreNoModel(t, []Segment{seg("k"), seg("t")}, []Segment{seg("t")})
	if asymmetric <= symmetric {
		t.Errorf("asymmetric %v not greater than symmetric %v", asymmetric, symmetric)
	}
}

func TestComputeDisplacementIdentityEmpty(t *testing.T) {
	d, err := ComputeDisplacement(seg("p"), seg("p"), "descriptive")
	if err != nil {
		t.Fatal(err)
	}
	if len(d) != 0 {
		t.Errorf("identity displacement = %v, want empty", d)
	}
}

func TestComputeDisplacementSubstitutionNonempty(t *testing.T) {
	d, err := ComputeDisplacement(seg("p"), seg("b"), "descriptive")
	if err != nil {
		t.Fatal(err)
	}
	if len(d) == 0 {
		t.Error("expected non-empty displacement for p->b")
	}
}

func TestComputeDisplacementInverseUnderSwap(t *testing.T) {
	fwd, _ := ComputeDisplacement(seg("p"), seg("b"), "descriptive")
	bwd, _ := ComputeDisplacement(seg("b"), seg("p"), "descriptive")
	if len(fwd) != len(bwd) {
		t.Fatalf("displacement lengths differ: %d vs %d", len(fwd), len(bwd))
	}
	// Each forward (feat, present, absent) should appear reversed in bwd.
	bwdSet := map[FeatureDisplacement]bool{}
	for _, d := range bwd {
		bwdSet[d] = true
	}
	for _, d := range fwd {
		inv := FeatureDisplacement{Feature: d.Feature, FromValue: d.ToValue, ToValue: d.FromValue}
		if !bwdSet[inv] {
			t.Errorf("inverse of %v not found in backward displacement", d)
		}
	}
}

func TestComputeDisplacementUnknownGrapheme(t *testing.T) {
	_, err := ComputeDisplacement(seg("QQZZ"), seg("p"), "descriptive")
	if err == nil {
		t.Fatal("expected error for unknown grapheme")
	}
	var ug *UnknownGraphemeError
	if !asUnknownGrapheme(err, &ug) || ug.Grapheme != "QQZZ" {
		t.Errorf("expected UnknownGraphemeError for QQZZ, got %v", err)
	}
}

func asUnknownGrapheme(err error, target **UnknownGraphemeError) bool {
	if ug, ok := err.(*UnknownGraphemeError); ok {
		*target = ug
		return true
	}
	return false
}

func TestContextDoesNotAffectPriorOnlyScore(t *testing.T) {
	withCtx := Link{
		SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("f")},
		Context:    Context{Position: "initial", Preceding: []FeatureConstraint{{"vowel", "+"}}},
		Confidence: 1.0,
	}
	withoutCtx := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("f")}, Confidence: 1.0}
	a, _ := ScoreLink(withCtx, "descriptive", nil)
	b, _ := ScoreLink(withoutCtx, "descriptive", nil)
	if a != b {
		t.Errorf("context affected prior-only score: %v vs %v", a, b)
	}
}
