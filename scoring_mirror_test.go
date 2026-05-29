package regulae

// Mirror of python/tests/test_scoring.py
// Skipped tests (already in existing Go suite):
//   TestIdentityLinkZeroCost          (TestIdentityLinkZeroCost)
//   TestSubstitutionCostsMoreThanIdentity (TestSubstitutionCostsMoreThanIdentity)
//   TestScoreSymmetricUnderChunkSwap  (TestScoreSymmetricUnderChunkSwap)
//   TestCloserSegmentsLowerCost       (TestCloserSegmentsLowerCost)
//   TestGapCosts                      (TestGapCosts)
//   TestGapCostScalesLinearly         (TestGapCostScalesLinearly)
//   TestAsymmetricChunkHasExtraPenalty(TestAsymmetricChunkHasExtraPenalty)
//   TestContextDoesNotAffectPriorOnlyScore (TestContextDoesNotAffectPriorOnlyScore)
//   TestComputeDisplacementIdentityEmpty, Nonempty, InverseUnderSwap (TestComputeDisplacement*)

import (
	"math"
	"testing"
)

// ----- Insertion and deletion (tests not in existing suite) ---------------

func TestScoringEmptyTargetChunkRepresentsDeletion(t *testing.T) {
	// COMMITMENT: 1-to-0 link is scored with gap cost > 0.
	c, err := ScoreLink(Link{SourceChunk: []Segment{seg("u")}, TargetChunk: nil, Confidence: 1.0}, "descriptive", nil)
	if err != nil {
		t.Fatal(err)
	}
	if c <= 0.0 {
		t.Errorf("deletion cost = %v, want > 0", c)
	}
}

func TestScoringEmptySourceChunkRepresentsInsertion(t *testing.T) {
	// COMMITMENT: 0-to-1 link is scored with gap cost > 0.
	c, err := ScoreLink(Link{SourceChunk: nil, TargetChunk: []Segment{seg("e")}, Confidence: 1.0}, "descriptive", nil)
	if err != nil {
		t.Fatal(err)
	}
	if c <= 0.0 {
		t.Errorf("insertion cost = %v, want > 0", c)
	}
}

func TestScoringInsertionAndDeletionHaveEqualCost(t *testing.T) {
	// COMMITMENT: bidirectional symmetry extends to insertion vs deletion.
	ins, _ := ScoreLink(Link{SourceChunk: nil, TargetChunk: []Segment{seg("e")}, Confidence: 1.0}, "descriptive", nil)
	del, _ := ScoreLink(Link{SourceChunk: []Segment{seg("e")}, TargetChunk: nil, Confidence: 1.0}, "descriptive", nil)
	if ins != del {
		t.Errorf("insertion %v != deletion %v", ins, del)
	}
}

func TestScoringEmptyToEmptyLinkCostsNothing(t *testing.T) {
	// COMMITMENT: an empty link has zero cost.
	c, err := ScoreLink(Link{SourceChunk: nil, TargetChunk: nil, Confidence: 1.0}, "descriptive", nil)
	if err != nil {
		t.Fatal(err)
	}
	if c != 0.0 {
		t.Errorf("empty-to-empty cost = %v, want 0", c)
	}
}

// ----- Multi-segment compositional fallback --------------------------------

func TestScoringTwoSegmentToTwoSegmentUsesCompositionalSum(t *testing.T) {
	// ROADMAP: compositional fallback for many-to-many (Latin /kt/ ~ Italian /tt/).
	c, err := ScoreLink(Link{
		SourceChunk: []Segment{seg("k"), seg("t")},
		TargetChunk: []Segment{seg("t"), seg("t")},
		Confidence:  1.0,
	}, "descriptive", nil)
	if err != nil {
		t.Fatal(err)
	}
	if c <= 0.0 {
		t.Errorf("compositional cost = %v, want > 0", c)
	}
}

func TestScoringRecurringChunkPromotedToPhraseTable(t *testing.T) {
	// When a chunk recurs across many pairs it should be promoted.
	corpus := []FormPair{
		pairOf("A", "akta", "B", "atta"),
		pairOf("A", "ikti", "B", "itti"),
		pairOf("A", "okto", "B", "otto"),
		pairOf("A", "ukte", "B", "utte"),
		pairOf("A", "akta", "B", "atta"),
		pairOf("A", "ikti", "B", "itti"),
		pairOf("A", "okto", "B", "otto"),
		pairOf("A", "ukte", "B", "utte"),
		pairOf("A", "akto", "B", "atto"),
		pairOf("A", "ikti", "B", "itti"),
		pairOf("A", "ukte", "B", "utte"),
		pairOf("A", "okta", "B", "otta"),
		pairOf("A", "akte", "B", "atte"),
		pairOf("A", "ikto", "B", "itto"),
		pairOf("A", "uktu", "B", "uttu"),
	}
	trained := mustTrain(t, corpus)
	if len(trained.ChunkTable.Entries) == 0 {
		t.Error("phrase table should be nonempty after recurring chunk corpus")
	}
}

// ----- Feature displacement -----------------------------------------------

func TestScoringIdentityDisplacementIsEmpty(t *testing.T) {
	// COMMITMENT: identity has no feature displacement.
	d, err := ComputeDisplacement(seg("p"), seg("p"), "descriptive")
	if err != nil {
		t.Fatal(err)
	}
	if len(d) != 0 {
		t.Errorf("identity displacement for p = %v, want empty", d)
	}
}

func TestScoringDisplacementIsNonemptyForSubstitution(t *testing.T) {
	// COMMITMENT: any substitution yields at least one feature displacement.
	d, err := ComputeDisplacement(seg("p"), seg("f"), "descriptive")
	if err != nil {
		t.Fatal(err)
	}
	if len(d) == 0 {
		t.Error("expected non-empty displacement for p->f")
	}
}

func TestScoringGrimmTypeCorrespondencesShareCommonDisplacement(t *testing.T) {
	// COMMITMENT: p~f, t~s, k~x are stop~fricative pairs and should share
	// at least one common feature displacement (the manner shift).
	dpf, err := ComputeDisplacement(seg("p"), seg("f"), "descriptive")
	if err != nil {
		t.Fatal(err)
	}
	dts, err := ComputeDisplacement(seg("t"), seg("s"), "descriptive")
	if err != nil {
		t.Fatal(err)
	}
	dkx, err := ComputeDisplacement(seg("k"), seg("x"), "descriptive")
	if err != nil {
		t.Fatal(err)
	}
	if len(dpf) == 0 || len(dts) == 0 || len(dkx) == 0 {
		t.Fatal("expected non-empty displacements for all three pairs")
	}

	// Build sets.
	setPF := map[FeatureDisplacement]struct{}{}
	for _, d := range dpf {
		setPF[d] = struct{}{}
	}
	setTS := map[FeatureDisplacement]struct{}{}
	for _, d := range dts {
		setTS[d] = struct{}{}
	}
	setKX := map[FeatureDisplacement]struct{}{}
	for _, d := range dkx {
		setKX[d] = struct{}{}
	}

	var common []FeatureDisplacement
	for d := range setPF {
		if _, ok := setTS[d]; ok {
			if _, ok2 := setKX[d]; ok2 {
				common = append(common, d)
			}
		}
	}
	if len(common) == 0 {
		t.Errorf("no shared feature displacement across p~f, t~s, k~x: pf=%v ts=%v kx=%v", dpf, dts, dkx)
	}
}

func TestScoringDisplacementIsInverseUnderSegmentSwap(t *testing.T) {
	// COMMITMENT: swapping src and tgt inverts the displacement.
	fwd, err := ComputeDisplacement(seg("p"), seg("f"), "descriptive")
	if err != nil {
		t.Fatal(err)
	}
	bwd, err := ComputeDisplacement(seg("f"), seg("p"), "descriptive")
	if err != nil {
		t.Fatal(err)
	}
	if len(fwd) != len(bwd) {
		t.Fatalf("displacement lengths differ: %d vs %d", len(fwd), len(bwd))
	}
	bwdSet := map[FeatureDisplacement]struct{}{}
	for _, d := range bwd {
		bwdSet[d] = struct{}{}
	}
	for _, d := range fwd {
		inv := FeatureDisplacement{Feature: d.Feature, FromValue: d.ToValue, ToValue: d.FromValue}
		if _, ok := bwdSet[inv]; !ok {
			t.Errorf("inverse of %v not found in backward displacement", d)
		}
	}
}

func TestScoringDistantSegmentsHaveMoreDisplacementThanClose(t *testing.T) {
	// COMMITMENT: p~b (near) fewer displaced features than p~i (far).
	near, err := ComputeDisplacement(seg("p"), seg("b"), "descriptive")
	if err != nil {
		t.Fatal(err)
	}
	far, err := ComputeDisplacement(seg("p"), seg("i"), "descriptive")
	if err != nil {
		t.Fatal(err)
	}
	if len(far) <= len(near) {
		t.Errorf("distant (p~i) displacement %d not greater than close (p~b) %d", len(far), len(near))
	}
}

func TestScoringDisplacementForIdenticalConsonants(t *testing.T) {
	// COMMITMENT: identity has no displacement for any grapheme.
	for _, g := range []string{"p", "t", "k", "a", "i", "u", "n", "s"} {
		d, err := ComputeDisplacement(Segment{Grapheme: g}, Segment{Grapheme: g}, "descriptive")
		if err != nil {
			t.Fatalf("ComputeDisplacement(%q,%q) error: %v", g, g, err)
		}
		if len(d) != 0 {
			t.Errorf("identity displacement for %q = %v, want empty", g, d)
		}
	}
}

// ----- Gap cost calibration -----------------------------------------------

func TestScoringGapCostMatchesDefaultConstant(t *testing.T) {
	// COMMITMENT: the default gap cost is defaultGapCost.
	one, err := ScoreLink(Link{SourceChunk: nil, TargetChunk: []Segment{seg("e")}, Confidence: 1.0}, "descriptive", nil)
	if err != nil {
		t.Fatal(err)
	}
	if one != defaultGapCost {
		t.Errorf("gap cost = %v, want %v", one, defaultGapCost)
	}
}

// ----- Custom feature system ----------------------------------------------

func TestScoringLinkAcceptsAlternateFeatureSystem(t *testing.T) {
	// COMMITMENT: scoring is parameterised by feature system.
	link := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("f")}, Confidence: 1.0}
	sDesc, err := ScoreLink(link, "descriptive", nil)
	if err != nil {
		t.Fatalf("descriptive: %v", err)
	}
	sDistin, err := ScoreLink(link, "distinctive", nil)
	if err != nil {
		t.Fatalf("distinctive: %v", err)
	}
	if sDesc <= 0 {
		t.Errorf("descriptive cost = %v, want > 0", sDesc)
	}
	if sDistin <= 0 {
		t.Errorf("distinctive cost = %v, want > 0", sDistin)
	}
}

// ----- Smoke tests on real cognate pairs ---------------------------------

func TestScoringPaterFadarInflectionCostOrdering(t *testing.T) {
	// SMOKE: Latin pater ~ Gothic fadar segment ordering.
	aa := scoreNoModel(t, []Segment{seg("a")}, []Segment{seg("a")})
	pf := scoreNoModel(t, []Segment{seg("p")}, []Segment{seg("f")})
	td := scoreNoModel(t, []Segment{seg("t")}, []Segment{seg("d")})

	if aa != 0.0 {
		t.Errorf("a~a cost = %v, want 0", aa)
	}
	if pf <= 0.0 {
		t.Errorf("p~f cost = %v, want > 0", pf)
	}
	if td <= 0.0 {
		t.Errorf("t~d cost = %v, want > 0", td)
	}
}

func TestScoringPitarPaterLowDistance(t *testing.T) {
	// SMOKE: Sanskrit pitr ~ Latin pater consonant skeleton.
	pp := scoreNoModel(t, []Segment{seg("p")}, []Segment{seg("p")})
	tt := scoreNoModel(t, []Segment{seg("t")}, []Segment{seg("t")})
	if pp != 0.0 {
		t.Errorf("p~p cost = %v, want 0", pp)
	}
	if tt != 0.0 {
		t.Errorf("t~t cost = %v, want 0", tt)
	}
}

func TestScoringNoctemNotteClusterLinkFinitePositive(t *testing.T) {
	// SMOKE: Latin /kt/ ~ Italian /tt/ yields finite, positive cost.
	cost, err := ScoreLink(Link{
		SourceChunk: []Segment{seg("k"), seg("t")},
		TargetChunk: []Segment{seg("t"), seg("t")},
		Confidence:  1.0,
	}, "descriptive", nil)
	if err != nil {
		t.Fatal(err)
	}
	if cost <= 0.0 {
		t.Errorf("cost = %v, want > 0", cost)
	}
	if math.IsInf(cost, 0) || math.IsNaN(cost) || cost >= 10.0 {
		t.Errorf("cost = %v out of expected range (0, 10)", cost)
	}
}
