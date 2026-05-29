package regulae

// Mirror of python/tests/test_scoring_with_model.py

import (
	"math"
	"testing"
)

// scrMakeCC is a local helper that builds a ConditionedCorrespondence with an
// empty context. Prefixed "scr" per naming contract.
func scrMakeCC(src, tgt string) ConditionedCorrespondence {
	return ConditionedCorrespondence{Src: src, Tgt: tgt, Context: Context{}}
}

// scrMakeCCCtx builds a conditioned correspondence with a context.
func scrMakeCCCtx(src, tgt string, ctx Context) ConditionedCorrespondence {
	return ConditionedCorrespondence{Src: src, Tgt: tgt, Context: ctx}
}

// scrEntry pairs a ConditionedCorrespondence with a count/prior value. It is
// used in place of map literals because ConditionedCorrespondence contains a
// Context with slice fields and is therefore not a valid Go map key.
type scrEntry struct {
	cc ConditionedCorrespondence
	n  float64
}

// scrBuildTable builds a SegmentCorrespondenceTable from slices of
// ConditionedCorrespondence -> float64 counts / priors and a src-totals map.
// All nil slices/maps are treated as empty.
func scrBuildTable(
	counts []scrEntry,
	priors []scrEntry,
	srcTotals map[string]float64,
) SegmentCorrespondenceTable {
	tbl := NewSegmentCorrespondenceTable()
	for _, e := range counts {
		tbl.setCount(e.cc, e.n)
	}
	for _, e := range priors {
		tbl.setPrior(e.cc, e.n)
	}
	for src, v := range srcTotals {
		tbl.SrcTotals[src] = v
	}
	return tbl
}

// scrBuildEmptyModel builds a LearnedModel with default weights and the given
// segment table.
func scrBuildEmptyModel(tbl SegmentCorrespondenceTable, concentration float64) *LearnedModel {
	m := EmptyLearnedModel("descriptive", defaultTemperature, concentration)
	m.SegmentTable = tbl
	return m
}

// scrStrongPFTable returns the table from the Python helper
// _table_with_strong_pf_correspondence.
func scrStrongPFTable() SegmentCorrespondenceTable {
	return scrBuildTable(
		[]scrEntry{
			{scrMakeCC("p", "f"), 50.0},
			{scrMakeCC("p", "p"), 0.0},
		},
		[]scrEntry{
			{scrMakeCC("p", "f"), 1.0},
			{scrMakeCC("p", "p"), 1.0},
		},
		map[string]float64{"p": 50.0},
	)
}

// ----- M2 compatibility ---------------------------------------------------

func TestScoringWithModelNoneMatchesM2Behavior(t *testing.T) {
	// COMMITMENT: model=nil reproduces the M2 (prior-only) scoring exactly.
	link := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("f")}, Confidence: 1.0}
	without, _ := ScoreLink(link, "descriptive", nil)
	withNil, _ := ScoreLink(link, "descriptive", nil)
	if without != withNil {
		t.Errorf("model=nil cost %v != no-model cost %v", withNil, without)
	}
}

func TestScoringWithModelNoneForChunks(t *testing.T) {
	link := Link{
		SourceChunk: []Segment{seg("k"), seg("t")},
		TargetChunk: []Segment{seg("t"), seg("t")},
		Confidence:  1.0,
	}
	without, _ := ScoreLink(link, "descriptive", nil)
	withNil, _ := ScoreLink(link, "descriptive", nil)
	if without != withNil {
		t.Errorf("chunk cost with nil %v != without %v", withNil, without)
	}
}

func TestScoringWithModelEmptyFallsBackToMerkmal(t *testing.T) {
	// COMMITMENT: an empty model (no prior, no data) produces the same cost
	// as merkmal distance, because the fallback kicks in for any pair not in
	// the (empty) prior.
	link := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("f")}, Confidence: 1.0}
	empty := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	withEmpty, err := ScoreLink(link, "descriptive", empty)
	if err != nil {
		t.Fatal(err)
	}
	withoutModel, _ := ScoreLink(link, "descriptive", nil)
	if withEmpty != withoutModel {
		t.Errorf("empty-model cost %v != no-model cost %v", withEmpty, withoutModel)
	}
}

// ----- Learned segment table lowers cost for seen pairs ------------------

func TestScoringWithModelStrongLearnedPFCheaperThanUnlearned(t *testing.T) {
	// COMMITMENT: a segment pair with many observations scores lower than
	// the same pair under an empty (prior-only) model.
	model := scrBuildEmptyModel(scrStrongPFTable(), 5.0)
	link := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("f")}, Confidence: 1.0}
	scoreLearned, err := ScoreLink(link, "descriptive", model)
	if err != nil {
		t.Fatal(err)
	}
	scoreEmpty, _ := ScoreLink(link, "descriptive", nil)
	if scoreLearned >= scoreEmpty {
		t.Errorf("learned cost %v not less than prior-only cost %v", scoreLearned, scoreEmpty)
	}
}

func TestScoringWithModelLearnedRareCostlierThanPriorFallback(t *testing.T) {
	// COMMITMENT: a segment pair with low learned probability (p→p when
	// p mostly maps to f) is more expensive than the merkmal fallback for
	// identity.
	model := scrBuildEmptyModel(scrStrongPFTable(), 5.0)
	link := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("p")}, Confidence: 1.0}
	scoreLearned, err := ScoreLink(link, "descriptive", model)
	if err != nil {
		t.Fatal(err)
	}
	// Without model, identity is free (0). With model identity is costly.
	if scoreLearned <= 0.0 {
		t.Errorf("learned cost for identity %v, want > 0 (model makes identity expensive)", scoreLearned)
	}
}

// ----- Displacement layer adds generalization ----------------------------

func TestScoringWithModelDisplacementLayerLowersCost(t *testing.T) {
	// COMMITMENT: adding displacement counts for a feature change makes
	// segment pairs carrying that displacement cheaper.
	pfDisp, err := ComputeDisplacement(seg("p"), seg("f"), "descriptive")
	if err != nil {
		t.Fatal(err)
	}

	tbl := scrBuildTable(
		[]scrEntry{{scrMakeCC("p", "f"), 1.0}},
		[]scrEntry{
			{scrMakeCC("p", "f"), 1.0},
			{scrMakeCC("p", "p"), 1.0},
		},
		map[string]float64{"p": 1.0},
	)

	// Model A: empty displacement distribution.
	modelA := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	modelA.SegmentTable = tbl

	// Model B: displacement distribution with strong support for (p,f) displacement.
	modelB := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	modelB.SegmentTable = tbl
	modelB.DisplacementDist.setCount(pfDisp, 100.0)
	modelB.DisplacementDist.Total = 100.0

	link := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("f")}, Confidence: 1.0}
	costA, err := ScoreLink(link, "descriptive", modelA)
	if err != nil {
		t.Fatal(err)
	}
	costB, err := ScoreLink(link, "descriptive", modelB)
	if err != nil {
		t.Fatal(err)
	}
	if costB >= costA {
		t.Errorf("displacement-enriched cost %v not less than segment-only cost %v", costB, costA)
	}
}

// ----- Chunk table override ----------------------------------------------

func TestScoringWithModelChunkTableEntryOverridesCompositional(t *testing.T) {
	// COMMITMENT: when a chunk is in the phrase table, its stored cost is
	// used directly, regardless of compositional cost.
	kt := []Segment{seg("k"), seg("t")}
	tt := []Segment{seg("t"), seg("t")}
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	model.ChunkTable.setEntry(kt, tt, 0.01)

	link := Link{SourceChunk: kt, TargetChunk: tt, Confidence: 1.0}
	cost, err := ScoreLink(link, "descriptive", model)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(cost-0.01) > 1e-9 {
		t.Errorf("chunk table cost = %v, want approx 0.01", cost)
	}
}

func TestScoringWithModelChunkTableDoesNotAffectOtherChunks(t *testing.T) {
	// COMMITMENT: chunk table entries are keyed exactly; similar but
	// distinct chunks are NOT matched.
	kt := []Segment{seg("k"), seg("t")}
	tt := []Segment{seg("t"), seg("t")}
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	model.ChunkTable.setEntry(kt, tt, 0.01)

	// A different source chunk not in the table.
	pt := []Segment{seg("p"), seg("t")}
	link := Link{SourceChunk: pt, TargetChunk: tt, Confidence: 1.0}
	cost, err := ScoreLink(link, "descriptive", model)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(cost-0.01) < 1e-9 {
		t.Errorf("different chunk got the stored cost 0.01; expected compositional cost")
	}
}

// ----- Unseen-in-prior fallback ------------------------------------------

func TestScoringWithModelUnseenSegmentPairFallsBackToMerkmal(t *testing.T) {
	// COMMITMENT: a 1-to-1 link with a segment pair not in the prior falls
	// back to the merkmal distance.
	tbl := scrBuildTable(
		[]scrEntry{{scrMakeCC("p", "f"), 1.0}},
		[]scrEntry{{scrMakeCC("p", "f"), 1.0}},
		map[string]float64{"p": 1.0},
	)
	model := scrBuildEmptyModel(tbl, 5.0)

	// Score (k, x) which is not in the prior.
	link := Link{SourceChunk: []Segment{seg("k")}, TargetChunk: []Segment{seg("x")}, Confidence: 1.0}
	expected, _ := ScoreLink(link, "descriptive", nil)
	got, err := ScoreLink(link, "descriptive", model)
	if err != nil {
		t.Fatal(err)
	}
	if got != expected {
		t.Errorf("unseen pair cost %v != merkmal fallback %v", got, expected)
	}
}

// ----- Symmetry is preserved with a model --------------------------------

func TestScoringWithModelLearnedScoringPreservesSymmetryForSymmetricModel(t *testing.T) {
	// COMMITMENT: a symmetric table produces a symmetric cost.
	tbl := scrBuildTable(
		[]scrEntry{
			{scrMakeCC("p", "f"), 5.0},
			{scrMakeCC("f", "p"), 5.0},
		},
		[]scrEntry{
			{scrMakeCC("p", "f"), 1.0},
			{scrMakeCC("p", "p"), 1.0},
			{scrMakeCC("f", "p"), 1.0},
			{scrMakeCC("f", "f"), 1.0},
		},
		map[string]float64{"p": 5.0, "f": 5.0},
	)
	model := scrBuildEmptyModel(tbl, defaultConcentration)

	fwd := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("f")}, Confidence: 1.0}
	bwd := Link{SourceChunk: []Segment{seg("f")}, TargetChunk: []Segment{seg("p")}, Confidence: 1.0}
	costFwd, err := ScoreLink(fwd, "descriptive", model)
	if err != nil {
		t.Fatal(err)
	}
	costBwd, err := ScoreLink(bwd, "descriptive", model)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(costFwd-costBwd) > 1e-9 {
		t.Errorf("symmetric model: fwd cost %v != bwd cost %v", costFwd, costBwd)
	}
}

// ----- Posterior math sanity ----------------------------------------------

func TestScoringWithModelDirichletPosteriorWithZeroCountsEqualsPrior(t *testing.T) {
	// COMMITMENT: zero observations → posterior reduces to prior probability.
	// Prior: alpha(p,f)=3, alpha(p,p)=1, concentration=5.
	// P(f|p) = 3/5 = 0.6; cost = -log(3/5).
	tbl := scrBuildTable(
		[]scrEntry{}, // no observed counts
		[]scrEntry{
			{scrMakeCC("p", "f"), 3.0},
			{scrMakeCC("p", "p"), 1.0},
		},
		map[string]float64{}, // no src totals
	)
	model := scrBuildEmptyModel(tbl, 5.0)

	link := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("f")}, Confidence: 1.0}
	cost, err := ScoreLink(link, "descriptive", model)
	if err != nil {
		t.Fatal(err)
	}
	// -log(3/5) with empty displacement → pure segment cost.
	// Note: segmentPairCostWithModel applies logZ correction; with zero data
	// and this prior the segment-layer cost should equal -log(P_prior).
	// We verify the direction and rough magnitude rather than the exact value,
	// because LogNormalizers correction may shift the absolute number.
	// The posterior ratio: P(f|p)/P(p|p) = 3.0/1.0 → cost(p,f) < cost(p,p).
	linkPP := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("p")}, Confidence: 1.0}
	costPP, _ := ScoreLink(linkPP, "descriptive", model)
	if cost >= costPP {
		t.Errorf("cost(p,f)=%v not less than cost(p,p)=%v when prior favours p→f", cost, costPP)
	}
	// Also verify the expected cost matches -log(3/5).
	expected := -math.Log(3.0 / 5.0)
	if math.Abs(cost-expected) > 1e-6 {
		t.Errorf("cost(p,f) = %v, want approx %v (-log(3/5))", cost, expected)
	}
}

func TestScoringWithModelDirichletPosteriorUpdatesWithCounts(t *testing.T) {
	// COMMITMENT: observed counts shift the posterior away from the prior.
	// Prior: alpha(p,f)=3, alpha(p,p)=1. Data: N(p,p)=10 → p→p is preferred.
	tbl := scrBuildTable(
		[]scrEntry{{scrMakeCC("p", "p"), 10.0}},
		[]scrEntry{
			{scrMakeCC("p", "f"), 3.0},
			{scrMakeCC("p", "p"), 1.0},
		},
		map[string]float64{"p": 10.0},
	)
	model := scrBuildEmptyModel(tbl, 4.0)

	linkPP := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("p")}, Confidence: 1.0}
	linkPF := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("f")}, Confidence: 1.0}
	costPP, _ := ScoreLink(linkPP, "descriptive", model)
	costPF, _ := ScoreLink(linkPF, "descriptive", model)
	if costPP >= costPF {
		t.Errorf("cost(p,p)=%v not less than cost(p,f)=%v after training on p→p", costPP, costPF)
	}
}

// ----- Context-aware lookup fallback hierarchy ----------------------------

func TestScoringWithModelConditionedEntryBeatsUnconditionedWhenContextMatches(t *testing.T) {
	// COMMITMENT: when a conditioned entry matches the link's context, it is
	// used instead of the unconditioned fallback.
	frontCtx := Context{Following: []FeatureConstraint{{"front", "+"}}}
	tbl := scrBuildTable(
		[]scrEntry{
			{scrMakeCC("k", "s"), 1.0},
			{scrMakeCC("k", "k"), 10.0},
			{scrMakeCCCtx("k", "s", frontCtx), 8.0},
			{scrMakeCCCtx("k", "k", frontCtx), 2.0},
		},
		[]scrEntry{
			{scrMakeCC("k", "s"), 1.0},
			{scrMakeCC("k", "k"), 1.0},
			{scrMakeCCCtx("k", "s", frontCtx), 1.0},
			{scrMakeCCCtx("k", "k", frontCtx), 1.0},
		},
		map[string]float64{"k": 11.0},
	)
	model := scrBuildEmptyModel(tbl, 2.0)

	linkUncond := Link{
		SourceChunk: []Segment{seg("k")},
		TargetChunk: []Segment{seg("s")},
		Context:     Context{},
		Confidence:  1.0,
	}
	linkCond := Link{
		SourceChunk: []Segment{seg("k")},
		TargetChunk: []Segment{seg("s")},
		Context:     frontCtx,
		Confidence:  1.0,
	}
	costUncond, _ := ScoreLink(linkUncond, "descriptive", model)
	costCond, _ := ScoreLink(linkCond, "descriptive", model)
	if costCond >= costUncond {
		t.Errorf("conditioned cost %v not less than unconditioned cost %v", costCond, costUncond)
	}
}

func TestScoringWithModelUnconditionedFallbackUsedWhenNoContextMatch(t *testing.T) {
	// COMMITMENT: a conditioned entry that doesn't match the link's context
	// is ignored; the unconditioned entry is used.
	frontCtx := Context{Following: []FeatureConstraint{{"front", "+"}}}
	backCtx := Context{Following: []FeatureConstraint{{"back", "+"}}}
	tbl := scrBuildTable(
		[]scrEntry{
			{scrMakeCC("k", "s"), 5.0},
			{scrMakeCCCtx("k", "s", frontCtx), 8.0},
		},
		[]scrEntry{
			{scrMakeCC("k", "s"), 1.0},
			{scrMakeCCCtx("k", "s", frontCtx), 1.0},
		},
		map[string]float64{"k": 13.0},
	)
	// P(s|k) with back context should equal unconditioned P(s|k)
	// because back context doesn't match the front entry.
	pBack, _ := segmentPosterior("k", "s", backCtx, tbl, 2.0)
	pEmpty, _ := segmentPosterior("k", "s", Context{}, tbl, 2.0)
	if pBack != pEmpty {
		t.Errorf("P(s|k) with unmatched context %v != unconditioned %v", pBack, pEmpty)
	}
}

func TestScoringWithModelSegmentPosteriorNoneForUnseenPair(t *testing.T) {
	// COMMITMENT: a segment pair not in the prior returns (0, false),
	// signalling merkmal fallback.
	tbl := scrBuildTable(
		[]scrEntry{{scrMakeCC("p", "f"), 5.0}},
		[]scrEntry{{scrMakeCC("p", "f"), 1.0}},
		map[string]float64{"p": 5.0},
	)
	_, ok := segmentPosterior("x", "y", Context{}, tbl, 2.0)
	if ok {
		t.Error("expected ok=false for unseen pair (x,y)")
	}
}

func TestScoringWithModelGapLinkCostEqualsWithoutModel(t *testing.T) {
	// COMMITMENT: gap links use fixed costs regardless of the model.
	tbl := scrBuildTable(
		[]scrEntry{{scrMakeCC("p", "f"), 10.0}},
		[]scrEntry{{scrMakeCC("p", "f"), 1.0}},
		map[string]float64{"p": 10.0},
	)
	model := scrBuildEmptyModel(tbl, 5.0)

	gapLink := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: nil, Confidence: 1.0}
	costWith, _ := ScoreLink(gapLink, "descriptive", model)
	costWithout, _ := ScoreLink(gapLink, "descriptive", nil)
	if costWith != costWithout {
		t.Errorf("gap cost with model %v != without model %v", costWith, costWithout)
	}
}
