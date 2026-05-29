package regulae

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// Link scoring via merkmal feature distances. Lower cost is better.
//
// Without a model, scoring is fixed-cost: 1-to-1 links cost the merkmal
// distance; 0-to-N / N-to-0 links cost a fixed gap per segment; N-to-M links
// use a compositional fallback (segments paired left-to-right, distances
// summed, plus a chunk-asymmetry penalty). With a model, the layered learned
// scoring applies (Dirichlet-posterior segment probability, displacement
// distribution, chunk phrase table, tonal and cross-dimensional overlays).
//
// Unknown graphemes are a hard error under the strict default policy. Deep
// scoring code panics with *UnknownGraphemeError; public entry points recover
// it and return it as an error. This mirrors the Python exception-propagation
// model without threading an error through every internal cost computation.

// UnknownGraphemeError is returned when a segment's grapheme is not recognised
// by the feature system in use. It carries the offending grapheme and system.
type UnknownGraphemeError struct {
	Grapheme      string
	FeatureSystem string
}

func (e *UnknownGraphemeError) Error() string {
	return fmt.Sprintf(
		"unknown grapheme %q in feature system %q. Either the grapheme is a typo, "+
			"or the feature system does not cover it. Fix the input or choose a different feature system.",
		e.Grapheme, e.FeatureSystem,
	)
}

// catchUnknownGrapheme recovers a panicked *UnknownGraphemeError into *err. Any
// other panic value is re-raised. Used as `defer catchUnknownGrapheme(&err)` at
// public boundaries.
func catchUnknownGrapheme(err *error) {
	if r := recover(); r != nil {
		if ug, ok := r.(*UnknownGraphemeError); ok {
			*err = ug
			return
		}
		panic(r)
	}
}

const (
	// defaultGapCost is the cost of inserting or deleting a single segment,
	// calibrated to roughly the cost of a moderate 1-to-1 substitution.
	defaultGapCost = 0.5
	// defaultChunkPenalty is added per unit of length asymmetry in a
	// multi-segment link.
	defaultChunkPenalty = 0.25
)

// segmentDistanceChecked returns the merkmal distance between two segments
// (grapheme only), panicking with *UnknownGraphemeError if either grapheme is
// unknown to the feature system. Suprasegmental annotations are not yet
// factored into the distance.
func segmentDistanceChecked(source, target Segment, featureSystem string) float64 {
	if getFeatures(source.Grapheme, featureSystem) == nil {
		panic(&UnknownGraphemeError{Grapheme: source.Grapheme, FeatureSystem: featureSystem})
	}
	if getFeatures(target.Grapheme, featureSystem) == nil {
		panic(&UnknownGraphemeError{Grapheme: target.Grapheme, FeatureSystem: featureSystem})
	}
	return segmentDistance(source.Grapheme, target.Grapheme, featureSystem)
}

// ScoreLink returns the cost of a single link. A nil model selects fixed-cost
// scoring; a non-nil model selects the layered learned scoring. The score is
// symmetric in the source and target chunks. Returns an *UnknownGraphemeError
// if a grapheme is not recognised.
func ScoreLink(link Link, featureSystem string, model *LearnedModel) (cost float64, err error) {
	defer catchUnknownGrapheme(&err)
	if featureSystem == "" {
		featureSystem = "descriptive"
	}
	cost = scoreLink(link, featureSystem, model)
	return cost, nil
}

// scoreLink is the internal, panic-on-unknown-grapheme implementation.
func scoreLink(link Link, featureSystem string, model *LearnedModel) float64 {
	src := link.SourceChunk
	tgt := link.TargetChunk

	// Pure insertion / deletion. Gap costs are fixed, not learned.
	if len(src) == 0 && len(tgt) == 0 {
		return 0.0
	}
	if len(src) == 0 {
		return defaultGapCost * float64(len(tgt))
	}
	if len(tgt) == 0 {
		return defaultGapCost * float64(len(src))
	}

	// Promoted chunk override: exact chunk pair in the phrase table.
	if model != nil {
		if cost, ok := model.ChunkTable.Entries[chunkPairKey(src, tgt)]; ok {
			return cost
		}
	}

	fs := featureSystem
	if model != nil {
		fs = model.FeatureSystem
	}

	// 1-to-1 link.
	if len(src) == 1 && len(tgt) == 1 {
		if model == nil {
			return segmentDistanceChecked(src[0], tgt[0], fs)
		}
		return segmentPairCostWithModel(src[0], tgt[0], model, link.Context)
	}

	// N-to-M compositional fallback.
	paired := len(src)
	if len(tgt) < paired {
		paired = len(tgt)
	}
	pairCost := 0.0
	for i := 0; i < paired; i++ {
		if model == nil {
			pairCost += segmentDistanceChecked(src[i], tgt[i], fs)
		} else {
			pairCost += segmentPairCostWithModel(src[i], tgt[i], model, link.Context)
		}
	}
	asymmetry := len(src) - len(tgt)
	if asymmetry < 0 {
		asymmetry = -asymmetry
	}
	gapCost := defaultGapCost * float64(asymmetry)
	chunkPenalty := defaultChunkPenalty * float64(asymmetry)
	return pairCost + gapCost + chunkPenalty
}

// segmentPairCostWithModel is the cost of a 1-to-1 link under the layered
// learned model: -log P_post(t|s) - log Z_prior(s) for the segment term,
// log-linearly combined with the displacement term, plus the additive tonal
// term. Falls back to the bare merkmal distance for pairs not in the prior.
func segmentPairCostWithModel(src, tgt Segment, model *LearnedModel, linkContext Context) float64 {
	s := src.Grapheme
	t := tgt.Grapheme

	pSeg, ok := segmentPosterior(s, t, linkContext, model.SegmentTable, model.Concentration)
	if !ok {
		return segmentDistanceChecked(src, tgt, model.FeatureSystem)
	}

	logZ := model.SegmentTable.LogNormalizers[s]
	costSeg := safeNegLog(pSeg) - logZ

	var costLayered float64
	if len(model.DisplacementDist.Counts) > 0 || model.DisplacementDist.Total > 0.0 {
		fellBack := false
		func() {
			defer func() {
				if r := recover(); r != nil {
					if _, isUG := r.(*UnknownGraphemeError); isUG {
						fellBack = true
						return
					}
					panic(r)
				}
			}()
			disp := computeDisplacementChecked(src, tgt, model.FeatureSystem)
			pDisp := displacementProbability(disp, model.DisplacementDist)
			vEff := len(model.DisplacementDist.Counts)
			if vEff < 1 {
				vEff = 1
			}
			costDisp := safeNegLog(pDisp) - math.Log(float64(vEff))
			costLayered = model.SegmentWeight*costSeg + model.DisplacementWeight*costDisp
		}()
		if fellBack {
			return segmentDistanceChecked(src, tgt, model.FeatureSystem)
		}
	} else {
		costLayered = costSeg
	}

	toneCost := tonalCost(src.Tone, tgt.Tone, model.TonalTable)
	return costLayered + model.ToneWeight*toneCost
}

// tonalCost is the negative log probability of a tonal correspondence, offset
// so an empty distribution contributes zero. Returns 0 when both tones are
// absent ("") or the tonal table is empty.
func tonalCost(srcTone, tgtTone string, table TonalCorrespondenceTable) float64 {
	if srcTone == "" && tgtTone == "" {
		return 0.0
	}
	if len(table.Counts) == 0 && len(table.PriorPseudoCounts) == 0 {
		return 0.0
	}
	key := TonalCorrespondence{SrcTone: srcTone, TgtTone: tgtTone}
	alpha := table.PriorPseudoCounts[key]
	n := table.Counts[key]
	totalForSrc := table.SrcTotals[srcTone]
	priorMass := 0.0
	for _, v := range table.PriorPseudoCounts {
		priorMass += v
	}
	if priorMass == 0.0 {
		priorMass = 1.0
	}
	denominator := totalForSrc + priorMass
	if denominator <= 0.0 {
		return 0.0
	}
	p := (alpha + n) / denominator
	if p <= 0.0 {
		return 0.0
	}
	union := map[TonalCorrespondence]struct{}{}
	for k := range table.Counts {
		union[k] = struct{}{}
	}
	for k := range table.PriorPseudoCounts {
		union[k] = struct{}{}
	}
	vEff := len(union)
	if vEff < 2 {
		vEff = 2
	}
	return safeNegLog(p) - math.Log(float64(vEff))
}

// ----- cross-dimensional overlay ------------------------------------------

const crossDimSmoothingAlpha = 1.0

// applyCrossDimensionalAdjustments returns the total cost adjustment from
// cross-dimensional rule matches across an alignment. Each committed rule
// contributes an additive Dirichlet-smoothed log-likelihood-ratio adjustment
// at every 1-to-1 link where its source predicate holds.
func applyCrossDimensionalAdjustments(alignment Alignment, model *LearnedModel) float64 {
	if len(model.CrossDimensionalTable.Entries) == 0 {
		return 0.0
	}

	type adjPair struct{ pos, neg float64 }
	precomputed := make([]adjPair, len(model.CrossDimensionalTable.Entries))
	for i, rule := range model.CrossDimensionalTable.Entries {
		pos, neg := precomputeRuleAdjustments(rule, model.TonalTable)
		precomputed[i] = adjPair{pos, neg}
	}

	total := 0.0
	srcPos := 0
	tgtPos := 0
	fs := model.FeatureSystem
	srcForm := alignment.SourceForm
	tgtForm := alignment.TargetForm

	for _, link := range alignment.Links {
		srcLen := len(link.SourceChunk)
		tgtLen := len(link.TargetChunk)
		if srcLen == 1 && tgtLen == 1 {
			for i, rule := range model.CrossDimensionalTable.Entries {
				total += applyRuleToLink(rule, precomputed[i].pos, precomputed[i].neg,
					srcForm, srcPos, tgtForm, tgtPos, fs)
			}
		}
		srcPos += srcLen
		tgtPos += tgtLen
	}
	return total
}

// precomputeRuleAdjustments returns (posAdj, negAdj) in nats for a rule against
// a tonal table: the cost adjustment when the rule's prediction is confirmed
// vs. contradicted.
func precomputeRuleAdjustments(rule CrossDimensionalLink, tonalTable TonalCorrespondenceTable) (float64, float64) {
	distinctTones := map[string]struct{}{}
	for key := range tonalTable.Counts {
		if key.TgtTone != "" {
			distinctTones[key.TgtTone] = struct{}{}
		}
	}
	vEff := len(distinctTones)
	if vEff < 2 {
		vEff = 2
	}

	alpha := crossDimSmoothingAlpha

	pCond := (rule.Count + alpha) / (rule.SrcCount + alpha*float64(vEff))
	pCond = math.Max(math.Min(pCond, 1.0-1e-12), 1e-12)

	totalForValue := 0.0
	totalAll := 0.0
	for key, count := range tonalTable.Counts {
		totalAll += count
		if key.TgtTone == rule.TgtValue {
			totalForValue += count
		}
	}
	var pBase float64
	if totalAll <= 0.0 {
		pBase = 1.0 / float64(vEff)
	} else {
		pBase = (totalForValue + alpha) / (totalAll + alpha*float64(vEff))
	}
	pBase = math.Max(math.Min(pBase, 1.0-1e-12), 1e-12)

	posAdj := -(math.Log(pCond) - math.Log(pBase))
	negAdj := -(math.Log(1.0-pCond) - math.Log(1.0-pBase))
	return posAdj, negAdj
}

// applyRuleToLink evaluates one rule against one 1-to-1 link position pair,
// returning its cost contribution (0 if the source predicate does not hold or
// the target context cannot be evaluated).
func applyRuleToLink(rule CrossDimensionalLink, posAdj, negAdj float64, srcForm Form, srcPos int, tgtForm Form, tgtPos int, featureSystem string) float64 {
	if !sourcePredicateHolds(rule.SrcFeature, rule.SrcPosition, srcForm, srcPos, featureSystem) {
		return 0.0
	}
	if rule.SrcFeature2 != nil {
		if rule.SrcPosition2 == "" {
			return 0.0
		}
		if !sourcePredicateHolds(*rule.SrcFeature2, rule.SrcPosition2, srcForm, srcPos, featureSystem) {
			return 0.0
		}
	}

	tgtIdx := tgtPos + rule.TgtPositionOffset
	if tgtIdx < 0 || tgtIdx >= len(tgtForm.Segments) {
		return 0.0
	}
	tgtSeg := tgtForm.Segments[tgtIdx]

	var actual string
	switch rule.TgtDimension {
	case "tone":
		actual = tgtSeg.Tone
	case "length":
		actual = tgtSeg.Length
	case "stress":
		actual = tgtSeg.Stress
	default:
		return 0.0
	}

	if actual == "" {
		return 0.0
	}
	if actual == rule.TgtValue {
		return posAdj
	}
	return negAdj
}

// sourcePredicateHolds reports whether featureConstraint holds at the segment
// identified by positionSpec relative to srcPos. Tonal predictors check
// Segment.Tone directly; others query merkmal. Returns false on out-of-bounds,
// absent annotation, unknown grapheme, or mismatched value.
func sourcePredicateHolds(fc FeatureConstraint, positionSpec string, srcForm Form, srcPos int, featureSystem string) bool {
	offset := parseRelativeOffset(positionSpec)
	idx := srcPos + offset
	if idx < 0 || idx >= len(srcForm.Segments) {
		return false
	}
	seg := srcForm.Segments[idx]
	if fc.Feature == "tone" {
		return seg.Tone != "" && seg.Tone == fc.Value
	}
	if seg.Grapheme == "" {
		return false
	}
	features := getFeatures(seg.Grapheme, featureSystem)
	if features == nil {
		return false
	}
	return features.has(fc.Feature)
}

// parseRelativeOffset parses a canonical src_position spec like "relative_-1".
// It panics on an unrecognised spec, which is an invariant violation (committed
// rules always carry valid relative specs).
func parseRelativeOffset(spec string) int {
	if !strings.HasPrefix(spec, "relative_") {
		panic(fmt.Sprintf("unrecognised src_position spec: %q", spec))
	}
	body := spec[len("relative_"):]
	n, err := strconv.Atoi(body)
	if err != nil {
		panic(fmt.Sprintf("unrecognised src_position spec: %q", spec))
	}
	return n
}

// segmentPosterior returns the Dirichlet posterior P(tgt | src, linkContext)
// from the segment table, using the most specific conditioned correspondence
// matching linkContext (falling through to the unconditioned entry). The bool
// is false if no matching entry exists in the prior pseudo-counts (caller
// should fall back to the merkmal distance).
func segmentPosterior(src, tgt string, linkContext Context, table SegmentCorrespondenceTable, concentration float64) (float64, bool) {
	bestKey, ok := findMostSpecificMatch(src, tgt, linkContext, table)
	if !ok {
		return 0, false
	}
	k := bestKey.key()
	alpha := table.PriorPseudoCounts[k]
	n := table.Counts[k]
	nSrc := table.SrcTotals[src]
	denominator := concentration + nSrc
	if denominator <= 0.0 {
		return 0, false
	}
	return (alpha + n) / denominator, true
}

// findMostSpecificMatch returns the most specific conditioned correspondence in
// the table whose context is a subset of linkContext, for the given (src, tgt)
// pair. Among equal-specificity matches the sorted-first key wins
// (deterministic tie-break).
func findMostSpecificMatch(src, tgt string, linkContext Context, table SegmentCorrespondenceTable) (ConditionedCorrespondence, bool) {
	// Collect candidate keys for (src, tgt) from counts and priors.
	candidateKeys := map[string]struct{}{}
	for k, cc := range table.Corr {
		if cc.Src == src && cc.Tgt == tgt {
			if _, inCounts := table.Counts[k]; inCounts {
				candidateKeys[k] = struct{}{}
			} else if _, inPriors := table.PriorPseudoCounts[k]; inPriors {
				candidateKeys[k] = struct{}{}
			}
		}
	}
	if len(candidateKeys) == 0 {
		return ConditionedCorrespondence{}, false
	}
	keys := make([]string, 0, len(candidateKeys))
	for k := range candidateKeys {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var best ConditionedCorrespondence
	found := false
	bestSpecificity := -1
	for _, k := range keys {
		cc := table.Corr[k]
		if !cc.Context.IsSubsetOf(linkContext) {
			continue
		}
		specificity := cc.Context.ConstraintCount()
		if specificity > bestSpecificity {
			best = cc
			found = true
			bestSpecificity = specificity
		}
	}
	return best, found
}

// displacementProbability is the posterior probability of a displacement vector
// under symmetric Dirichlet smoothing: (alpha + N(d)) / (alpha*V + N_total),
// with V the number of observed displacement vectors. Unseen displacements get
// alpha / (alpha*V + N_total).
func displacementProbability(disp []FeatureDisplacement, dist DisplacementDistribution) float64 {
	alpha := dist.PriorPseudoCount
	nD := dist.Counts[displacementKey(disp)]
	v := len(dist.Counts)
	if v < 1 {
		v = 1
	}
	denominator := alpha*float64(v) + dist.Total
	if denominator <= 0.0 {
		return 1.0
	}
	return (alpha + nD) / denominator
}

// safeNegLog returns -log(p), guarding against zero and negatives (returns +Inf).
func safeNegLog(p float64) float64 {
	if p <= 0.0 {
		return math.Inf(1)
	}
	return -math.Log(p)
}

// ComputeDisplacement computes the feature displacement vector between two
// segments: features present in one but not the other, as FeatureDisplacement
// entries. Identity returns an empty slice. Returns an *UnknownGraphemeError if
// a grapheme is unrecognised.
func ComputeDisplacement(source, target Segment, featureSystem string) (disp []FeatureDisplacement, err error) {
	defer catchUnknownGrapheme(&err)
	if featureSystem == "" {
		featureSystem = "descriptive"
	}
	disp = computeDisplacementChecked(source, target, featureSystem)
	return disp, nil
}

// computeDisplacementChecked is the internal, panic-on-unknown-grapheme form.
func computeDisplacementChecked(source, target Segment, featureSystem string) []FeatureDisplacement {
	srcFeatures := getFeatures(source.Grapheme, featureSystem)
	tgtFeatures := getFeatures(target.Grapheme, featureSystem)
	if srcFeatures == nil {
		panic(&UnknownGraphemeError{Grapheme: source.Grapheme, FeatureSystem: featureSystem})
	}
	if tgtFeatures == nil {
		panic(&UnknownGraphemeError{Grapheme: target.Grapheme, FeatureSystem: featureSystem})
	}

	if featureSetsEqual(srcFeatures, tgtFeatures) {
		return nil
	}

	var displacements []FeatureDisplacement
	// Features present in source but not target: lost.
	lost := setDifferenceSorted(srcFeatures, tgtFeatures)
	for _, feat := range lost {
		displacements = append(displacements, FeatureDisplacement{Feature: feat, FromValue: "present", ToValue: "absent"})
	}
	// Features present in target but not source: gained.
	gained := setDifferenceSorted(tgtFeatures, srcFeatures)
	for _, feat := range gained {
		displacements = append(displacements, FeatureDisplacement{Feature: feat, FromValue: "absent", ToValue: "present"})
	}
	return displacements
}

func featureSetsEqual(a, b featureSet) bool {
	if len(a) != len(b) {
		return false
	}
	for f := range a {
		if _, ok := b[f]; !ok {
			return false
		}
	}
	return true
}

// setDifferenceSorted returns the sorted features present in a but not b.
func setDifferenceSorted(a, b featureSet) []string {
	var out []string
	for f := range a {
		if _, ok := b[f]; !ok {
			out = append(out, f)
		}
	}
	sort.Strings(out)
	return out
}
