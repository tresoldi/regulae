package regulae

import (
	"math"
	"sort"
	"strconv"
	"strings"
)

// Context-split discovery: greedy BIC-gated splitting of segment
// correspondences by immediate-neighbour and long-range conditioning
// predicates.

// splitPredicate is a candidate conditioning predicate. Slot is the Context
// field the constraint applies to (e.g. "preceding", "following", "position",
// "preceding@2", "self_stress"); Feature is the feature name (or the position
// name for the "position" slot); Value is the expected value ("" means None,
// used only for the position slot — _applyPredicate substitutes "+").
type splitPredicate struct {
	slot    string
	feature string
	value   string
}

// splitFeatureInventory: candidate immediate-neighbour feature dimensions.
var splitFeatureInventory = []splitPredicate{
	{"following", "vowel", "+"},
	{"following", "front", "+"},
	{"following", "back", "+"},
	{"following", "close", "+"},
	{"following", "open", "+"},
	{"following", "long", "+"},
	{"preceding", "vowel", "+"},
	{"preceding", "front", "+"},
	{"preceding", "back", "+"},
	{"preceding", "voiced", "+"},
	{"preceding", "voiceless", "+"},
	{"preceding", "consonant", "+"},
	{"preceding", "long", "+"},
}

var splitPositions = []string{"initial", "medial", "final"}

// Long-range feature inventory (narrower than the immediate-neighbour set;
// "vowel"/"consonant" are omitted as tautological on syllable slots).
var longRangeFeatures = []string{"front", "back", "close", "open", "voiced", "voiceless", "long"}

var longRangeStructuralSlots = []string{"same_syllable", "next_syllable", "previous_syllable"}
var longRangeDistanceSlots = []string{"preceding@2", "preceding@3", "following@2", "following@3"}
var longRangeExistentialSlots = []string{"somewhere_preceding", "somewhere_following"}

// obsRow is one flattened observation for a fixed source grapheme: the target,
// the link context, and the confidence weight.
type obsRow struct {
	tgt    string
	ctx    Context
	weight float64
}

// flatObs is one flattened observation carrying its source grapheme.
type flatObs struct {
	src    string
	tgt    string
	ctx    Context
	weight float64
}

// countAcc accumulates conditioned counts plus their structured keys.
type countAcc struct {
	counts map[string]float64
	corr   map[string]ConditionedCorrespondence
}

func (a *countAcc) set(cc ConditionedCorrespondence, v float64) {
	k := cc.key()
	a.counts[k] = v
	a.corr[k] = cc
}

// contextDiscovery discovers immediate-neighbour context splits. For each
// source grapheme with at least two observed targets and enough mass, it
// greedily commits BIC-improving splits as new conditioned correspondence
// entries; unconditioned entries remain as fallbacks.
func contextDiscovery(corpus []FormPair, pairWeights []float64, model *LearnedModel, maxChunkSize int, bicConfig BICConfig) *LearnedModel {
	alignments := alignCorpus(corpus, model, maxChunkSize)

	// Flatten alignments into (src, tgt, context, weight) observations,
	// decomposing multi-segment links via a no-chunks sub-alignment.
	noChunks := *model
	noChunks.ChunkTable = NewChunkPhraseTable()

	var observations []flatObs
	for ai, alignment := range alignments {
		w := pairWeights[ai]
		if w <= 0.0 {
			continue
		}
		for _, link := range alignment.Links {
			sc, tc := link.SourceChunk, link.TargetChunk
			if len(sc) == 1 && len(tc) == 1 {
				observations = append(observations, flatObs{sc[0].Grapheme, tc[0].Grapheme, link.Context, w})
				continue
			}
			if len(sc) == 0 || len(tc) == 0 {
				continue
			}
			subSrc := Form{LectID: "_sub", Segments: sc}
			subTgt := Form{LectID: "_sub", Segments: tc}
			sub := alignForms(subSrc, subTgt, noChunks.FeatureSystem, 1, &noChunks)
			for _, subLink := range sub.Links {
				if len(subLink.SourceChunk) == 1 && len(subLink.TargetChunk) == 1 {
					observations = append(observations, flatObs{subLink.SourceChunk[0].Grapheme, subLink.TargetChunk[0].Grapheme, link.Context, w})
				}
			}
		}
	}

	if len(observations) == 0 {
		return model
	}

	bySource := map[string][]obsRow{}
	for _, o := range observations {
		bySource[o.src] = append(bySource[o.src], obsRow{tgt: o.tgt, ctx: o.ctx, weight: o.weight})
	}

	acc := &countAcc{
		counts: cloneFloatMap(model.SegmentTable.Counts),
		corr:   cloneCorr(model.SegmentTable.Corr),
	}
	newSrcTotals := cloneFloatMap(model.SegmentTable.SrcTotals)

	nTotal := 0.0
	for _, o := range observations {
		nTotal += o.weight
	}
	if nTotal == 0 {
		return model
	}
	lnN := math.Log(nTotal)

	observedStress := collectStressValuesFlat(observations)

	for _, src := range sortedStringKeysObs(bySource) {
		obs := bySource[src]
		targetSet := map[string]struct{}{}
		massSum := 0.0
		for _, o := range obs {
			targetSet[o.tgt] = struct{}{}
			massSum += o.weight
		}
		if len(targetSet) < 2 || massSum < 4.0 {
			continue
		}
		commitSplitsForSource(src, obs, acc, lnN, bicConfig.MaxSplitDepth, observedStress, bicConfig)
	}

	newTable := SegmentCorrespondenceTable{
		Counts:            acc.counts,
		PriorPseudoCounts: model.SegmentTable.PriorPseudoCounts,
		SrcTotals:         newSrcTotals,
		LogNormalizers:    model.SegmentTable.LogNormalizers,
		Uncertainty:       segmentCountsUncertainty(acc.counts, acc.corr, newSrcTotals),
		Corr:              acc.corr,
	}
	return replaceSegmentTable(model, newTable)
}

func collectStressValuesFlat(observations []flatObs) map[string]struct{} {
	values := map[string]struct{}{}
	for _, o := range observations {
		for _, slot := range [][]FeatureConstraint{o.ctx.SelfStress, o.ctx.PrecedingStress, o.ctx.FollowingStress} {
			for _, fc := range slot {
				if fc.Feature == "stress" {
					values[fc.Value] = struct{}{}
				}
			}
		}
	}
	return values
}

// segmentCountsUncertainty: Wilson interval on P(tgt | src, context) for every
// count entry; conditioned entries share src_totals[src] with the unconditioned
// entry, matching scoring-time denominators.
func segmentCountsUncertainty(counts map[string]float64, corr map[string]ConditionedCorrespondence, srcTotals map[string]float64) map[string]UncertaintyEstimate {
	out := make(map[string]UncertaintyEstimate, len(counts))
	for k, c := range counts {
		n := srcTotals[corr[k].Src]
		out[k] = WilsonInterval(c, n, DefaultAlpha)
	}
	return out
}

// commitSplitsForSource performs sequential greedy context splitting for one
// source grapheme: repeatedly finds the best BIC-improving split on the
// remaining observations, commits it (and a refinement of its YES partition),
// and continues on the NO partition.
func commitSplitsForSource(src string, observations []obsRow, acc *countAcc, lnN float64, maxDepth int, observedStress map[string]struct{}, bicConfig BICConfig) {
	minObs := float64(bicConfig.MinSplitObservations)
	deltaThreshold := bicConfig.DeltaBICThreshold

	remaining := observations
	committedCount := 0
	for committedCount < maxDepth*4 && observationWeight(remaining) >= minObs {
		baselineCost := groupCost(remaining)
		var bestPredicate *splitPredicate
		var bestYes, bestNo []obsRow
		bestDelta := deltaThreshold
		for _, predicate := range candidatePredicates(Context{}, observedStress) {
			yesObs, noObs := partition(remaining, predicate)
			if observationWeight(yesObs) < minObs || observationWeight(noObs) < minObs {
				continue
			}
			splitCost := groupCost(yesObs) + groupCost(noObs)
			reduction := baselineCost - splitCost
			deltaBIC := -2.0*reduction + lnN
			if deltaBIC < bestDelta {
				p := predicate
				bestDelta = deltaBIC
				bestPredicate = &p
				bestYes, bestNo = yesObs, noObs
			}
		}
		if bestPredicate == nil {
			break
		}
		yesContext := applyPredicate(Context{}, *bestPredicate)
		commitGroup(src, bestYes, yesContext, acc)
		refineSplit(src, bestYes, yesContext, 1, acc, lnN, maxDepth, observedStress, bicConfig)
		remaining = bestNo
		committedCount++
	}
}

// refineSplit recursively refines an already-committed conditioned entry,
// looking for a single additional conditioning axis that improves BIC further.
func refineSplit(src string, observations []obsRow, baseContext Context, depth int, acc *countAcc, lnN float64, maxDepth int, observedStress map[string]struct{}, bicConfig BICConfig) {
	minObs := float64(bicConfig.MinSplitObservations)
	deltaThreshold := bicConfig.DeltaBICThreshold

	if depth >= maxDepth || observationWeight(observations) < minObs {
		return
	}
	baselineCost := groupCost(observations)
	var bestPredicate *splitPredicate
	var bestYes []obsRow
	bestDelta := deltaThreshold
	for _, predicate := range candidatePredicates(baseContext, observedStress) {
		yesObs, noObs := partition(observations, predicate)
		if observationWeight(yesObs) < minObs || observationWeight(noObs) < minObs {
			continue
		}
		splitCost := groupCost(yesObs) + groupCost(noObs)
		reduction := baselineCost - splitCost
		deltaBIC := -2.0*reduction + lnN
		if deltaBIC < bestDelta {
			p := predicate
			bestDelta = deltaBIC
			bestPredicate = &p
			bestYes = yesObs
		}
	}
	if bestPredicate == nil {
		return
	}
	yesContext := applyPredicate(baseContext, *bestPredicate)
	commitGroup(src, bestYes, yesContext, acc)
	refineSplit(src, bestYes, yesContext, depth+1, acc, lnN, maxDepth, observedStress, bicConfig)
}

// observationWeight returns the total confidence-weighted mass.
func observationWeight(observations []obsRow) float64 {
	total := 0.0
	for _, o := range observations {
		total += o.weight
	}
	return total
}

// groupCost is the negative log-likelihood of a group of observations under a
// single unconditioned correspondence (MLE categorical fit). Target keys are
// summed in sorted order for deterministic floating-point results.
func groupCost(observations []obsRow) float64 {
	if len(observations) == 0 {
		return 0.0
	}
	targetCounts := map[string]float64{}
	for _, o := range observations {
		targetCounts[o.tgt] += o.weight
	}
	total := observationWeight(observations)
	cost := 0.0
	keys := make([]string, 0, len(targetCounts))
	for t := range targetCounts {
		keys = append(keys, t)
	}
	sort.Strings(keys)
	for _, t := range keys {
		n := targetCounts[t]
		p := n / total
		if p > 0 {
			cost += -n * math.Log(p)
		}
	}
	return cost
}

// candidatePredicates returns split predicates not already in baseContext.
func candidatePredicates(baseContext Context, observedStress map[string]struct{}) []splitPredicate {
	var candidates []splitPredicate
	existingFollowing := map[string]struct{}{}
	for _, c := range baseContext.Following {
		existingFollowing[c.Feature] = struct{}{}
	}
	existingPreceding := map[string]struct{}{}
	for _, c := range baseContext.Preceding {
		existingPreceding[c.Feature] = struct{}{}
	}
	for _, sp := range splitFeatureInventory {
		if sp.slot == "following" {
			if _, ok := existingFollowing[sp.feature]; ok {
				continue
			}
		}
		if sp.slot == "preceding" {
			if _, ok := existingPreceding[sp.feature]; ok {
				continue
			}
		}
		candidates = append(candidates, sp)
	}
	if baseContext.Position == "" {
		for _, pos := range splitPositions {
			candidates = append(candidates, splitPredicate{"position", pos, ""})
		}
	}
	if len(observedStress) > 0 {
		stressVals := make([]string, 0, len(observedStress))
		for v := range observedStress {
			stressVals = append(stressVals, v)
		}
		sort.Strings(stressVals)
		slots := []struct {
			name string
			base []FeatureConstraint
		}{
			{"self_stress", baseContext.SelfStress},
			{"preceding_stress", baseContext.PrecedingStress},
			{"following_stress", baseContext.FollowingStress},
		}
		for _, sl := range slots {
			existingValues := map[string]struct{}{}
			for _, c := range sl.base {
				if c.Feature == "stress" {
					existingValues[c.Value] = struct{}{}
				}
			}
			for _, value := range stressVals {
				if _, ok := existingValues[value]; ok {
					continue
				}
				candidates = append(candidates, splitPredicate{sl.name, "stress", value})
			}
		}
	}
	return candidates
}

// partition splits observations by whether the predicate is satisfied.
func partition(observations []obsRow, predicate splitPredicate) (yes, no []obsRow) {
	for _, o := range observations {
		if predicateHolds(o.ctx, predicate.slot, predicate.feature, predicate.value) {
			yes = append(yes, o)
		} else {
			no = append(no, o)
		}
	}
	return yes, no
}

func anyConstraint(cs []FeatureConstraint, feature, value string) bool {
	for _, c := range cs {
		if c.Feature == feature && c.Value == value {
			return true
		}
	}
	return false
}

func predicateHolds(ctx Context, slot, feature, value string) bool {
	switch {
	case slot == "following":
		return anyConstraint(ctx.Following, feature, value)
	case slot == "preceding":
		return anyConstraint(ctx.Preceding, feature, value)
	case slot == "position":
		return ctx.Position == feature
	case strings.HasPrefix(slot, "preceding@"):
		d, err := strconv.Atoi(slot[len("preceding@"):])
		if err != nil {
			return false
		}
		for _, dc := range ctx.PrecedingAtDistance {
			if dc.Offset == d && dc.Constraint.Feature == feature && dc.Constraint.Value == value {
				return true
			}
		}
		return false
	case strings.HasPrefix(slot, "following@"):
		d, err := strconv.Atoi(slot[len("following@"):])
		if err != nil {
			return false
		}
		for _, dc := range ctx.FollowingAtDistance {
			if dc.Offset == d && dc.Constraint.Feature == feature && dc.Constraint.Value == value {
				return true
			}
		}
		return false
	case slot == "somewhere_preceding":
		return anyConstraint(ctx.SomewherePreceding, feature, value)
	case slot == "somewhere_following":
		return anyConstraint(ctx.SomewhereFollowing, feature, value)
	case slot == "same_syllable":
		return anyConstraint(ctx.SameSyllable, feature, value)
	case slot == "next_syllable":
		return anyConstraint(ctx.NextSyllable, feature, value)
	case slot == "previous_syllable":
		return anyConstraint(ctx.PreviousSyllable, feature, value)
	case slot == "self_stress":
		return anyConstraint(ctx.SelfStress, feature, value)
	case slot == "preceding_stress":
		return anyConstraint(ctx.PrecedingStress, feature, value)
	case slot == "following_stress":
		return anyConstraint(ctx.FollowingStress, feature, value)
	}
	return false
}

// applyPredicate returns a new context with the predicate's constraint added.
func applyPredicate(baseContext Context, predicate splitPredicate) Context {
	slot, feature, value := predicate.slot, predicate.feature, predicate.value
	fcValue := value
	if fcValue == "" {
		fcValue = "+"
	}
	fc := FeatureConstraint{Feature: feature, Value: fcValue}
	out := baseContext
	switch {
	case slot == "following":
		out.Following = appendConstraint(baseContext.Following, fc)
	case slot == "preceding":
		out.Preceding = appendConstraint(baseContext.Preceding, fc)
	case slot == "position":
		out.Position = feature
	case strings.HasPrefix(slot, "preceding@"):
		d, _ := strconv.Atoi(slot[len("preceding@"):])
		out.PrecedingAtDistance = appendDistance(baseContext.PrecedingAtDistance, distanceConstraint{Offset: d, Constraint: fc})
	case strings.HasPrefix(slot, "following@"):
		d, _ := strconv.Atoi(slot[len("following@"):])
		out.FollowingAtDistance = appendDistance(baseContext.FollowingAtDistance, distanceConstraint{Offset: d, Constraint: fc})
	case slot == "somewhere_preceding":
		out.SomewherePreceding = appendConstraint(baseContext.SomewherePreceding, fc)
	case slot == "somewhere_following":
		out.SomewhereFollowing = appendConstraint(baseContext.SomewhereFollowing, fc)
	case slot == "same_syllable":
		out.SameSyllable = appendConstraint(baseContext.SameSyllable, fc)
	case slot == "next_syllable":
		out.NextSyllable = appendConstraint(baseContext.NextSyllable, fc)
	case slot == "previous_syllable":
		out.PreviousSyllable = appendConstraint(baseContext.PreviousSyllable, fc)
	case slot == "self_stress":
		out.SelfStress = appendConstraint(baseContext.SelfStress, fc)
	case slot == "preceding_stress":
		out.PrecedingStress = appendConstraint(baseContext.PrecedingStress, fc)
	case slot == "following_stress":
		out.FollowingStress = appendConstraint(baseContext.FollowingStress, fc)
	}
	return out
}

// appendConstraint returns a new slice with fc appended, not sharing the
// backing array (Context is treated as immutable).
func appendConstraint(base []FeatureConstraint, fc FeatureConstraint) []FeatureConstraint {
	out := make([]FeatureConstraint, len(base)+1)
	copy(out, base)
	out[len(base)] = fc
	return out
}

func appendDistance(base []distanceConstraint, dc distanceConstraint) []distanceConstraint {
	out := make([]distanceConstraint, len(base)+1)
	copy(out, base)
	out[len(base)] = dc
	return out
}

// commitGroup writes counts for each observed target in this group as new
// conditioned correspondence entries under the given context.
func commitGroup(src string, observations []obsRow, context Context, acc *countAcc) {
	targetCounts := map[string]float64{}
	for _, o := range observations {
		targetCounts[o.tgt] += o.weight
	}
	for t, n := range targetCounts {
		acc.set(ConditionedCorrespondence{Src: src, Tgt: t, Context: context}, n)
	}
}

// tonalAggregation re-aligns the corpus under the post-chunk-promotion model
// and counts each 1-to-1 link's tonal correspondence. The table is inert for
// non-tonal corpora (left empty when no link carries a tone).
func tonalAggregation(corpus []FormPair, pairWeights []float64, model *LearnedModel, maxChunkSize int) *LearnedModel {
	alignments := alignCorpus(corpus, model, maxChunkSize)
	counts := map[TonalCorrespondence]float64{}
	srcTotals := map[string]float64{}

	anyToned := false
	for ai, alignment := range alignments {
		w := pairWeights[ai]
		if w <= 0.0 {
			continue
		}
		for _, link := range alignment.Links {
			if len(link.SourceChunk) != 1 || len(link.TargetChunk) != 1 {
				continue
			}
			srcTone := link.SourceChunk[0].Tone
			tgtTone := link.TargetChunk[0].Tone
			if srcTone == "" && tgtTone == "" {
				continue
			}
			anyToned = true
			key := TonalCorrespondence{SrcTone: srcTone, TgtTone: tgtTone}
			counts[key] += w
			srcTotals[srcTone] += w
		}
	}

	if !anyToned {
		return model
	}

	priorPseudo := make(map[TonalCorrespondence]float64, len(counts))
	uncertainty := make(map[TonalCorrespondence]UncertaintyEstimate, len(counts))
	for k, c := range counts {
		priorPseudo[k] = 1.0
		uncertainty[k] = WilsonInterval(c, srcTotals[k.SrcTone], DefaultAlpha)
	}

	newTonal := TonalCorrespondenceTable{
		Counts:            counts,
		PriorPseudoCounts: priorPseudo,
		SrcTotals:         srcTotals,
		Uncertainty:       uncertainty,
	}
	m := *model
	m.TonalTable = newTonal
	return &m
}

// longRangeCandidatePredicates returns candidate long-range split predicates
// not already satisfied by baseContext.
func longRangeCandidatePredicates(baseContext Context) []splitPredicate {
	var candidates []splitPredicate
	slots := make([]string, 0, len(longRangeStructuralSlots)+len(longRangeDistanceSlots)+len(longRangeExistentialSlots))
	slots = append(slots, longRangeStructuralSlots...)
	slots = append(slots, longRangeDistanceSlots...)
	slots = append(slots, longRangeExistentialSlots...)
	for _, slot := range slots {
		for _, feat := range longRangeFeatures {
			if predicateHolds(baseContext, slot, feat, "+") {
				continue
			}
			candidates = append(candidates, splitPredicate{slot, feat, "+"})
		}
	}
	return candidates
}

// longRangeDiscovery discovers long-range context splits (distance-bounded,
// existential, syllable-structural), running after cross-dimensional
// discovery. Only 1-to-1 observations are used. Long-range entries are
// additional, more specific overlays alongside immediate-neighbour entries.
func longRangeDiscovery(corpus []FormPair, pairWeights []float64, model *LearnedModel, maxChunkSize int, bicConfig BICConfig) *LearnedModel {
	alignments := alignCorpus(corpus, model, maxChunkSize)

	bySource := map[string][]obsRow{}
	nTotal := 0.0
	hasObs := false
	for ai, alignment := range alignments {
		w := pairWeights[ai]
		if w <= 0.0 {
			continue
		}
		for _, link := range alignment.Links {
			sc, tc := link.SourceChunk, link.TargetChunk
			if len(sc) == 1 && len(tc) == 1 {
				bySource[sc[0].Grapheme] = append(bySource[sc[0].Grapheme], obsRow{tgt: tc[0].Grapheme, ctx: link.Context, weight: w})
				nTotal += w
				hasObs = true
			}
		}
	}

	if !hasObs {
		return model
	}

	acc := &countAcc{
		counts: cloneFloatMap(model.SegmentTable.Counts),
		corr:   cloneCorr(model.SegmentTable.Corr),
	}
	newSrcTotals := cloneFloatMap(model.SegmentTable.SrcTotals)
	lnN := math.Log(nTotal)

	for _, src := range sortedStringKeysObs(bySource) {
		obs := bySource[src]
		targetSet := map[string]struct{}{}
		for _, o := range obs {
			targetSet[o.tgt] = struct{}{}
		}
		if len(targetSet) < 2 || observationWeight(obs) < 4.0 {
			continue
		}
		commitLongRangeSplitsForSource(src, obs, acc, lnN, bicConfig.MaxSplitDepth, bicConfig)
	}

	newTable := SegmentCorrespondenceTable{
		Counts:            acc.counts,
		PriorPseudoCounts: model.SegmentTable.PriorPseudoCounts,
		SrcTotals:         newSrcTotals,
		LogNormalizers:    model.SegmentTable.LogNormalizers,
		Uncertainty:       segmentCountsUncertainty(acc.counts, acc.corr, newSrcTotals),
		Corr:              acc.corr,
	}
	return replaceSegmentTable(model, newTable)
}

// commitLongRangeSplitsForSource performs sequential greedy long-range
// splitting for one source. Mirrors commitSplitsForSource but uses long-range
// predicates, searches the full observation set, and applies a dominance
// filter (the YES group must be carried by a single target outcome).
func commitLongRangeSplitsForSource(src string, observations []obsRow, acc *countAcc, lnN float64, maxDepth int, bicConfig BICConfig) {
	minObs := float64(bicConfig.LongRangeMinSplitObservations)
	deltaThreshold := bicConfig.LongRangeDeltaBICThreshold
	dominantFraction := bicConfig.LongRangeMinDominantFraction

	remaining := observations
	committedCount := 0
	for committedCount < maxDepth*4 && observationWeight(remaining) >= minObs {
		baselineCost := groupCost(remaining)
		var bestPredicate *splitPredicate
		var bestYes, bestNo []obsRow
		bestDelta := deltaThreshold
		for _, predicate := range longRangeCandidatePredicates(Context{}) {
			yesObs, noObs := partition(remaining, predicate)
			if observationWeight(yesObs) < minObs || observationWeight(noObs) < minObs {
				continue
			}
			// Dominance filter.
			yesTargetCounts := map[string]float64{}
			for _, o := range yesObs {
				yesTargetCounts[o.tgt] += o.weight
			}
			yesMode := 0.0
			for _, v := range yesTargetCounts {
				if v > yesMode {
					yesMode = v
				}
			}
			yesWeight := observationWeight(yesObs)
			if yesWeight <= 0.0 || yesMode/yesWeight < dominantFraction {
				continue
			}
			splitCost := groupCost(yesObs) + groupCost(noObs)
			reduction := baselineCost - splitCost
			deltaBIC := -2.0*reduction + lnN
			if deltaBIC < bestDelta {
				p := predicate
				bestDelta = deltaBIC
				bestPredicate = &p
				bestYes, bestNo = yesObs, noObs
			}
		}
		if bestPredicate == nil {
			break
		}
		yesContext := applyPredicate(Context{}, *bestPredicate)
		commitGroup(src, bestYes, yesContext, acc)
		remaining = bestNo
		committedCount++
	}
}

// ----- small shared helpers -----------------------------------------------

func cloneFloatMap(src map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(src))
	for k, v := range src {
		out[k] = v
	}
	return out
}

func sortedStringKeysObs(m map[string][]obsRow) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
