package regulae

import (
	"math/rand"
	"strconv"
	"strings"
)

// Bootstrap percentile intervals on trained-model counts. Given a base model
// and its corpus, resample the corpus with replacement, retrain, and replace
// the closed-form Wilson intervals with percentile intervals over the bootstrap
// distribution. Seeded for determinism. A key absent from a resample
// contributes a rate of 0, giving meaningful low lower-bounds on marginal rules.

// pairSampleAccumulators holds per-key sample buckets for one pairwise model.
type pairSampleAccumulators struct {
	segment      map[string][]float64
	displacement map[string][]float64
	tonal        map[TonalCorrespondence][]float64
	chunk        map[string][]float64
	crossDim     map[string][]float64
}

func newPairAccumulators(pm *LearnedModel) *pairSampleAccumulators {
	a := &pairSampleAccumulators{
		segment:      map[string][]float64{},
		displacement: map[string][]float64{},
		tonal:        map[TonalCorrespondence][]float64{},
		chunk:        map[string][]float64{},
		crossDim:     map[string][]float64{},
	}
	for k := range pm.SegmentTable.Counts {
		a.segment[k] = nil
	}
	for k := range pm.DisplacementDist.Counts {
		a.displacement[k] = nil
	}
	for k := range pm.TonalTable.Counts {
		a.tonal[k] = nil
	}
	for k := range pm.ChunkTable.Entries {
		a.chunk[k] = nil
	}
	for _, r := range pm.CrossDimensionalTable.Entries {
		a.crossDim[crossDimKey(r)] = nil
	}
	return a
}

// accumulatePairModelSamples records one resample's rate for each base key,
// each computed within the sample's own denominator (bounded to [0, 1]).
func accumulatePairModelSamples(sample *LearnedModel, a *pairSampleAccumulators) {
	segCounts := sample.SegmentTable.Counts
	segTotals := sample.SegmentTable.SrcTotals
	for k := range a.segment {
		cc := sample.SegmentTable.Corr[k]
		n := segTotals[cc.Src]
		c := segCounts[k]
		rate := 0.0
		if n > 0.0 {
			rate = c / n
		}
		a.segment[k] = append(a.segment[k], rate)
	}
	dispCounts := sample.DisplacementDist.Counts
	dispTotal := sample.DisplacementDist.Total
	for k := range a.displacement {
		rate := 0.0
		if dispTotal > 0.0 {
			rate = dispCounts[k] / dispTotal
		}
		a.displacement[k] = append(a.displacement[k], rate)
	}
	toneCounts := sample.TonalTable.Counts
	toneTotals := sample.TonalTable.SrcTotals
	for k := range a.tonal {
		n := toneTotals[k.SrcTone]
		rate := 0.0
		if n > 0.0 {
			rate = toneCounts[k] / n
		}
		a.tonal[k] = append(a.tonal[k], rate)
	}
	chunkCounts := sample.ChunkTable.ObservationCounts
	chunkDenom := chunkBootstrapDenominator(sample)
	for k := range a.chunk {
		rate := 0.0
		if chunkDenom > 0.0 {
			rate = chunkCounts[k] / chunkDenom
		}
		a.chunk[k] = append(a.chunk[k], rate)
	}
	sampleCD := map[string]CrossDimensionalLink{}
	for _, r := range sample.CrossDimensionalTable.Entries {
		sampleCD[crossDimKey(r)] = r
	}
	for k := range a.crossDim {
		rate := 0.0
		if r, ok := sampleCD[k]; ok {
			rate = r.Confidence
		}
		a.crossDim[k] = append(a.crossDim[k], rate)
	}
}

// applyPairBootstrap returns a copy of base with bootstrap percentile intervals
// on every count-bearing entry.
func applyPairBootstrap(base *LearnedModel, a *pairSampleAccumulators) *LearnedModel {
	out := *base

	segUnc := map[string]UncertaintyEstimate{}
	for k, samples := range a.segment {
		cc := base.SegmentTable.Corr[k]
		segUnc[k] = bootstrapRateInterval(samples, base.SegmentTable.SrcTotals[cc.Src], DefaultAlpha)
	}
	st := base.SegmentTable
	st.Uncertainty = segUnc
	out.SegmentTable = st

	dispUnc := map[string]UncertaintyEstimate{}
	for k, samples := range a.displacement {
		dispUnc[k] = bootstrapRateInterval(samples, base.DisplacementDist.Total, DefaultAlpha)
	}
	dd := base.DisplacementDist
	dd.Uncertainty = dispUnc
	out.DisplacementDist = dd

	toneUnc := map[TonalCorrespondence]UncertaintyEstimate{}
	for k, samples := range a.tonal {
		toneUnc[k] = bootstrapRateInterval(samples, base.TonalTable.SrcTotals[k.SrcTone], DefaultAlpha)
	}
	tt := base.TonalTable
	tt.Uncertainty = toneUnc
	out.TonalTable = tt

	chunkUnc := map[string]UncertaintyEstimate{}
	chunkDenom := chunkBootstrapDenominator(base)
	for k, samples := range a.chunk {
		chunkUnc[k] = bootstrapRateInterval(samples, chunkDenom, DefaultAlpha)
	}
	ct := base.ChunkTable
	ct.Uncertainty = chunkUnc
	out.ChunkTable = ct

	newEntries := make([]CrossDimensionalLink, len(base.CrossDimensionalTable.Entries))
	for i, r := range base.CrossDimensionalTable.Entries {
		u := bootstrapRateInterval(a.crossDim[crossDimKey(r)], r.SrcCount, DefaultAlpha)
		r.Uncertainty = &u
		newEntries[i] = r
	}
	out.CrossDimensionalTable = CrossDimensionalLinkTable{Entries: newEntries}

	return &out
}

// chunkBootstrapDenominator recovers the total 1-to-1 observation mass (the n
// used for chunk uncertainty at base training) from src_totals.
func chunkBootstrapDenominator(model *LearnedModel) float64 {
	total := 0.0
	for _, v := range model.SegmentTable.SrcTotals {
		total += v
	}
	return total
}

func crossDimKey(rule CrossDimensionalLink) string {
	f2, v2 := "", ""
	if rule.SrcFeature2 != nil {
		f2, v2 = rule.SrcFeature2.Feature, rule.SrcFeature2.Value
	}
	return strings.Join([]string{
		rule.SrcFeature.Feature, rule.SrcFeature.Value, rule.SrcPosition,
		rule.TgtDimension, rule.TgtValue, strconv.Itoa(rule.TgtPositionOffset),
		f2, v2, rule.SrcPosition2,
	}, "\x1f")
}

// bootstrapPairwiseUncertainty resamples a pair corpus bootstrapN times and
// replaces base's Wilson intervals with bootstrap percentile intervals.
func bootstrapPairwiseUncertainty(base *LearnedModel, corpus []FormPair, bootstrapN, bootstrapSeed int, opts TrainOptions) *LearnedModel {
	rng := rand.New(rand.NewSource(int64(bootstrapSeed)))
	nPairs := len(corpus)
	acc := newPairAccumulators(base)
	for i := 0; i < bootstrapN; i++ {
		resample := make([]FormPair, nPairs)
		for j := 0; j < nPairs; j++ {
			resample[j] = corpus[rng.Intn(nPairs)]
		}
		sample := trainPairwiseLegacy(resample, nil, opts)
		accumulatePairModelSamples(sample, acc)
	}
	return applyPairBootstrap(base, acc)
}

func crossDimKeyML(r MultiLectCrossDimensionalLink) string {
	f2, v2 := "", ""
	if r.SrcFeature2 != nil {
		f2, v2 = r.SrcFeature2.Feature, r.SrcFeature2.Value
	}
	return strings.Join([]string{
		r.SrcFeature.Feature, r.SrcFeature.Value, r.SrcPosition,
		r.TgtDimension, r.TgtValue, strconv.Itoa(r.TgtPositionOffset),
		f2, v2, r.SrcPosition2,
	}, "\x1f")
}

func mlCrossKey(r MultiLectCrossDimensionalLink) string {
	return r.SrcLect + "\x1f" + r.TgtLect + "\x1f" + crossDimKeyML(r)
}

// condClassKey is a canonical identity for a conditioned class (segments +
// per-lect context signatures).
func condClassKey(c MultiLectCorrespondenceClass) string {
	var b strings.Builder
	b.WriteString(segItemsKey(sortedSegmentItems(c.Segments)))
	b.WriteString("\x1e")
	if c.Contexts != nil {
		lects := make([]string, 0, len(c.Contexts))
		for l := range c.Contexts {
			lects = append(lects, l)
		}
		sortStrings(lects)
		for _, l := range lects {
			b.WriteString(l)
			b.WriteString("=")
			b.WriteString(c.Contexts[l].key())
			b.WriteString(";")
		}
	}
	return b.String()
}

// bootstrapMultiLectUncertainty resamples the cognate-set corpus and replaces
// intervals on the base MultiLectModel with bootstrap percentile intervals.
func bootstrapMultiLectUncertainty(base *MultiLectModel, corpus []CognateSet, bootstrapN, bootstrapSeed int, opts TrainOptions) *MultiLectModel {
	rng := rand.New(rand.NewSource(int64(bootstrapSeed)))
	n := len(corpus)

	pairAcc := map[string]*pairSampleAccumulators{}
	for key, pm := range base.PairwiseModels {
		pairAcc[key] = newPairAccumulators(pm)
	}
	uncondSamples := map[string][]float64{}
	for _, c := range base.UnconditionedClasses {
		uncondSamples[segItemsKey(sortedSegmentItems(c.Segments))] = nil
	}
	condSamples := map[string][]float64{}
	for _, c := range base.ConditionedClasses {
		condSamples[condClassKey(c)] = nil
	}
	mlCrossSamples := map[string][]float64{}
	for _, r := range base.CrossDimensionalTable.Entries {
		mlCrossSamples[mlCrossKey(r)] = nil
	}

	for i := 0; i < bootstrapN; i++ {
		resample := make([]CognateSet, n)
		for j := 0; j < n; j++ {
			resample[j] = corpus[rng.Intn(n)]
		}
		sample := trainMultiLect(resample, opts)
		for key, acc := range pairAcc {
			if pm, ok := sample.PairwiseModels[key]; ok {
				accumulatePairModelSamples(pm, acc)
			} else {
				accumulatePairModelSamples(EmptyLearnedModel(opts.FeatureSystem, opts.Temperature, opts.Concentration), acc)
			}
		}
		sampleUncond := map[string]MultiLectCorrespondenceClass{}
		for _, c := range sample.UnconditionedClasses {
			sampleUncond[segItemsKey(sortedSegmentItems(c.Segments))] = c
		}
		for k := range uncondSamples {
			rate := 0.0
			if c, ok := sampleUncond[k]; ok && c.Uncertainty != nil && c.Uncertainty.N > 0.0 {
				rate = c.Count / c.Uncertainty.N
			}
			uncondSamples[k] = append(uncondSamples[k], rate)
		}
		sampleCond := map[string]MultiLectCorrespondenceClass{}
		for _, c := range sample.ConditionedClasses {
			sampleCond[condClassKey(c)] = c
		}
		for k := range condSamples {
			rate := 0.0
			if c, ok := sampleCond[k]; ok {
				rate = c.Confidence
			}
			condSamples[k] = append(condSamples[k], rate)
		}
		sampleMLCross := map[string]MultiLectCrossDimensionalLink{}
		for _, r := range sample.CrossDimensionalTable.Entries {
			sampleMLCross[mlCrossKey(r)] = r
		}
		for k := range mlCrossSamples {
			rate := 0.0
			if r, ok := sampleMLCross[k]; ok {
				rate = r.Confidence
			}
			mlCrossSamples[k] = append(mlCrossSamples[k], rate)
		}
	}

	out := *base
	newPairwise := map[string]*LearnedModel{}
	for key, pm := range base.PairwiseModels {
		newPairwise[key] = applyPairBootstrap(pm, pairAcc[key])
	}
	out.PairwiseModels = newPairwise

	newUncond := make([]MultiLectCorrespondenceClass, len(base.UnconditionedClasses))
	for i, c := range base.UnconditionedClasses {
		baseN := 0.0
		if c.Uncertainty != nil {
			baseN = c.Uncertainty.N
		}
		u := bootstrapRateInterval(uncondSamples[segItemsKey(sortedSegmentItems(c.Segments))], baseN, DefaultAlpha)
		c.Uncertainty = &u
		newUncond[i] = c
	}
	out.UnconditionedClasses = newUncond

	newCond := make([]MultiLectCorrespondenceClass, len(base.ConditionedClasses))
	for i, c := range base.ConditionedClasses {
		baseN := 0.0
		if c.Uncertainty != nil {
			baseN = c.Uncertainty.N
		}
		u := bootstrapRateInterval(condSamples[condClassKey(c)], baseN, DefaultAlpha)
		c.Uncertainty = &u
		newCond[i] = c
	}
	out.ConditionedClasses = newCond

	newMLCross := make([]MultiLectCrossDimensionalLink, len(base.CrossDimensionalTable.Entries))
	for i, r := range base.CrossDimensionalTable.Entries {
		u := bootstrapRateInterval(mlCrossSamples[mlCrossKey(r)], r.SrcCount, DefaultAlpha)
		r.Uncertainty = &u
		newMLCross[i] = r
	}
	out.CrossDimensionalTable = MultiLectCrossDimensionalLinkTable{Entries: newMLCross}

	return &out
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}
