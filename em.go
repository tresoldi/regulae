package regulae

import (
	"math"
	"sort"
)

// Segment-level expectation-maximization and the initial-model construction.

// initialModel builds the starting model: no counts, a merkmal-derived
// Dirichlet prior over the corpus grapheme inventory, and empty displacement /
// chunk / tonal tables. Pairs with weight <= 0 are excluded from the inventory.
// If prior is non-nil, its return value (nats) is added to the merkmal-distance
// logit before the softmax.
func initialModel(corpus []FormPair, pairWeights []float64, featureSystem string, temperature, concentration, segmentWeight, displacementWeight float64, prior TypologicalPrior) *LearnedModel {
	if pairWeights == nil {
		pairWeights = make([]float64, len(corpus))
		for i := range pairWeights {
			pairWeights[i] = 1.0
		}
	}
	srcGraphemes := map[string]struct{}{}
	tgtGraphemes := map[string]struct{}{}
	for i, p := range corpus {
		if pairWeights[i] <= 0.0 {
			continue
		}
		for _, s := range p.Src.Segments {
			srcGraphemes[s.Grapheme] = struct{}{}
		}
		for _, s := range p.Tgt.Segments {
			tgtGraphemes[s.Grapheme] = struct{}{}
		}
	}

	// Union of source and target vocabularies, sorted for deterministic
	// summation (float non-associativity in the softmax makes iteration
	// order otherwise visible in the prior values).
	vocabSet := map[string]struct{}{}
	for g := range srcGraphemes {
		vocabSet[g] = struct{}{}
	}
	for g := range tgtGraphemes {
		vocabSet[g] = struct{}{}
	}
	vocab := make([]string, 0, len(vocabSet))
	for g := range vocabSet {
		vocab = append(vocab, g)
	}
	sort.Strings(vocab)

	segTable := NewSegmentCorrespondenceTable()
	for _, s := range vocab {
		sKnown := getFeatures(s, featureSystem) != nil
		type logitEntry struct {
			t     string
			logit float64
		}
		var logits []logitEntry
		for _, t := range vocab {
			if !sKnown || getFeatures(t, featureSystem) == nil {
				continue
			}
			d := segmentDistance(s, t, featureSystem)
			logit := -temperature * d
			if prior != nil {
				logit += prior(s, t)
			}
			logits = append(logits, logitEntry{t: t, logit: logit})
		}
		if len(logits) == 0 {
			continue
		}
		maxLogit := logits[0].logit
		for _, e := range logits {
			if e.logit > maxLogit {
				maxLogit = e.logit
			}
		}
		total := 0.0
		exps := make([]float64, len(logits))
		for i, e := range logits {
			exps[i] = math.Exp(e.logit - maxLogit)
			total += exps[i]
		}
		if total <= 0.0 {
			continue
		}
		segTable.LogNormalizers[s] = maxLogit + math.Log(total)
		for i, e := range logits {
			cc := ConditionedCorrespondence{Src: s, Tgt: e.t, Context: Context{}}
			segTable.setPrior(cc, concentration*(exps[i]/total))
		}
	}

	return &LearnedModel{
		SegmentTable:          segTable,
		DisplacementDist:      NewDisplacementDistribution(),
		ChunkTable:            NewChunkPhraseTable(),
		TonalTable:            NewTonalCorrespondenceTable(),
		CrossDimensionalTable: CrossDimensionalLinkTable{},
		FeatureSystem:         featureSystem,
		Temperature:           temperature,
		Concentration:         concentration,
		SegmentWeight:         segmentWeight,
		DisplacementWeight:    displacementWeight,
		ToneWeight:            defaultToneWeight,
	}
}

// segmentEM runs iterative segment-level EM: align under the current model
// (E-step), rebuild segment counts from 1-to-1 links (M-step), until the
// relative change in total corpus cost drops below convergenceEps or maxIter is
// reached.
func segmentEM(corpus []FormPair, pairWeights []float64, initial *LearnedModel, maxChunkSize, maxIter int, convergenceEps float64) *LearnedModel {
	model := initial
	prevCost := math.Inf(1)

	for iter := 0; iter < maxIter; iter++ {
		alignments := alignCorpus(corpus, model, maxChunkSize)
		totalCost := 0.0
		for _, a := range alignments {
			totalCost += alignmentCost(a, model.FeatureSystem, model)
		}

		if !math.IsInf(prevCost, 1) {
			denom := math.Abs(prevCost)
			if denom < 1.0 {
				denom = 1.0
			}
			if math.Abs(prevCost-totalCost)/denom < convergenceEps {
				break
			}
		}
		prevCost = totalCost

		model = updateSegmentTable(model, alignments, pairWeights)
	}
	return model
}

// updateSegmentTable returns a new model with the segment table rebuilt from
// the 1-to-1 links of the given alignments (chunks and gaps ignored). Counts
// use the unconditioned (empty) context; the prior and log-normalizers are
// carried over unchanged.
func updateSegmentTable(model *LearnedModel, alignments []Alignment, pairWeights []float64) *LearnedModel {
	counts := map[string]float64{}
	corr := map[string]ConditionedCorrespondence{}
	srcTotals := map[string]float64{}

	if pairWeights == nil {
		pairWeights = make([]float64, len(alignments))
		for i := range pairWeights {
			pairWeights[i] = 1.0
		}
	}

	for ai, alignment := range alignments {
		w := pairWeights[ai]
		if w <= 0.0 {
			continue
		}
		for _, link := range alignment.Links {
			if len(link.SourceChunk) == 1 && len(link.TargetChunk) == 1 {
				s := link.SourceChunk[0].Grapheme
				t := link.TargetChunk[0].Grapheme
				cc := ConditionedCorrespondence{Src: s, Tgt: t, Context: Context{}}
				k := cc.key()
				counts[k] += w
				corr[k] = cc
				srcTotals[s] += w
			}
		}
	}

	// Corr must cover both the new count keys and the carried-over prior keys.
	mergedCorr := cloneCorr(model.SegmentTable.Corr)
	for k, cc := range corr {
		mergedCorr[k] = cc
	}

	newTable := SegmentCorrespondenceTable{
		Counts:            counts,
		PriorPseudoCounts: model.SegmentTable.PriorPseudoCounts,
		SrcTotals:         srcTotals,
		LogNormalizers:    model.SegmentTable.LogNormalizers,
		Uncertainty:       segmentUncertainty(counts, corr, srcTotals),
		Corr:              mergedCorr,
	}
	return replaceSegmentTable(model, newTable)
}

// segmentUncertainty returns Wilson intervals on P(tgt | src, context) =
// count / src_totals[src] for every count entry.
func segmentUncertainty(counts map[string]float64, corr map[string]ConditionedCorrespondence, srcTotals map[string]float64) map[string]UncertaintyEstimate {
	out := make(map[string]UncertaintyEstimate, len(counts))
	for k, c := range counts {
		n := srcTotals[corr[k].Src]
		out[k] = WilsonInterval(c, n, DefaultAlpha)
	}
	return out
}

// replaceSegmentTable returns a copy of the model with a new segment table,
// preserving all other fields.
func replaceSegmentTable(model *LearnedModel, segmentTable SegmentCorrespondenceTable) *LearnedModel {
	m := *model
	m.SegmentTable = segmentTable
	return &m
}

// displacementAggregation re-aligns the corpus under the post-EM model and
// builds the feature displacement distribution from every 1-to-1 link.
func displacementAggregation(corpus []FormPair, pairWeights []float64, model *LearnedModel, maxChunkSize int) *LearnedModel {
	alignments := alignCorpus(corpus, model, maxChunkSize)

	counts := map[string]float64{}
	disp := map[string][]FeatureDisplacement{}
	total := 0.0

	for ai, alignment := range alignments {
		w := pairWeights[ai]
		if w <= 0.0 {
			continue
		}
		for _, link := range alignment.Links {
			if len(link.SourceChunk) != 1 || len(link.TargetChunk) != 1 {
				continue
			}
			d, ok := tryComputeDisplacement(link.SourceChunk[0], link.TargetChunk[0], model.FeatureSystem)
			if !ok {
				continue
			}
			k := displacementKey(d)
			counts[k] += w
			disp[k] = d
			total += w
		}
	}

	uncertainty := make(map[string]UncertaintyEstimate, len(counts))
	for k, c := range counts {
		uncertainty[k] = WilsonInterval(c, total, DefaultAlpha)
	}

	newDist := DisplacementDistribution{
		Counts:           counts,
		Total:            total,
		PriorPseudoCount: model.DisplacementDist.PriorPseudoCount,
		Uncertainty:      uncertainty,
		Disp:             disp,
	}
	m := *model
	m.DisplacementDist = newDist
	return &m
}

// tryComputeDisplacement computes a displacement, recovering any panic (the
// Python code wraps this in a broad except: continue) and returning ok=false.
func tryComputeDisplacement(src, tgt Segment, featureSystem string) (d []FeatureDisplacement, ok bool) {
	defer func() {
		if r := recover(); r != nil {
			d = nil
			ok = false
		}
	}()
	d = computeDisplacementChecked(src, tgt, featureSystem)
	return d, true
}
