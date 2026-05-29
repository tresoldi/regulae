package regulae

import "math"

// Chunk promotion: extract candidate multi-segment correspondences from corpus
// alignments and promote those that improve BIC. Each candidate's BIC delta is
// evaluated independently (the compositional baseline uses only the segment
// table, unchanged during this phase). Optionally screens promoted chunks by
// their transparency score.

// chunkCandidates holds candidate chunk counts plus their structured pairs,
// keyed by chunkPairKey.
type chunkCandidates struct {
	counts map[string]float64
	pairs  map[string]ChunkPair
}

// chunkPromotion returns a new model with the chunk table populated by the
// BIC-promoted chunks (optionally filtered by transparency).
func chunkPromotion(corpus []FormPair, pairWeights []float64, model *LearnedModel, maxChunkSize int, chunkMinTransparency float64, bicConfig BICConfig) *LearnedModel {
	minChunkObs := float64(bicConfig.MinChunkObservations)

	alignments := alignCorpus(corpus, model, maxChunkSize)
	candidates := extractChunkCandidates(alignments, pairWeights, maxChunkSize)

	// 1-to-1 observation mass as the BIC N proxy.
	nObservations := 0.0
	for ai, a := range alignments {
		w := pairWeights[ai]
		if w <= 0.0 {
			continue
		}
		for _, link := range a.Links {
			if len(link.SourceChunk) == 1 && len(link.TargetChunk) == 1 {
				nObservations += w
			}
		}
	}
	if nObservations <= 0 {
		nObservations = 1
	}

	promoted := map[string]float64{}
	promotedCounts := map[string]float64{}
	promotedPairs := map[string]ChunkPair{}
	for k, nC := range candidates.counts {
		if nC < minChunkObs {
			continue
		}
		pair := candidates.pairs[k]
		compCostPer := compositionalChunkCostRaw(pair.Src, pair.Tgt, model)
		promotedCostPer := promotedChunkCost(pair.Src, pair.Tgt, candidates, 1.0)
		if math.IsInf(compCostPer, 1) || math.IsInf(promotedCostPer, 1) {
			continue
		}
		reduction := nC * (compCostPer - promotedCostPer)
		kParams := maxInt(len(pair.Src), len(pair.Tgt))
		deltaBIC := -2.0*reduction + float64(kParams)*math.Log(nObservations)
		if deltaBIC < 0.0 {
			logZSum := 0.0
			for _, s := range pair.Src {
				logZSum += model.SegmentTable.LogNormalizers[s.Grapheme]
			}
			promoted[k] = promotedCostPer - logZSum
			promotedCounts[k] = nC
			promotedPairs[k] = pair
		}
	}

	chunkUncertainty := make(map[string]UncertaintyEstimate, len(promotedCounts))
	for k, c := range promotedCounts {
		chunkUncertainty[k] = WilsonInterval(c, nObservations, DefaultAlpha)
	}

	// Provisional model so the transparency analyzer can run against the
	// promoted entries.
	provisional := *model
	provisional.ChunkTable = ChunkPhraseTable{
		Entries:           promoted,
		ObservationCounts: promotedCounts,
		Uncertainty:       chunkUncertainty,
		Diagnostics:       map[string]ChunkTransparencyReport{},
		Pairs:             promotedPairs,
	}

	var reports []ChunkTransparencyReport
	if len(promoted) > 0 {
		reports = analyzePromotedChunks(&provisional)
	}

	final := NewChunkPhraseTable()
	for _, report := range reports {
		if report.TransparencyScore < chunkMinTransparency {
			continue
		}
		k := chunkPairKey(report.SrcChunk, report.TgtChunk)
		final.Entries[k] = promoted[k]
		final.Diagnostics[k] = report
		final.Pairs[k] = ChunkPair{Src: report.SrcChunk, Tgt: report.TgtChunk}
	}

	out := provisional
	out.ChunkTable = final
	return &out
}

// extractChunkCandidates enumerates contiguous sub-alignments as candidate
// chunks: both sides non-empty, within maxChunkSize, combined length >= 3, and
// not straddling a morpheme boundary on either side.
func extractChunkCandidates(alignments []Alignment, pairWeights []float64, maxChunkSize int) chunkCandidates {
	counts := map[string]float64{}
	pairs := map[string]ChunkPair{}
	for ai, alignment := range alignments {
		w := pairWeights[ai]
		if w <= 0.0 {
			continue
		}
		links := alignment.Links
		srcStarts := make([]int, len(links))
		srcEnds := make([]int, len(links))
		tgtStarts := make([]int, len(links))
		tgtEnds := make([]int, len(links))
		sPos, tPos := 0, 0
		for i, link := range links {
			srcStarts[i] = sPos
			tgtStarts[i] = tPos
			sPos += len(link.SourceChunk)
			tPos += len(link.TargetChunk)
			srcEnds[i] = sPos
			tgtEnds[i] = tPos
		}
		srcBreaks := intSet(alignment.SourceForm.MorphemeBreaks)
		tgtBreaks := intSet(alignment.TargetForm.MorphemeBreaks)
		for i := range links {
			var srcChunk, tgtChunk []Segment
			for j := i; j < len(links); j++ {
				srcChunk = append(srcChunk, links[j].SourceChunk...)
				tgtChunk = append(tgtChunk, links[j].TargetChunk...)
				if len(srcChunk) > maxChunkSize || len(tgtChunk) > maxChunkSize {
					break
				}
				if len(srcChunk) == 0 || len(tgtChunk) == 0 {
					continue
				}
				if len(srcChunk)+len(tgtChunk) < 3 {
					continue
				}
				if spansBreak(srcStarts[i], srcEnds[j], srcBreaks) {
					continue
				}
				if spansBreak(tgtStarts[i], tgtEnds[j], tgtBreaks) {
					continue
				}
				// Copy the chunk slices so distinct candidates don't share a
				// growing backing array.
				sc := append([]Segment(nil), srcChunk...)
				tc := append([]Segment(nil), tgtChunk...)
				k := chunkPairKey(sc, tc)
				counts[k] += w
				pairs[k] = ChunkPair{Src: sc, Tgt: tc}
			}
		}
	}
	return chunkCandidates{counts: counts, pairs: pairs}
}

func intSet(xs []int) map[int]struct{} {
	if len(xs) == 0 {
		return nil
	}
	out := make(map[int]struct{}, len(xs))
	for _, x := range xs {
		out[x] = struct{}{}
	}
	return out
}

// spansBreak reports whether any boundary index strictly inside (start, end) is
// present in breaks. A boundary at exactly start or end is on the edge.
func spansBreak(start, end int, breaks map[int]struct{}) bool {
	if len(breaks) == 0 {
		return false
	}
	for b := range breaks {
		if start < b && b < end {
			return true
		}
	}
	return false
}

// compositionalChunkCostRaw is the raw compositional cost of a chunk — the
// negative log probability of generating it under independent segment-level
// draws, using the best sub-alignment decomposition. No log Z offset.
func compositionalChunkCostRaw(srcChunk, tgtChunk []Segment, model *LearnedModel) float64 {
	noChunks := *model
	noChunks.ChunkTable = NewChunkPhraseTable()
	subSrc := Form{LectID: "_sub_src", Segments: srcChunk}
	subTgt := Form{LectID: "_sub_tgt", Segments: tgtChunk}
	sub := alignForms(subSrc, subTgt, noChunks.FeatureSystem, 1, &noChunks)

	table := model.SegmentTable
	concentration := model.Concentration
	vocabSize := maxInt(len(table.LogNormalizers), 1)
	gapCostPerSegment := 1.0
	if vocabSize > 1 {
		gapCostPerSegment = math.Log(float64(vocabSize))
	}

	cost := 0.0
	for _, link := range sub.Links {
		sc := link.SourceChunk
		tc := link.TargetChunk
		if len(sc) == 1 && len(tc) == 1 {
			p, ok := segmentPosterior(sc[0].Grapheme, tc[0].Grapheme, Context{}, table, concentration)
			if !ok || p <= 0.0 {
				return math.Inf(1)
			}
			cost += -math.Log(p)
		} else {
			cost += gapCostPerSegment * float64(maxInt(len(sc), len(tc)))
		}
	}
	return cost
}

// promotedChunkCost is the MLE chunk cost with Laplace smoothing: -log P(tgt |
// src), estimated from candidate counts.
func promotedChunkCost(srcChunk, tgtChunk []Segment, candidates chunkCandidates, alpha float64) float64 {
	srcTotal := 0.0
	vSrc := 0
	var nC float64
	tgtKey := chunkPairKey(srcChunk, tgtChunk)
	for k, n := range candidates.counts {
		pair := candidates.pairs[k]
		if segmentSlicesEqualCD(pair.Src, srcChunk) {
			srcTotal += n
			vSrc++
			if k == tgtKey {
				nC = n
			}
		}
	}
	if vSrc == 0 {
		return math.Inf(1)
	}
	prob := (nC + alpha) / (srcTotal + alpha*float64(vSrc))
	if prob <= 0.0 {
		return math.Inf(1)
	}
	return -math.Log(prob)
}
