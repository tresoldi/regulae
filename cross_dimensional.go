package regulae

import (
	"math"
	"sort"
	"strconv"
	"strings"
)

// Cross-dimensional correspondence discovery from residual mutual information.
// Runs after tonal aggregation: it asks anomaly detection for ranked candidate
// hypotheses, tests each via an exact BIC comparison (re-scoring the whole
// corpus with the candidate rule added), and greedily commits the single
// best-scoring hypothesis per iteration, up to CrossDimMaxIterations.

const (
	crossDimMaxIterations     = 5
	crossDimMinRuleCount      = 3
	crossDimMinRuleConfidence = 0.5
)

func crossDimensionalDiscovery(corpus []FormPair, pairWeights []float64, model *LearnedModel, randomSeed int, bicConfig BICConfig) *LearnedModel {
	maxIterations := bicConfig.CrossDimMaxIterations
	minRuleCount := float64(bicConfig.CrossDimMinRuleCount)
	minRuleConfidence := bicConfig.CrossDimMinRuleConfidence
	deltaThreshold := bicConfig.DeltaBICThreshold

	current := model
	for iter := 0; iter < maxIterations; iter++ {
		hypotheses := findResidualPatterns(corpus, current, 5, 100, 0.95, "", randomSeed, []string{"cross_dimensional_tonogenesis"})
		if len(hypotheses) == 0 {
			break
		}

		baselineCost := totalCorpusCost(corpus, pairWeights, current)
		nObs := count1to1Observations(corpus, pairWeights, current)
		if nObs <= 0 {
			break
		}
		lnN := math.Log(nObs)

		committedSignatures := map[string]bool{}
		for _, entry := range current.CrossDimensionalTable.Entries {
			committedSignatures[ruleSignature(entry)] = true
		}

		var bestLink *CrossDimensionalLink
		bestDeltaBIC := deltaThreshold
		for _, hyp := range hypotheses {
			trialLink := hypothesisToCrossDimensionalLink(hyp, corpus, pairWeights, current)
			if trialLink == nil {
				continue
			}
			if committedSignatures[ruleSignature(*trialLink)] {
				continue
			}
			if trialLink.Count < minRuleCount {
				continue
			}
			if trialLink.Confidence < minRuleConfidence {
				continue
			}

			trialEntries := append(append([]CrossDimensionalLink(nil), current.CrossDimensionalTable.Entries...), *trialLink)
			trialModel := *current
			trialModel.CrossDimensionalTable = CrossDimensionalLinkTable{Entries: trialEntries}
			trialCost := totalCorpusCost(corpus, pairWeights, &trialModel)
			reduction := baselineCost - trialCost
			kParams := 1.0
			if trialLink.SrcFeature2 != nil {
				kParams = 2.0
			}
			deltaBIC := -2.0*reduction + kParams*lnN
			if deltaBIC < bestDeltaBIC {
				bestDeltaBIC = deltaBIC
				bl := *trialLink
				bestLink = &bl
			}
		}

		if bestLink == nil {
			break
		}

		newEntries := append(append([]CrossDimensionalLink(nil), current.CrossDimensionalTable.Entries...), *bestLink)
		m := *current
		m.CrossDimensionalTable = CrossDimensionalLinkTable{Entries: newEntries}
		current = &m
	}

	deduped := dedupDualFramings(current.CrossDimensionalTable.Entries)
	if len(deduped) != len(current.CrossDimensionalTable.Entries) {
		m := *current
		m.CrossDimensionalTable = CrossDimensionalLinkTable{Entries: deduped}
		current = &m
	}
	return current
}

// dedupDualFramings collapses committed rules that describe the same underlying
// pattern from different link-position perspectives, keeping the leftmost
// (smallest src_offset) framing, ties broken by first-committed order.
func dedupDualFramings(entries []CrossDimensionalLink) []CrossDimensionalLink {
	type member struct {
		srcOff int
		idx    int
	}
	groups := map[string][]member{}
	for idx, entry := range entries {
		srcOff := parseRelativePositionOffset(entry.SrcPosition)
		delta := entry.TgtPositionOffset - srcOff
		var key string
		if entry.SrcFeature2 == nil || entry.SrcPosition2 == "" {
			key = strings.Join([]string{
				entry.SrcFeature.Feature, entry.SrcFeature.Value,
				entry.TgtDimension, entry.TgtValue,
				strconv.Itoa(delta), "", "", "",
			}, "\x1f")
		} else {
			srcOff2 := parseRelativePositionOffset(entry.SrcPosition2)
			delta2 := entry.TgtPositionOffset - srcOff2
			key = strings.Join([]string{
				entry.SrcFeature.Feature, entry.SrcFeature.Value,
				entry.TgtDimension, entry.TgtValue,
				strconv.Itoa(delta),
				entry.SrcFeature2.Feature, entry.SrcFeature2.Value,
				strconv.Itoa(delta2),
			}, "\x1f")
		}
		groups[key] = append(groups[key], member{srcOff: srcOff, idx: idx})
	}
	chosen := map[int]bool{}
	for _, members := range groups {
		sort.SliceStable(members, func(a, b int) bool {
			if members[a].srcOff != members[b].srcOff {
				return members[a].srcOff < members[b].srcOff
			}
			return members[a].idx < members[b].idx
		})
		chosen[members[0].idx] = true
	}
	var out []CrossDimensionalLink
	for i, e := range entries {
		if chosen[i] {
			out = append(out, e)
		}
	}
	return out
}

// totalCorpusCost is the sum of per-pair alignment costs (including the
// cross-dimensional overlay) over the whole corpus.
func totalCorpusCost(corpus []FormPair, pairWeights []float64, model *LearnedModel) float64 {
	total := 0.0
	for i, p := range corpus {
		w := pairWeights[i]
		if w <= 0.0 {
			continue
		}
		alignment := alignForms(p.Src, p.Tgt, model.FeatureSystem, DefaultMaxChunkSize, model)
		total += w * alignmentCost(alignment, model.FeatureSystem, model)
	}
	return total
}

// count1to1Observations is the total 1-to-1 link observation mass (the BIC N).
func count1to1Observations(corpus []FormPair, pairWeights []float64, model *LearnedModel) float64 {
	count := 0.0
	for i, p := range corpus {
		w := pairWeights[i]
		if w <= 0.0 {
			continue
		}
		alignment := alignForms(p.Src, p.Tgt, model.FeatureSystem, DefaultMaxChunkSize, model)
		for _, link := range alignment.Links {
			if len(link.SourceChunk) == 1 && len(link.TargetChunk) == 1 {
				count += w
			}
		}
	}
	return count
}

// hypothesisToCrossDimensionalLink builds a CrossDimensionalLink from a
// hypothesis by walking the corpus under the model and counting actual rule
// matches. Returns nil if there is no supporting (or no positive) observation.
func hypothesisToCrossDimensionalLink(hypothesis PatternHypothesis, corpus []FormPair, pairWeights []float64, model *LearnedModel) *CrossDimensionalLink {
	srcOffset := parseRelativePositionOffset(hypothesis.SrcPositionSpec)
	tgtOffset := hypothesis.TgtPositionOffset
	featureName := hypothesis.SrcFeature
	srcValue := hypothesis.SrcValue
	tgtValue := hypothesis.TgtFeatureOrValue
	tgtDimension := hypothesis.TgtDimension
	featureSystem := model.FeatureSystem

	isJoint := hypothesis.SrcFeature2 != nil && hypothesis.SrcPositionSpec2 != nil
	var srcOffset2 int
	var featureName2, srcValue2 string
	if isJoint {
		srcOffset2 = parseRelativePositionOffset(*hypothesis.SrcPositionSpec2)
		featureName2 = *hypothesis.SrcFeature2
		srcValue2 = "+"
		if hypothesis.SrcValue2 != nil && *hypothesis.SrcValue2 != "" {
			srcValue2 = *hypothesis.SrcValue2
		}
	}

	predicateHolds := func(segForm Form, segIdx int, name, value string) bool {
		if segIdx < 0 || segIdx >= len(segForm.Segments) {
			return false
		}
		seg := segForm.Segments[segIdx]
		if name == "tone" {
			return seg.Tone == value
		}
		if seg.Grapheme == "" {
			return false
		}
		features := getFeatures(seg.Grapheme, featureSystem)
		return features != nil && features.has(name)
	}

	srcCount := 0.0
	count := 0.0
	for i, p := range corpus {
		w := pairWeights[i]
		if w <= 0.0 {
			continue
		}
		alignment := alignForms(p.Src, p.Tgt, model.FeatureSystem, DefaultMaxChunkSize, model)
		linkSrcPos := 0
		linkTgtPos := 0
		for _, link := range alignment.Links {
			srcLen := len(link.SourceChunk)
			tgtLen := len(link.TargetChunk)
			if srcLen == 1 && tgtLen == 1 {
				holds := predicateHolds(p.Src, linkSrcPos+srcOffset, featureName, srcValue)
				if holds && isJoint {
					holds = predicateHolds(p.Src, linkSrcPos+srcOffset2, featureName2, srcValue2)
				}
				if holds {
					srcCount += w
					tgtIdx := linkTgtPos + tgtOffset
					if tgtIdx >= 0 && tgtIdx < len(p.Tgt.Segments) {
						tgtSeg := p.Tgt.Segments[tgtIdx]
						var actual string
						switch tgtDimension {
						case "tone":
							actual = tgtSeg.Tone
						case "length":
							actual = tgtSeg.Length
						case "stress":
							actual = tgtSeg.Stress
						}
						if actual == tgtValue {
							count += w
						}
					}
				}
			}
			linkSrcPos += srcLen
			linkTgtPos += tgtLen
		}
	}

	if srcCount == 0 {
		return nil
	}
	if count == 0 {
		return nil
	}

	var baseFeature2 *FeatureConstraint
	var basePosition2 string
	if isJoint && featureName2 != "" {
		fc := FeatureConstraint{Feature: featureName2, Value: srcValue2}
		baseFeature2 = &fc
		basePosition2 = *hypothesis.SrcPositionSpec2
	}
	u := WilsonInterval(count, srcCount, DefaultAlpha)
	return &CrossDimensionalLink{
		SrcFeature:        FeatureConstraint{Feature: featureName, Value: srcValue},
		SrcPosition:       hypothesis.SrcPositionSpec,
		TgtDimension:      tgtDimension,
		TgtValue:          tgtValue,
		TgtPositionOffset: tgtOffset,
		Count:             count,
		SrcCount:          srcCount,
		Confidence:        count / srcCount,
		SrcFeature2:       baseFeature2,
		SrcPosition2:      basePosition2,
		Uncertainty:       &u,
	}
}

// parseRelativePositionOffset parses "relative_-1" etc.; panics on a malformed
// spec (an invariant violation).
func parseRelativePositionOffset(spec string) int {
	if !strings.HasPrefix(spec, "relative_") {
		panic("unrecognised src_position spec: " + spec)
	}
	n, err := strconv.Atoi(spec[len("relative_"):])
	if err != nil {
		panic("unrecognised src_position spec: " + spec)
	}
	return n
}

// ruleSignature returns a hashable signature used to dedup cross-dimensional
// commits across iterations.
func ruleSignature(rule CrossDimensionalLink) string {
	parts := []string{
		rule.SrcFeature.Feature, rule.SrcFeature.Value, rule.SrcPosition,
		rule.TgtDimension, rule.TgtValue, strconv.Itoa(rule.TgtPositionOffset),
	}
	if rule.SrcFeature2 == nil {
		parts = append(parts, "", "", "")
	} else {
		parts = append(parts, rule.SrcFeature2.Feature, rule.SrcFeature2.Value, rule.SrcPosition2)
	}
	return strings.Join(parts, "\x1f")
}
