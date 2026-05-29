package regulae

import (
	"errors"
	"math"
	"sort"
	"sync"
)

// Alignment search via dynamic programming. Given two forms, find the
// lowest-cost Alignment — a sequence of Links that exhaustively covers both
// forms. cost[i][j] is the minimum total cost to align source[:i] against
// target[:j]; transitions allow 1-to-1, 1-to-0, 0-to-1, and N-to-M links up to
// maxChunkSize. Ties are broken toward smaller link sizes (compositional over
// chunk interpretations) via iteration order and a tiny chunk-complexity
// penalty.

// contextFeatures is the small, hand-picked inventory used when computing a
// link's conditioning context. Larger inventories are more expressive but
// slower; this set covers the real conditioning dimensions.
var contextFeatures = map[string]struct{}{
	"vowel": {}, "consonant": {}, "front": {}, "back": {}, "close": {},
	"open": {}, "voiced": {}, "voiceless": {}, "stop": {}, "fricative": {},
	"nasal": {}, "sonorant": {}, "long": {},
}

// DefaultMaxChunkSize is the default maximum chunk size on either side of a
// link. C=3 covers almost all real phonological clusters.
const DefaultMaxChunkSize = 3

// chunkComplexityPenalty is a per-step tie-breaking penalty added to every DP
// transition, equal to chunkComplexityPenalty*(k+l-2) per link. It is ~20
// orders of magnitude below any real cost, so it only decides among
// exactly-tied alternatives, preferring decompositions into fewer/smaller
// multi-segment chunks.
const chunkComplexityPenalty = 1e-9

// AlignForms finds the lowest-cost alignment between two forms. The returned
// Alignment's source/target chunks concatenate to the respective form
// segments. A nil model selects prior-only scoring. maxChunkSize must be >= 1
// (0 means use DefaultMaxChunkSize). Returns an *UnknownGraphemeError if a
// grapheme is unrecognised.
func AlignForms(source, target Form, featureSystem string, maxChunkSize int, model *LearnedModel) (al Alignment, err error) {
	defer catchUnknownGrapheme(&err)
	if maxChunkSize == 0 {
		maxChunkSize = DefaultMaxChunkSize
	}
	if maxChunkSize < 1 {
		return Alignment{}, errors.New("max_chunk_size must be at least 1")
	}
	if featureSystem == "" {
		featureSystem = "descriptive"
	}
	al = alignForms(source, target, featureSystem, maxChunkSize, model)
	return al, nil
}

// longRangeData holds the precomputed auxiliary arrays for long-range context
// construction.
type longRangeData struct {
	leftCum     [][]FeatureConstraint
	rightCum    [][]FeatureConstraint
	sylOf       []int
	sylStarts   []int
	sylFeatures [][]FeatureConstraint
	sameSylExcl [][]FeatureConstraint
}

// alignForms is the internal, panic-on-unknown-grapheme implementation.
func alignForms(source, target Form, featureSystem string, maxChunkSize int, model *LearnedModel) Alignment {
	if model != nil {
		featureSystem = model.FeatureSystem
	}

	n := len(source.Segments)
	m := len(target.Segments)

	var sourceFeatures [][]FeatureConstraint
	var lr longRangeData
	if model != nil {
		sourceFeatures = make([][]FeatureConstraint, n)
		for idx := 0; idx < n; idx++ {
			sourceFeatures[idx] = featuresForNeighbor(source, idx, featureSystem)
		}
		lr = computeLongRangeData(source, sourceFeatures, featureSystem)
	}

	inf := math.Inf(1)
	cost := make([][]float64, n+1)
	back := make([][]*dpStep, n+1)
	for i := range cost {
		cost[i] = make([]float64, m+1)
		back[i] = make([]*dpStep, m+1)
		for j := range cost[i] {
			cost[i][j] = inf
		}
	}
	cost[0][0] = 0.0

	for i := 0; i <= n; i++ {
		for j := 0; j <= m; j++ {
			if i == 0 && j == 0 {
				continue
			}
			kMax := maxChunkSize
			if i < kMax {
				kMax = i
			}
			lMax := maxChunkSize
			if j < lMax {
				lMax = j
			}
			for k := 0; k <= kMax; k++ {
				for l := 0; l <= lMax; l++ {
					if k == 0 && l == 0 {
						continue
					}
					prev := cost[i-k][j-l]
					if prev >= inf {
						continue
					}

					srcChunk := source.Segments[i-k : i]
					tgtChunk := target.Segments[j-l : j]

					var displacement []FeatureDisplacement
					if k == 1 && l == 1 {
						displacement = computeDisplacementChecked(srcChunk[0], tgtChunk[0], featureSystem)
					}

					var linkContext Context
					if model != nil {
						linkContext = buildLinkContext(source, sourceFeatures, lr, n, i-k, i, k, l)
					}

					link := Link{
						SourceChunk:         srcChunk,
						TargetChunk:         tgtChunk,
						Context:             linkContext,
						FeatureDisplacement: displacement,
						Confidence:          1.0,
					}
					linkCost := scoreLink(link, featureSystem, model)
					total := prev + linkCost + chunkComplexityPenalty*float64(k+l-2)

					if total < cost[i][j] {
						cost[i][j] = total
						back[i][j] = &dpStep{prevI: i - k, prevJ: j - l, link: link}
					}
				}
			}
		}
	}

	var links []Link
	i, j := n, m
	for i > 0 || j > 0 {
		step := back[i][j]
		if step == nil {
			break
		}
		links = append(links, step.link)
		i, j = step.prevI, step.prevJ
	}
	// Reverse.
	for a, b := 0, len(links)-1; a < b; a, b = a+1, b-1 {
		links[a], links[b] = links[b], links[a]
	}

	return Alignment{SourceForm: source, TargetForm: target, Links: links}
}

type dpStep struct {
	prevI int
	prevJ int
	link  Link
}

// buildLinkContext constructs the conditioning Context for a link spanning
// source positions [srcStart, srcEnd) with chunk lengths (k, l).
func buildLinkContext(source Form, sourceFeatures [][]FeatureConstraint, lr longRangeData, n, srcStart, srcEnd, k, l int) Context {
	var preceding, following []FeatureConstraint
	if srcStart > 0 {
		preceding = sourceFeatures[srcStart-1]
	}
	if srcEnd < n {
		following = sourceFeatures[srcEnd]
	}
	var position string
	switch {
	case srcStart == 0:
		position = "initial"
	case srcEnd == n:
		position = "final"
	default:
		position = "medial"
	}

	var preAt, folAt []distanceConstraint
	for _, d := range [2]int{2, 3} {
		li := srcStart - d
		if li >= 0 {
			for _, fc := range sourceFeatures[li] {
				preAt = append(preAt, distanceConstraint{Offset: d, Constraint: fc})
			}
		}
		ri := srcEnd + (d - 1)
		if ri < n {
			for _, fc := range sourceFeatures[ri] {
				folAt = append(folAt, distanceConstraint{Offset: d, Constraint: fc})
			}
		}
	}

	var somewherePre, somewhereFol []FeatureConstraint
	if srcStart > 0 {
		somewherePre = lr.leftCum[srcStart]
	}
	if srcEnd < n {
		somewhereFol = lr.rightCum[srcEnd]
	}

	var sameSyl, nextSyl, prevSyl []FeatureConstraint
	if k == 1 && l == 1 && n > 0 {
		segIdx := srcStart
		sameSyl = lr.sameSylExcl[segIdx]
		sIdx := lr.sylOf[segIdx]
		if sIdx+1 < len(lr.sylFeatures) {
			nextSyl = lr.sylFeatures[sIdx+1]
		}
		if sIdx > 0 {
			prevSyl = lr.sylFeatures[sIdx-1]
		}
	}

	var selfStress, precedingStress, followingStress []FeatureConstraint
	if k == 1 && l == 1 {
		if own := source.Segments[srcStart].Stress; own != "" {
			selfStress = []FeatureConstraint{{Feature: "stress", Value: own}}
		}
		if srcStart > 0 {
			if preS := source.Segments[srcStart-1].Stress; preS != "" {
				precedingStress = []FeatureConstraint{{Feature: "stress", Value: preS}}
			}
		}
		if srcEnd < n {
			if folS := source.Segments[srcEnd].Stress; folS != "" {
				followingStress = []FeatureConstraint{{Feature: "stress", Value: folS}}
			}
		}
	}

	return Context{
		Position:            position,
		Preceding:           preceding,
		Following:           following,
		PrecedingAtDistance: preAt,
		FollowingAtDistance: folAt,
		SomewherePreceding:  somewherePre,
		SomewhereFollowing:  somewhereFol,
		SameSyllable:        sameSyl,
		NextSyllable:        nextSyl,
		PreviousSyllable:    prevSyl,
		SelfStress:          selfStress,
		PrecedingStress:     precedingStress,
		FollowingStress:     followingStress,
	}
}

// computeLinkContext builds a Context from the segments surrounding a proposed
// link spanning source [srcStart, srcEnd). It recomputes the source feature and
// long-range arrays internally; used outside the DP where those arrays are not
// already available.
func computeLinkContext(sourceForm, targetForm Form, srcStart, srcEnd, tgtStart, tgtEnd int, featureSystem string) Context {
	n := len(sourceForm.Segments)
	sourceFeatures := make([][]FeatureConstraint, n)
	for idx := 0; idx < n; idx++ {
		sourceFeatures[idx] = featuresForNeighbor(sourceForm, idx, featureSystem)
	}
	lr := computeLongRangeData(sourceForm, sourceFeatures, featureSystem)
	// Mirror the DP: syllable/stress fields only for 1-segment source spans.
	k := 0
	if srcEnd-srcStart == 1 {
		k = 1
	}
	l := k
	return buildLinkContext(sourceForm, sourceFeatures, lr, n, srcStart, srcEnd, k, l)
}

// computeLongRangeData precomputes auxiliary arrays for long-range context
// construction. Unions are sorted (by feature then value) for deterministic,
// reproducible context representations.
func computeLongRangeData(form Form, sourceFeatures [][]FeatureConstraint, featureSystem string) longRangeData {
	n := len(form.Segments)

	leftCum := make([][]FeatureConstraint, n+1)
	leftCum[0] = nil
	seenL := map[FeatureConstraint]struct{}{}
	for i := 0; i < n; i++ {
		for _, fc := range sourceFeatures[i] {
			seenL[fc] = struct{}{}
		}
		leftCum[i+1] = sortedConstraints(seenL)
	}

	rightCum := make([][]FeatureConstraint, n+1)
	seenR := map[FeatureConstraint]struct{}{}
	for i := n - 1; i >= 0; i-- {
		for _, fc := range sourceFeatures[i] {
			seenR[fc] = struct{}{}
		}
		rightCum[i] = sortedConstraints(seenR)
	}

	if n == 0 {
		return longRangeData{leftCum: leftCum, rightCum: rightCum, sylStarts: []int{0}}
	}

	breaks := ComputeSyllableBreaks(form, featureSystem)
	sylStarts := make([]int, 0, len(breaks)+2)
	sylStarts = append(sylStarts, 0)
	sylStarts = append(sylStarts, breaks...)
	sylStarts = append(sylStarts, n)
	numSyl := len(sylStarts) - 1

	sylOf := make([]int, n)
	s := 0
	for i := 0; i < n; i++ {
		for s+1 < numSyl && i >= sylStarts[s+1] {
			s++
		}
		sylOf[i] = s
	}

	sylFeatures := make([][]FeatureConstraint, 0, numSyl)
	for s := 0; s < numSyl; s++ {
		seen := map[FeatureConstraint]struct{}{}
		for i := sylStarts[s]; i < sylStarts[s+1]; i++ {
			for _, fc := range sourceFeatures[i] {
				seen[fc] = struct{}{}
			}
		}
		sylFeatures = append(sylFeatures, sortedConstraints(seen))
	}

	sameSylExcl := make([][]FeatureConstraint, 0, n)
	for i := 0; i < n; i++ {
		s := sylOf[i]
		seen := map[FeatureConstraint]struct{}{}
		for j := sylStarts[s]; j < sylStarts[s+1]; j++ {
			if j != i {
				for _, fc := range sourceFeatures[j] {
					seen[fc] = struct{}{}
				}
			}
		}
		sameSylExcl = append(sameSylExcl, sortedConstraints(seen))
	}

	return longRangeData{
		leftCum:     leftCum,
		rightCum:    rightCum,
		sylOf:       sylOf,
		sylStarts:   sylStarts,
		sylFeatures: sylFeatures,
		sameSylExcl: sameSylExcl,
	}
}

// sortedConstraints returns the constraints in a set as a slice sorted by
// (feature, value).
func sortedConstraints(set map[FeatureConstraint]struct{}) []FeatureConstraint {
	if len(set) == 0 {
		return nil
	}
	out := make([]FeatureConstraint, 0, len(set))
	for fc := range set {
		out = append(out, fc)
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a].Feature != out[b].Feature {
			return out[a].Feature < out[b].Feature
		}
		return out[a].Value < out[b].Value
	})
	return out
}

var (
	neighborCacheMu sync.Mutex
	neighborCache   = map[sonorityCacheKey][]FeatureConstraint{}
)

// featuresForNeighbor extracts the context-relevant features of the segment at
// index, as FeatureConstraints with value "+". Returns nil if the index is out
// of bounds or the grapheme is unknown. Only contextFeatures are emitted;
// results are cached by (grapheme, featureSystem).
func featuresForNeighbor(form Form, index int, featureSystem string) []FeatureConstraint {
	if index < 0 || index >= len(form.Segments) {
		return nil
	}
	grapheme := form.Segments[index].Grapheme
	key := sonorityCacheKey{grapheme: grapheme, featureSystem: featureSystem}
	neighborCacheMu.Lock()
	if cached, ok := neighborCache[key]; ok {
		neighborCacheMu.Unlock()
		return cached
	}
	neighborCacheMu.Unlock()

	feats := getFeatures(grapheme, featureSystem)
	var result []FeatureConstraint
	if feats != nil {
		var relevant []string
		for f := range feats {
			if _, ok := contextFeatures[f]; ok {
				relevant = append(relevant, f)
			}
		}
		sort.Strings(relevant)
		result = make([]FeatureConstraint, len(relevant))
		for i, f := range relevant {
			result[i] = FeatureConstraint{Feature: f, Value: "+"}
		}
	}

	neighborCacheMu.Lock()
	neighborCache[key] = result
	neighborCacheMu.Unlock()
	return result
}

// AlignmentCost returns the total cost of an alignment: the sum of its link
// costs under the given feature system and optional model, plus the
// cross-dimensional adjustment when a model is supplied. Returns an
// *UnknownGraphemeError if a grapheme is unrecognised.
func AlignmentCost(alignment Alignment, featureSystem string, model *LearnedModel) (cost float64, err error) {
	defer catchUnknownGrapheme(&err)
	if model != nil {
		featureSystem = model.FeatureSystem
	}
	if featureSystem == "" {
		featureSystem = "descriptive"
	}
	cost = alignmentCost(alignment, featureSystem, model)
	return cost, nil
}

// alignmentCost is the internal, panic-on-unknown-grapheme implementation.
func alignmentCost(alignment Alignment, featureSystem string, model *LearnedModel) float64 {
	base := 0.0
	for _, link := range alignment.Links {
		base += scoreLink(link, featureSystem, model)
	}
	if model != nil {
		base += applyCrossDimensionalAdjustments(alignment, model)
	}
	return base
}
