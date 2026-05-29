package regulae

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// Residual-mutual-information anomaly detection. Computes residual MI between
// feature pairs in training-corpus alignments, conditional on what the model
// already explains, and returns ranked PatternHypothesis candidates that the
// cross-dimensional discovery loop consumes. Pure diagnostic: never mutates the
// model or corpus.

// HypothesisKinds are the hypothesis kinds reported by FindResidualPatterns.
// Currently only cross-dimensional tonogenesis (a segmental feature at one
// position conditioning a tonal value at another).
var HypothesisKinds = []string{"cross_dimensional_tonogenesis"}

// PatternHypothesis is a residual-MI-based hypothesis about unmodelled
// structure. SrcFeature "tone" is special (checks Segment.Tone, with SrcValue
// the tone to match). SrcFeature2/SrcValue2/SrcPositionSpec2 (nil = absent)
// describe a conjunctive joint-predictor rule.
type PatternHypothesis struct {
	Kind                   string
	SrcFeature             string
	SrcPositionSpec        string
	TgtFeatureOrValue      string
	TgtDimension           string
	TgtPositionOffset      int
	ResidualMI             float64
	NullPercentile         float64
	SupportingObservations int
	SrcValue               string

	SrcFeature2      *string
	SrcValue2        *string
	SrcPositionSpec2 *string
}

// anomalyObservation is one 1-to-1 link observation harvested from a trained
// alignment.
type anomalyObservation struct {
	srcForm Form
	srcPos  int
	tgtForm Form
	tgtPos  int
}

// findResidualPatterns ranks suspicious feature-pair combinations by residual
// MI. Returns hypotheses sorted by residual MI descending. The permutation
// null is seeded by randomSeed for deterministic results. Panics on unknown
// graphemes (internal use).
func findResidualPatterns(corpus []FormPair, model *LearnedModel, minObservations, nullPermutations int, nullPercentileThreshold float64, featureSystem string, randomSeed int, kinds []string) []PatternHypothesis {
	fs := featureSystem
	if fs == "" {
		fs = model.FeatureSystem
	}
	selected := map[string]bool{}
	if kinds == nil {
		for _, k := range HypothesisKinds {
			selected[k] = true
		}
	} else {
		known := map[string]bool{}
		for _, k := range HypothesisKinds {
			known[k] = true
		}
		for _, k := range kinds {
			if !known[k] {
				panic(fmt.Sprintf("unknown hypothesis kind: %q", k))
			}
			selected[k] = true
		}
	}

	observations := collectAnomalyObservations(corpus, model)
	if len(observations) == 0 {
		return nil
	}

	rng := rand.New(rand.NewSource(int64(randomSeed)))
	var hypotheses []PatternHypothesis

	if selected["cross_dimensional_tonogenesis"] {
		hypotheses = append(hypotheses, findCrossDimensionalTonogenesis(observations, fs, rng, minObservations, nullPermutations, nullPercentileThreshold)...)
	}

	sort.SliceStable(hypotheses, func(a, b int) bool {
		ha, hb := hypotheses[a], hypotheses[b]
		if ha.ResidualMI != hb.ResidualMI {
			return ha.ResidualMI > hb.ResidualMI
		}
		if ha.Kind != hb.Kind {
			return ha.Kind < hb.Kind
		}
		if ha.SrcFeature != hb.SrcFeature {
			return ha.SrcFeature < hb.SrcFeature
		}
		if ha.SrcPositionSpec != hb.SrcPositionSpec {
			return ha.SrcPositionSpec < hb.SrcPositionSpec
		}
		return ha.TgtFeatureOrValue < hb.TgtFeatureOrValue
	})
	return hypotheses
}

// sortedContextFeatures returns the context-feature inventory in sorted order
// (the Python set iteration order is irrelevant since hypotheses are sorted).
func sortedContextFeatures() []string {
	out := make([]string, 0, len(contextFeatures))
	for f := range contextFeatures {
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}

func findCrossDimensionalTonogenesis(observations []anomalyObservation, featureSystem string, rng *rand.Rand, minObservations, nullPermutations int, nullPercentileThreshold float64) []PatternHypothesis {
	tgtToneSet := map[string]struct{}{}
	for _, obs := range observations {
		for _, seg := range obs.tgtForm.Segments {
			if seg.Tone != "" {
				tgtToneSet[seg.Tone] = struct{}{}
			}
		}
	}
	if len(tgtToneSet) == 0 {
		return nil
	}
	sortedTgtTones := sortedStringSet(tgtToneSet)

	srcToneSet := map[string]struct{}{}
	for _, obs := range observations {
		for _, seg := range obs.srcForm.Segments {
			if seg.Tone != "" {
				srcToneSet[seg.Tone] = struct{}{}
			}
		}
	}
	sortedSrcTones := sortedStringSet(srcToneSet)

	features := sortedContextFeatures()
	var hypotheses []PatternHypothesis

	// Segmental source predictors.
	for _, srcOffset := range []int{-1, 0, 1} {
		for _, feature := range features {
			for _, tgtOffset := range []int{0, 1, 2} {
				for _, toneValue := range sortedTgtTones {
					var xs, ys []int
					for _, obs := range observations {
						x, xok := segmentHasFeature(obs.srcForm, obs.srcPos+srcOffset, feature, featureSystem)
						yTone, yok := segmentTone(obs.tgtForm, obs.tgtPos+tgtOffset)
						if !xok || !yok {
							continue
						}
						xs = append(xs, b2i(x))
						ys = append(ys, b2i(yTone == toneValue))
					}
					if len(xs) < minObservations {
						continue
					}
					observedMI := binaryMutualInformation(xs, ys)
					if observedMI <= 0.0 {
						continue
					}
					nullScores := permutationNullMI(xs, ys, rng, nullPermutations)
					percentile := fractionBelow(observedMI, nullScores)
					if percentile < nullPercentileThreshold {
						continue
					}
					hypotheses = append(hypotheses, PatternHypothesis{
						Kind:                   "cross_dimensional_tonogenesis",
						SrcFeature:             feature,
						SrcValue:               "+",
						SrcPositionSpec:        formatOffset(srcOffset),
						TgtFeatureOrValue:      toneValue,
						TgtDimension:           "tone",
						TgtPositionOffset:      tgtOffset,
						ResidualMI:             observedMI,
						NullPercentile:         percentile,
						SupportingObservations: len(xs),
					})
				}
			}
		}
	}

	// Source-tone predictors.
	for _, srcOffset := range []int{-1, 0, 1} {
		for _, srcToneValue := range sortedSrcTones {
			for _, tgtOffset := range []int{0, 1, 2} {
				for _, tgtToneValue := range sortedTgtTones {
					if srcOffset == 0 && tgtOffset == 0 && srcToneValue == tgtToneValue {
						continue
					}
					var xs, ys []int
					for _, obs := range observations {
						srcTone, sok := segmentTone(obs.srcForm, obs.srcPos+srcOffset)
						tgtTone, tok := segmentTone(obs.tgtForm, obs.tgtPos+tgtOffset)
						if !sok || !tok {
							continue
						}
						xs = append(xs, b2i(srcTone == srcToneValue))
						ys = append(ys, b2i(tgtTone == tgtToneValue))
					}
					if len(xs) < minObservations {
						continue
					}
					observedMI := binaryMutualInformation(xs, ys)
					if observedMI <= 0.0 {
						continue
					}
					nullScores := permutationNullMI(xs, ys, rng, nullPermutations)
					percentile := fractionBelow(observedMI, nullScores)
					if percentile < nullPercentileThreshold {
						continue
					}
					hypotheses = append(hypotheses, PatternHypothesis{
						Kind:                   "cross_dimensional_tonogenesis",
						SrcFeature:             "tone",
						SrcValue:               srcToneValue,
						SrcPositionSpec:        formatOffset(srcOffset),
						TgtFeatureOrValue:      tgtToneValue,
						TgtDimension:           "tone",
						TgtPositionOffset:      tgtOffset,
						ResidualMI:             observedMI,
						NullPercentile:         percentile,
						SupportingObservations: len(xs),
					})
				}
			}
		}
	}

	// Joint predictors (segmental feature, source tone value).
	const jointMargin = 0.05
	if len(sortedSrcTones) > 0 {
		for _, segOffset := range []int{-1, 0, 1} {
			for _, segFeature := range features {
				for _, toneOffset := range []int{-1, 0, 1} {
					for _, srcToneValue := range sortedSrcTones {
						for _, tgtOffset := range []int{0, 1, 2} {
							for _, tgtToneValue := range sortedTgtTones {
								var xs, xsA, xsB, ys []int
								for _, obs := range observations {
									hasFeat, fok := segmentHasFeature(obs.srcForm, obs.srcPos+segOffset, segFeature, featureSystem)
									srcTone, sok := segmentTone(obs.srcForm, obs.srcPos+toneOffset)
									tgtTone, tok := segmentTone(obs.tgtForm, obs.tgtPos+tgtOffset)
									if !fok || !sok || !tok {
										continue
									}
									a := b2i(hasFeat)
									bb := b2i(srcTone == srcToneValue)
									xsA = append(xsA, a)
									xsB = append(xsB, bb)
									xs = append(xs, b2i(a == 1 && bb == 1))
									ys = append(ys, b2i(tgtTone == tgtToneValue))
								}
								if len(xs) < minObservations {
									continue
								}
								if sumInts(xs) < minObservations {
									continue
								}
								confJoint := float64(countBoth(xs, ys)) / float64(maxInt(sumInts(xs), 1))
								nA := sumInts(xsA)
								nB := sumInts(xsB)
								confA := 0.0
								if nA > 0 {
									confA = float64(countBoth(xsA, ys)) / float64(nA)
								}
								confB := 0.0
								if nB > 0 {
									confB = float64(countBoth(xsB, ys)) / float64(nB)
								}
								if confJoint < confA+jointMargin || confJoint < confB+jointMargin {
									continue
								}
								observedMI := binaryMutualInformation(xs, ys)
								if observedMI <= 0.0 {
									continue
								}
								nullScores := permutationNullMI(xs, ys, rng, nullPermutations)
								percentile := fractionBelow(observedMI, nullScores)
								if percentile < nullPercentileThreshold {
									continue
								}
								tone := "tone"
								srcTV := srcToneValue
								spec2 := formatOffset(toneOffset)
								hypotheses = append(hypotheses, PatternHypothesis{
									Kind:                   "cross_dimensional_tonogenesis",
									SrcFeature:             segFeature,
									SrcValue:               "+",
									SrcPositionSpec:        formatOffset(segOffset),
									SrcFeature2:            &tone,
									SrcValue2:              &srcTV,
									SrcPositionSpec2:       &spec2,
									TgtFeatureOrValue:      tgtToneValue,
									TgtDimension:           "tone",
									TgtPositionOffset:      tgtOffset,
									ResidualMI:             observedMI,
									NullPercentile:         percentile,
									SupportingObservations: len(xs),
								})
							}
						}
					}
				}
			}
		}
	}

	return hypotheses
}

func collectAnomalyObservations(corpus []FormPair, model *LearnedModel) []anomalyObservation {
	var observations []anomalyObservation
	for _, p := range corpus {
		alignment := alignForms(p.Src, p.Tgt, model.FeatureSystem, DefaultMaxChunkSize, model)
		srcPos := 0
		tgtPos := 0
		for _, link := range alignment.Links {
			srcLen := len(link.SourceChunk)
			tgtLen := len(link.TargetChunk)
			if srcLen == 1 && tgtLen == 1 {
				observations = append(observations, anomalyObservation{srcForm: p.Src, srcPos: srcPos, tgtForm: p.Tgt, tgtPos: tgtPos})
			}
			srcPos += srcLen
			tgtPos += tgtLen
		}
	}
	return observations
}

// segmentHasFeature reports whether the segment at pos has feature. The bool
// ok is false if the position is out of bounds or the grapheme is unknown (the
// MI calculation skips such observations).
func segmentHasFeature(form Form, pos int, feature, featureSystem string) (val bool, ok bool) {
	if pos < 0 || pos >= len(form.Segments) {
		return false, false
	}
	grapheme := form.Segments[pos].Grapheme
	if grapheme == "" {
		return false, false
	}
	features := getFeatures(grapheme, featureSystem)
	if features == nil {
		return false, false
	}
	return features.has(feature), true
}

// segmentTone returns the tone at pos; ok is false if out of bounds or untoned.
func segmentTone(form Form, pos int) (string, bool) {
	if pos < 0 || pos >= len(form.Segments) {
		return "", false
	}
	tone := form.Segments[pos].Tone
	if tone == "" {
		return "", false
	}
	return tone, true
}

// binaryMutualInformation returns the MI (nats) between two parallel binary
// sequences.
func binaryMutualInformation(xs, ys []int) float64 {
	n := len(xs)
	if n == 0 {
		return 0.0
	}
	var c00, c01, c10, c11 int
	for i := range xs {
		if xs[i] == 0 {
			if ys[i] == 0 {
				c00++
			} else {
				c01++
			}
		} else {
			if ys[i] == 0 {
				c10++
			} else {
				c11++
			}
		}
	}
	fn := float64(n)
	p00 := float64(c00) / fn
	p01 := float64(c01) / fn
	p10 := float64(c10) / fn
	p11 := float64(c11) / fn
	px0 := p00 + p01
	px1 := p10 + p11
	py0 := p00 + p10
	py1 := p01 + p11
	mi := 0.0
	for _, t := range [][3]float64{{p00, px0, py0}, {p01, px0, py1}, {p10, px1, py0}, {p11, px1, py1}} {
		pXY, pX, pY := t[0], t[1], t[2]
		if pXY > 0.0 && pX > 0.0 && pY > 0.0 {
			mi += pXY * math.Log(pXY/(pX*pY))
		}
	}
	return mi
}

// permutationNullMI computes the permutation null distribution of MI by
// repeatedly shuffling the target side and recomputing MI. The shuffles are
// cumulative (the same slice is reshuffled each iteration), matching the
// Python implementation. Returns a sorted list of null MI values.
func permutationNullMI(xs, ys []int, rng *rand.Rand, permutations int) []float64 {
	scores := make([]float64, 0, permutations)
	shuffled := append([]int(nil), ys...)
	for i := 0; i < permutations; i++ {
		rng.Shuffle(len(shuffled), func(a, b int) {
			shuffled[a], shuffled[b] = shuffled[b], shuffled[a]
		})
		scores = append(scores, binaryMutualInformation(xs, shuffled))
	}
	sort.Float64s(scores)
	return scores
}

// fractionBelow returns the fraction of sortedScores strictly less than value.
func fractionBelow(value float64, sortedScores []float64) float64 {
	if len(sortedScores) == 0 {
		return 0.0
	}
	below := 0
	for _, s := range sortedScores {
		if s < value {
			below++
		} else {
			break
		}
	}
	return float64(below) / float64(len(sortedScores))
}

func formatOffset(offset int) string {
	if offset == 0 {
		return "relative_0"
	}
	return fmt.Sprintf("relative_%+d", offset)
}

func sortedStringSet(set map[string]struct{}) []string {
	out := make([]string, 0, len(set))
	for s := range set {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

func sumInts(xs []int) int {
	total := 0
	for _, x := range xs {
		total += x
	}
	return total
}

func countBoth(xs, ys []int) int {
	count := 0
	for i := range xs {
		if xs[i] == 1 && ys[i] == 1 {
			count++
		}
	}
	return count
}
