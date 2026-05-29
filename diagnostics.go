package regulae

import (
	"math"
	"sort"
)

// Diagnostic helpers over a trained MultiLectModel. These consume a trained
// model and emit human-oriented reports; nothing here feeds back into training,
// scoring, or alignment.

// CognateOutlierReport is one row of a cognate outlier ranking. ZScore is how
// many standard deviations above the corpus mean the length-normalized
// alignment cost sits; positive values (especially > 2) are the outliers.
type CognateOutlierReport struct {
	CognateID      string
	NPairs         int
	CostPerSegment float64
	ZScore         float64
}

// pairCostPerSegment computes the length-normalized alignment cost for one
// pair (total cost divided by the mean form length).
func pairCostPerSegment(formA, formB Form, model *LearnedModel, maxChunkSize int) float64 {
	alignment := alignForms(formA, formB, model.FeatureSystem, maxChunkSize, model)
	cost := alignmentCost(alignment, model.FeatureSystem, model)
	denom := float64(len(formA.Segments)+len(formB.Segments)) / 2.0
	if denom <= 0.0 {
		return 0.0
	}
	return cost / denom
}

// FindCognateOutliers ranks cognate sets by alignment-cost anomaly (z-score of
// the per-segment cost against the corpus mean), most anomalous first. topK <= 0
// returns every set with at least one computable pair. It is a diagnostic, not
// a filter. Returns an *UnknownGraphemeError on an unrecognised grapheme.
func FindCognateOutliers(corpus []CognateSet, model *MultiLectModel, topK, maxChunkSize int) (reports []CognateOutlierReport, err error) {
	defer catchUnknownGrapheme(&err)
	if maxChunkSize == 0 {
		maxChunkSize = DefaultMaxChunkSize
	}

	type scoredEntry struct {
		id     string
		nPairs int
		cost   float64
	}
	var scored []scoredEntry
	for _, cs := range corpus {
		lects := make([]string, 0, len(cs.Forms))
		for l := range cs.Forms {
			lects = append(lects, l)
		}
		sort.Strings(lects)
		total := 0.0
		nPairs := 0
		for i := 0; i < len(lects); i++ {
			for j := i + 1; j < len(lects); j++ {
				pm, ok := model.PairwiseModel(lects[i], lects[j])
				if !ok {
					continue
				}
				total += pairCostPerSegment(cs.Forms[lects[i]], cs.Forms[lects[j]], pm, maxChunkSize)
				nPairs++
			}
		}
		if nPairs == 0 {
			continue
		}
		scored = append(scored, scoredEntry{id: cs.CognateID, nPairs: nPairs, cost: total / float64(nPairs)})
	}

	if len(scored) == 0 {
		return nil, nil
	}

	n := len(scored)
	mean := 0.0
	for _, s := range scored {
		mean += s.cost
	}
	mean /= float64(n)
	std := 0.0
	if n >= 2 {
		variance := 0.0
		for _, s := range scored {
			d := s.cost - mean
			variance += d * d
		}
		variance /= float64(n - 1)
		std = math.Sqrt(variance)
	}

	reports = make([]CognateOutlierReport, 0, len(scored))
	for _, s := range scored {
		z := 0.0
		if std > 0.0 {
			z = (s.cost - mean) / std
		}
		reports = append(reports, CognateOutlierReport{CognateID: s.id, NPairs: s.nPairs, CostPerSegment: s.cost, ZScore: z})
	}
	sort.SliceStable(reports, func(i, j int) bool {
		if reports[i].ZScore != reports[j].ZScore {
			return reports[i].ZScore > reports[j].ZScore
		}
		return reports[i].CognateID < reports[j].CognateID
	})
	if topK > 0 && len(reports) > topK {
		reports = reports[:topK]
	}
	return reports, nil
}
