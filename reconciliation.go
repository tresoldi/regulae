package regulae

import (
	"math"
	"sort"
	"strings"
)

// Multi-lect reconciliation. Trains a per-pair model for each lect pair with
// shared cognates, reconciles the pairwise alignments through a union-find over
// corresponding positions into multi-lect correspondence-class observations,
// aggregates unconditioned classes, runs class-level context discovery, and
// lifts per-pair cross-dimensional commits to the multi-lect table.

// lectGrapheme is a (lect, grapheme) pair, used as the element of a sorted
// "sister tuple" and of the class segment key.
type lectGrapheme struct {
	Lect     string
	Grapheme string
}

// ufNode is a union-find node: a (lect, position) pair. Comparable, so usable
// directly as a map key.
type ufNode struct {
	Lect string
	Pos  int
}

// unionFind is a tiny union-find over ufNodes, with deterministic component
// iteration.
type unionFind struct {
	parent map[ufNode]ufNode
}

func newUnionFind() *unionFind {
	return &unionFind{parent: map[ufNode]ufNode{}}
}

func (uf *unionFind) find(x ufNode) ufNode {
	if _, ok := uf.parent[x]; !ok {
		uf.parent[x] = x
		return x
	}
	for uf.parent[x] != x {
		uf.parent[x] = uf.parent[uf.parent[x]]
		x = uf.parent[x]
	}
	return x
}

func (uf *unionFind) union(a, b ufNode) {
	ra := uf.find(a)
	rb := uf.find(b)
	if ra != rb {
		uf.parent[ra] = rb
	}
}

// components returns the connected components, each a slice of nodes. Nodes are
// processed in sorted order and components are returned in first-seen order of
// their sorted nodes, making iteration deterministic.
func (uf *unionFind) components() [][]ufNode {
	nodes := make([]ufNode, 0, len(uf.parent))
	for n := range uf.parent {
		nodes = append(nodes, n)
	}
	sort.Slice(nodes, func(i, j int) bool {
		if nodes[i].Lect != nodes[j].Lect {
			return nodes[i].Lect < nodes[j].Lect
		}
		return nodes[i].Pos < nodes[j].Pos
	})
	groups := map[ufNode][]ufNode{}
	var order []ufNode
	for _, n := range nodes {
		root := uf.find(n)
		if _, ok := groups[root]; !ok {
			order = append(order, root)
		}
		groups[root] = append(groups[root], n)
	}
	out := make([][]ufNode, 0, len(order))
	for _, root := range order {
		out = append(out, groups[root])
	}
	return out
}

// collectLectIDs returns the canonical lect-id list in first-seen order across
// the corpus (using each cognate set's FormsOrder).
func collectLectIDs(corpus []CognateSet) []string {
	var seen []string
	seenSet := map[string]bool{}
	for _, cs := range corpus {
		for _, lectID := range cs.orderedLects() {
			if !seenSet[lectID] {
				seen = append(seen, lectID)
				seenSet[lectID] = true
			}
		}
	}
	return seen
}

// subsetForPair extracts the pairwise (form_a, form_b) corpus and confidence
// weights for cognate sets where both lects are present, propagating cognate-set
// morpheme boundaries onto forms.
func subsetForPair(corpus []CognateSet, lectA, lectB string) ([]FormPair, []float64) {
	var out []FormPair
	var weights []float64
	for _, cs := range corpus {
		fa, okA := cs.Forms[lectA]
		fb, okB := cs.Forms[lectB]
		if okA && okB {
			out = append(out, FormPair{
				Src: formWithBoundaries(fa, cs, lectA),
				Tgt: formWithBoundaries(fb, cs, lectB),
			})
			weights = append(weights, cs.Confidence)
		}
	}
	return out, weights
}

func formWithBoundaries(form Form, cs CognateSet, lectID string) Form {
	if len(form.MorphemeBreaks) > 0 {
		return form
	}
	if cs.MorphemeBoundaries == nil {
		return form
	}
	bounds := cs.MorphemeBoundaries[lectID]
	if len(bounds) == 0 {
		return form
	}
	out := form
	out.MorphemeBreaks = append([]int(nil), bounds...)
	return out
}

// pairAlignmentEdges returns the position-level union-find edges induced by
// aligning one pair. Equal-length chunks pair position-by-position;
// unequal-length non-gap chunks are decomposed via a no-chunks sub-alignment.
func pairAlignmentEdges(formA, formB Form, pairModel *LearnedModel, lectA, lectB string, maxChunkSize int) [][2]ufNode {
	alignment := alignForms(formA, formB, pairModel.FeatureSystem, maxChunkSize, pairModel)
	var noChunks *LearnedModel

	var edges [][2]ufNode
	aPos := 0
	bPos := 0
	for _, link := range alignment.Links {
		aLen := len(link.SourceChunk)
		bLen := len(link.TargetChunk)
		if aLen == bLen {
			for i := 0; i < aLen; i++ {
				edges = append(edges, [2]ufNode{{lectA, aPos + i}, {lectB, bPos + i}})
			}
		} else if aLen == 0 || bLen == 0 {
			// pure gap: no edges
		} else {
			if noChunks == nil {
				nc := *pairModel
				nc.ChunkTable = NewChunkPhraseTable()
				noChunks = &nc
			}
			subSrc := Form{LectID: lectA, Segments: link.SourceChunk}
			subTgt := Form{LectID: lectB, Segments: link.TargetChunk}
			sub := alignForms(subSrc, subTgt, noChunks.FeatureSystem, 1, noChunks)
			subA := 0
			subB := 0
			for _, subLink := range sub.Links {
				sa := len(subLink.SourceChunk)
				sb := len(subLink.TargetChunk)
				if sa == 1 && sb == 1 {
					edges = append(edges, [2]ufNode{{lectA, aPos + subA}, {lectB, bPos + subB}})
				}
				subA += sa
				subB += sb
			}
		}
		aPos += aLen
		bPos += bLen
	}
	return edges
}

// reconcileObservation is one reconciled class observation: the participating
// lects' graphemes and positions.
type reconcileObservation struct {
	segments  map[string]string
	positions map[string]int
}

// reconcileCognateSet reconciles one cognate set into observation rows via
// union-find over corresponding positions. Components with more than one
// position per lect (alignment disagreement) or fewer than two lects are
// dropped.
func reconcileCognateSet(cs CognateSet, pairwiseModels map[string]*LearnedModel, maxChunkSize int) []reconcileObservation {
	uf := newUnionFind()
	for lectID, form := range cs.Forms {
		for pos := 0; pos < len(form.Segments); pos++ {
			uf.find(ufNode{lectID, pos})
		}
	}

	sortedLects := make([]string, 0, len(cs.Forms))
	for l := range cs.Forms {
		sortedLects = append(sortedLects, l)
	}
	sort.Strings(sortedLects)

	for i := 0; i < len(sortedLects); i++ {
		for j := i + 1; j < len(sortedLects); j++ {
			lectA, lectB := sortedLects[i], sortedLects[j]
			pm, ok := pairwiseModels[lectPairKey(lectA, lectB)]
			if !ok {
				continue
			}
			edges := pairAlignmentEdges(cs.Forms[lectA], cs.Forms[lectB], pm, lectA, lectB, maxChunkSize)
			for _, e := range edges {
				uf.union(e[0], e[1])
			}
		}
	}

	var observations []reconcileObservation
	for _, component := range uf.components() {
		byLect := map[string]int{}
		inconsistent := false
		for _, node := range component {
			if _, ok := byLect[node.Lect]; ok {
				inconsistent = true
				break
			}
			byLect[node.Lect] = node.Pos
		}
		if inconsistent || len(byLect) < 2 {
			continue
		}
		segments := map[string]string{}
		for lectID, pos := range byLect {
			segments[lectID] = cs.Forms[lectID].Segments[pos].Grapheme
		}
		observations = append(observations, reconcileObservation{segments: segments, positions: byLect})
	}
	return observations
}

// sortedSegmentItems returns the (lect, grapheme) pairs of a segments map
// sorted by lect.
func sortedSegmentItems(segments map[string]string) []lectGrapheme {
	out := make([]lectGrapheme, 0, len(segments))
	for l, g := range segments {
		out = append(out, lectGrapheme{Lect: l, Grapheme: g})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Lect != out[j].Lect {
			return out[i].Lect < out[j].Lect
		}
		return out[i].Grapheme < out[j].Grapheme
	})
	return out
}

func segItemsKey(items []lectGrapheme) string {
	var b strings.Builder
	for i, it := range items {
		if i > 0 {
			b.WriteString(keySep)
		}
		b.WriteString(it.Lect)
		b.WriteString(":")
		b.WriteString(it.Grapheme)
	}
	return b.String()
}

func lessLectGraphemeItems(a, b []lectGrapheme) bool {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i].Lect != b[i].Lect {
			return a[i].Lect < b[i].Lect
		}
		if a[i].Grapheme != b[i].Grapheme {
			return a[i].Grapheme < b[i].Grapheme
		}
	}
	return len(a) < len(b)
}

// aggregatedObservation is (cognate_id, segments, weight).
type aggregatedObservation struct {
	cognateID string
	segments  map[string]string
	weight    float64
}

// aggregateUnconditionedClasses groups observations by their sorted segment
// tuple and produces unconditioned classes, sorted by count desc then key.
func aggregateUnconditionedClasses(observations []aggregatedObservation) []MultiLectCorrespondenceClass {
	type bucket struct {
		items   []lectGrapheme
		count   float64
		support []string
	}
	buckets := map[string]*bucket{}
	var keyOrder []string
	for _, obs := range observations {
		if obs.weight <= 0.0 {
			continue
		}
		items := sortedSegmentItems(obs.segments)
		key := segItemsKey(items)
		b := buckets[key]
		if b == nil {
			b = &bucket{items: items}
			buckets[key] = b
			keyOrder = append(keyOrder, key)
		}
		b.count += obs.weight
		b.support = append(b.support, obs.cognateID)
	}

	participantTotals := map[string]float64{}
	for _, b := range buckets {
		lects := make([]string, len(b.items))
		for i, it := range b.items {
			lects[i] = it.Lect
		}
		sort.Strings(lects)
		participantTotals[strings.Join(lects, keySep)] += b.count
	}

	ordered := make([]*bucket, 0, len(buckets))
	for _, k := range keyOrder {
		ordered = append(ordered, buckets[k])
	}
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].count != ordered[j].count {
			return ordered[i].count > ordered[j].count
		}
		return lessLectGraphemeItems(ordered[i].items, ordered[j].items)
	})

	classes := make([]MultiLectCorrespondenceClass, 0, len(ordered))
	for classID, b := range ordered {
		lects := make([]string, len(b.items))
		segments := map[string]string{}
		for i, it := range b.items {
			lects[i] = it.Lect
			segments[it.Lect] = it.Grapheme
		}
		sort.Strings(lects)
		n := participantTotals[strings.Join(lects, keySep)]
		u := WilsonInterval(b.count, n, DefaultAlpha)
		classes = append(classes, MultiLectCorrespondenceClass{
			ClassID:            classID,
			Segments:           segments,
			Contexts:           nil,
			Count:              b.count,
			SupportingCognates: b.support,
			Confidence:         1.0,
			Uncertainty:        &u,
		})
	}
	return classes
}

// computeSinglePositionContext returns the phonological Context at one position
// in one form (a 1-segment window), so observation contexts match link contexts.
func computeSinglePositionContext(form Form, pos int, featureSystem string) Context {
	return computeLinkContext(form, form, pos, pos+1, 0, 0, featureSystem)
}

// perLectObservation is (cognate_id, segments, positions, weight).
type perLectObservation struct {
	cognateID string
	segments  map[string]string
	positions map[string]int
	weight    float64
}

// pivotObs is one observation in a pivot bucket: the sister-tuple key, the
// pivot context, and the weight.
type pivotObs struct {
	sisterKey string
	ctx       Context
	weight    float64
}

// committedSplit records one committed multi-lect split.
type committedSplit struct {
	pivotLect     string
	pivotGrapheme string
	ctx           Context
	sister        []lectGrapheme
	count         float64
	bucketSize    float64
}

func multiLectMinCommitCount(nTotal, scale float64) int {
	if scale <= 0.0 {
		return 2
	}
	v := int(math.Ceil(scale * math.Log2(nTotal+1)))
	if v < 2 {
		return 2
	}
	return v
}

// multiLectContextDiscovery runs class-level context discovery: for each
// (pivot_lect, pivot_grapheme) appearing across multiple distinct sister
// tuples, a greedy BIC-driven split loop on the pivot's phonological context;
// committed splits become conditioned classes (deduplicated across pivots).
func multiLectContextDiscovery(corpus []CognateSet, observations []perLectObservation, featureSystem string, startClassID int, bicSmallSampleCorrection bool, minCommitScale float64, bicConfig BICConfig) []MultiLectCorrespondenceClass {
	if len(observations) == 0 {
		return nil
	}

	corpusByID := map[string]CognateSet{}
	for _, cs := range corpus {
		corpusByID[cs.CognateID] = cs
	}

	type pivotKey struct{ lect, grapheme string }
	pivotBuckets := map[pivotKey][]pivotObs{}
	var pivotOrder []pivotKey
	sisterByKey := map[string][]lectGrapheme{}

	for _, obs := range observations {
		if obs.weight <= 0.0 {
			continue
		}
		cs := corpusByID[obs.cognateID]
		// Iterate pivots in sorted lect order for determinism.
		pivotLects := make([]string, 0, len(obs.segments))
		for l := range obs.segments {
			pivotLects = append(pivotLects, l)
		}
		sort.Strings(pivotLects)
		for _, pivotLect := range pivotLects {
			pivotG := obs.segments[pivotLect]
			var sister []lectGrapheme
			for _, l := range pivotLects {
				if l != pivotLect {
					sister = append(sister, lectGrapheme{Lect: l, Grapheme: obs.segments[l]})
				}
			}
			if len(sister) == 0 {
				continue
			}
			// sister is already sorted by lect (pivotLects sorted).
			sisterKey := sisterTupleKey(sister)
			sisterByKey[sisterKey] = sister
			form := cs.Forms[pivotLect]
			pos := obs.positions[pivotLect]
			ctx := computeSinglePositionContext(form, pos, featureSystem)
			pk := pivotKey{pivotLect, pivotG}
			if _, ok := pivotBuckets[pk]; !ok {
				pivotOrder = append(pivotOrder, pk)
			}
			pivotBuckets[pk] = append(pivotBuckets[pk], pivotObs{sisterKey: sisterKey, ctx: ctx, weight: obs.weight})
		}
	}

	// Observed stress values across all pivot observations.
	observedStress := map[string]struct{}{}
	for _, obsList := range pivotBuckets {
		for _, o := range obsList {
			for _, slot := range [][]FeatureConstraint{o.ctx.SelfStress, o.ctx.PrecedingStress, o.ctx.FollowingStress} {
				for _, fc := range slot {
					if fc.Feature == "stress" {
						observedStress[fc.Value] = struct{}{}
					}
				}
			}
		}
	}

	var committed []committedSplit
	// Process pivots in sorted order for deterministic tie-breaking in merge.
	sort.Slice(pivotOrder, func(i, j int) bool {
		if pivotOrder[i].lect != pivotOrder[j].lect {
			return pivotOrder[i].lect < pivotOrder[j].lect
		}
		return pivotOrder[i].grapheme < pivotOrder[j].grapheme
	})
	for _, pk := range pivotOrder {
		obs := pivotBuckets[pk]
		targetSet := map[string]struct{}{}
		massSum := 0.0
		for _, o := range obs {
			targetSet[o.sisterKey] = struct{}{}
			massSum += o.weight
		}
		if len(targetSet) < 2 || massSum < 4.0 {
			continue
		}
		commitMultiLectSplitsForPivot(pk.lect, pk.grapheme, obs, sisterByKey, &committed, bicSmallSampleCorrection, minCommitScale, observedStress, bicConfig)
		commitMultiLectLongRangeSplitsForPivot(pk.lect, pk.grapheme, obs, sisterByKey, &committed, minCommitScale, bicConfig)
	}

	// Merge committed splits across pivots, keyed by full segment tuple.
	type mergedEntry struct {
		segments     map[string]string
		contexts     map[string]Context
		count        float64
		confidence   float64
		bucketSize   float64
		winningCount float64
	}
	merged := map[string]*mergedEntry{}
	var mergedOrder []string
	for _, c := range committed {
		segmentsMap := map[string]string{c.pivotLect: c.pivotGrapheme}
		for _, lg := range c.sister {
			segmentsMap[lg.Lect] = lg.Grapheme
		}
		segKey := segItemsKey(sortedSegmentItems(segmentsMap))
		coverage := 0.0
		if c.bucketSize > 0 {
			coverage = c.count / c.bucketSize
		}
		entry := merged[segKey]
		if entry == nil {
			entry = &mergedEntry{
				segments:     segmentsMap,
				contexts:     map[string]Context{c.pivotLect: c.ctx},
				count:        c.count,
				confidence:   coverage,
				bucketSize:   c.bucketSize,
				winningCount: c.count,
			}
			merged[segKey] = entry
			mergedOrder = append(mergedOrder, segKey)
		} else {
			if c.count > entry.count {
				entry.count = c.count
			}
			if coverage > entry.confidence {
				entry.confidence = coverage
				entry.bucketSize = c.bucketSize
				entry.winningCount = c.count
			}
			existing, ok := entry.contexts[c.pivotLect]
			if !ok || c.ctx.ConstraintCount() > existing.ConstraintCount() {
				entry.contexts[c.pivotLect] = c.ctx
			}
		}
	}

	classes := make([]MultiLectCorrespondenceClass, 0, len(merged))
	for _, segKey := range mergedOrder {
		entry := merged[segKey]
		contextsMap := map[string]Context{}
		for l, c := range entry.contexts {
			contextsMap[l] = c
		}
		for lectID := range entry.segments {
			if _, ok := contextsMap[lectID]; !ok {
				contextsMap[lectID] = Context{}
			}
		}
		u := WilsonInterval(entry.winningCount, entry.bucketSize, DefaultAlpha)
		classes = append(classes, MultiLectCorrespondenceClass{
			ClassID:     0,
			Segments:    entry.segments,
			Contexts:    contextsMap,
			Count:       entry.count,
			Confidence:  entry.confidence,
			Uncertainty: &u,
		})
	}

	sort.SliceStable(classes, func(i, j int) bool {
		if classes[i].Count != classes[j].Count {
			return classes[i].Count > classes[j].Count
		}
		return lessLectGraphemeItems(sortedSegmentItems(classes[i].Segments), sortedSegmentItems(classes[j].Segments))
	})
	for i := range classes {
		classes[i].ClassID = startClassID + i
	}
	return classes
}

func sisterTupleKey(sister []lectGrapheme) string {
	var b strings.Builder
	for i, lg := range sister {
		if i > 0 {
			b.WriteString("|")
		}
		b.WriteString(lg.Lect)
		b.WriteString(":")
		b.WriteString(lg.Grapheme)
	}
	return b.String()
}

// pivotObsToRows converts pivotObs to the obsRow form used by the shared
// split machinery (target = sister key).
func pivotObsToRows(obs []pivotObs) []obsRow {
	rows := make([]obsRow, len(obs))
	for i, o := range obs {
		rows[i] = obsRow{tgt: o.sisterKey, ctx: o.ctx, weight: o.weight}
	}
	return rows
}

func commitMultiLectSplitsForPivot(pivotLect, pivotGrapheme string, obs []pivotObs, sisterByKey map[string][]lectGrapheme, committed *[]committedSplit, useBICSmallSampleCorrection bool, minCommitScale float64, observedStress map[string]struct{}, bicConfig BICConfig) {
	minObs := float64(bicConfig.MinSplitObservations)
	maxDepth := bicConfig.MaxSplitDepth
	deltaThreshold := bicConfig.DeltaBICThreshold

	rows := pivotObsToRows(obs)
	nTotal := observationWeight(rows)
	if nTotal < 4 {
		return
	}
	lnN := math.Log(nTotal)
	penalty := lnN
	if useBICSmallSampleCorrection {
		penalty += 2.0 / math.Max(nTotal-1, 1)
	}
	minCommit := float64(multiLectMinCommitCount(nTotal, minCommitScale))

	remaining := rows
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
			deltaBIC := -2.0*reduction + penalty
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
		yesCtx := applyPredicate(Context{}, *bestPredicate)
		emitSisterClasses(pivotLect, pivotGrapheme, yesCtx, bestYes, sisterByKey, committed, minCommit, nTotal)
		remaining = bestNo
		committedCount++
	}
}

func commitMultiLectLongRangeSplitsForPivot(pivotLect, pivotGrapheme string, obs []pivotObs, sisterByKey map[string][]lectGrapheme, committed *[]committedSplit, minCommitScale float64, bicConfig BICConfig) {
	minObs := float64(bicConfig.LongRangeMinSplitObservations)
	maxDepth := bicConfig.MaxSplitDepth
	deltaThreshold := bicConfig.LongRangeDeltaBICThreshold
	dominantFraction := bicConfig.LongRangeMinDominantFraction

	rows := pivotObsToRows(obs)
	nTotal := observationWeight(rows)
	if nTotal < minObs {
		return
	}
	lnN := math.Log(nTotal)
	minCommit := math.Max(float64(multiLectMinCommitCount(nTotal, minCommitScale)), minObs)

	remaining := rows
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
		yesCtx := applyPredicate(Context{}, *bestPredicate)
		emitSisterClasses(pivotLect, pivotGrapheme, yesCtx, bestYes, sisterByKey, committed, minCommit, nTotal)
		remaining = bestNo
		committedCount++
	}
}

// emitSisterClasses commits one class per distinct sister tuple in the YES
// partition whose count meets the adaptive minimum. Sister tuples are iterated
// in sorted key order for determinism.
func emitSisterClasses(pivotLect, pivotGrapheme string, yesCtx Context, yesObs []obsRow, sisterByKey map[string][]lectGrapheme, committed *[]committedSplit, minCommit, nTotal float64) {
	sisterCounts := map[string]float64{}
	for _, o := range yesObs {
		sisterCounts[o.tgt] += o.weight
	}
	keys := make([]string, 0, len(sisterCounts))
	for k := range sisterCounts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		count := sisterCounts[k]
		if count < minCommit {
			continue
		}
		*committed = append(*committed, committedSplit{
			pivotLect:     pivotLect,
			pivotGrapheme: pivotGrapheme,
			ctx:           yesCtx,
			sister:        sisterByKey[k],
			count:         count,
			bucketSize:    nTotal,
		})
	}
}

// trainMultiLect is the multi-lect training pipeline.
func trainMultiLect(corpus []CognateSet, opts TrainOptions) *MultiLectModel {
	lectIDs := collectLectIDs(corpus)
	if len(lectIDs) == 0 {
		return EmptyMultiLectModel()
	}
	if len(lectIDs) == 1 {
		m := EmptyMultiLectModel()
		m.CognateCorpus = append([]CognateSet(nil), corpus...)
		m.LectIDs = lectIDs
		return m
	}

	pairwiseModels := map[string]*LearnedModel{}
	for i := 0; i < len(lectIDs); i++ {
		for j := i + 1; j < len(lectIDs); j++ {
			lectA, lectB := lectIDs[i], lectIDs[j]
			pairCorpus, pairWeights := subsetForPair(corpus, lectA, lectB)
			if len(pairCorpus) == 0 {
				continue
			}
			pairModel := trainPairwiseLegacy(pairCorpus, pairWeights, opts)
			pairwiseModels[lectPairKey(lectA, lectB)] = pairModel
		}
	}

	var allObservations []aggregatedObservation
	var perLectObservations []perLectObservation
	for _, cs := range corpus {
		rows := reconcileCognateSet(cs, pairwiseModels, opts.MaxChunkSize)
		for _, row := range rows {
			allObservations = append(allObservations, aggregatedObservation{cognateID: cs.CognateID, segments: row.segments, weight: cs.Confidence})
			perLectObservations = append(perLectObservations, perLectObservation{cognateID: cs.CognateID, segments: row.segments, positions: row.positions, weight: cs.Confidence})
		}
	}
	unconditioned := aggregateUnconditionedClasses(allObservations)

	conditioned := multiLectContextDiscovery(corpus, perLectObservations, opts.FeatureSystem, len(unconditioned), opts.BICConfig.MultiLectBICSmallSampleCorrection, opts.BICConfig.MultiLectMinCommitScale, opts.BICConfig)

	crossDimTable := liftCrossDimensionalRules(pairwiseModels, lectIDs)

	return &MultiLectModel{
		PairwiseModels:        pairwiseModels,
		UnconditionedClasses:  unconditioned,
		ConditionedClasses:    conditioned,
		CognateCorpus:         append([]CognateSet(nil), corpus...),
		LectIDs:               lectIDs,
		CrossDimensionalTable: crossDimTable,
	}
}

// liftCrossDimensionalRules lifts per-pair cross-dimensional commits to the
// multi-lect table with explicit src_lect/tgt_lect labels, iterating the
// canonical pair ordering.
func liftCrossDimensionalRules(pairwiseModels map[string]*LearnedModel, lectIDs []string) MultiLectCrossDimensionalLinkTable {
	var entries []MultiLectCrossDimensionalLink
	for i := 0; i < len(lectIDs); i++ {
		for j := i + 1; j < len(lectIDs); j++ {
			lectA, lectB := lectIDs[i], lectIDs[j]
			pm, ok := pairwiseModels[lectPairKey(lectA, lectB)]
			if !ok {
				continue
			}
			for _, rule := range pm.CrossDimensionalTable.Entries {
				entries = append(entries, MultiLectCrossDimensionalLink{
					SrcLect:           lectA,
					TgtLect:           lectB,
					SrcFeature:        rule.SrcFeature,
					SrcPosition:       rule.SrcPosition,
					TgtDimension:      rule.TgtDimension,
					TgtValue:          rule.TgtValue,
					TgtPositionOffset: rule.TgtPositionOffset,
					Count:             rule.Count,
					SrcCount:          rule.SrcCount,
					Confidence:        rule.Confidence,
					SrcFeature2:       rule.SrcFeature2,
					SrcPosition2:      rule.SrcPosition2,
					Uncertainty:       rule.Uncertainty,
				})
			}
		}
	}
	return MultiLectCrossDimensionalLinkTable{Entries: entries}
}
