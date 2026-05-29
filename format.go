package regulae

import (
	"fmt"
	"sort"
	"strings"
)

// Human-readable formatting for alignments and learned models. These are
// debugging/inspection helpers, not part of the inference path. Output is plain
// text; callers should check content, not byte-for-byte formatting.

// emptyChunkSymbol renders an empty chunk (0-to-N or N-to-0 link).
const emptyChunkSymbol = "ε"

// FormatSegments renders a chunk of segments as a compact string, with
// suprasegmental annotations in brackets. An empty chunk renders as ε.
func FormatSegments(segments []Segment) string {
	if len(segments) == 0 {
		return emptyChunkSymbol
	}
	var parts []string
	for _, seg := range segments {
		piece := seg.Grapheme
		var ann []string
		if seg.Tone != "" {
			ann = append(ann, "T="+seg.Tone)
		}
		if seg.Length != "" {
			ann = append(ann, "L="+seg.Length)
		}
		if seg.Stress != "" {
			ann = append(ann, "S="+seg.Stress)
		}
		if len(ann) > 0 {
			piece += "[" + strings.Join(ann, ",") + "]"
		}
		parts = append(parts, piece)
	}
	return strings.Join(parts, "")
}

// FormatLink renders a single link as "source ~ target".
func FormatLink(link Link) string {
	return FormatSegments(link.SourceChunk) + " ~ " + FormatSegments(link.TargetChunk)
}

// FormatModelOptions configures FormatModel. The zero value is sensible
// (top 15 segments, top 5 displacements, min count 1).
type FormatModelOptions struct {
	TopSegments             int
	TopDisplacements        int
	MinCount                float64
	AnnotateChunks          bool
	ChunkWarningThreshold   float64
	SummarizeChunkProcesses bool
	SummarizeChunkSubtypes  bool
}

// DefaultFormatModelOptions returns the standard formatting options.
func DefaultFormatModelOptions() FormatModelOptions {
	return FormatModelOptions{
		TopSegments:           15,
		TopDisplacements:      5,
		MinCount:              1.0,
		ChunkWarningThreshold: 0.35,
	}
}

func fmtCount(n float64) string {
	if n == float64(int64(n)) {
		return fmt.Sprintf("%d", int64(n))
	}
	return fmt.Sprintf("%.1f", n)
}

type ccCount struct {
	cc    ConditionedCorrespondence
	count float64
}

// FormatModel renders a LearnedModel summary as a multi-line string.
func FormatModel(model *LearnedModel, opts FormatModelOptions) string {
	if opts.TopSegments == 0 {
		opts.TopSegments = 15
	}
	if opts.TopDisplacements == 0 {
		opts.TopDisplacements = 5
	}
	if opts.MinCount == 0 {
		opts.MinCount = 1.0
	}
	if opts.ChunkWarningThreshold == 0 {
		opts.ChunkWarningThreshold = 0.35
	}

	var lines []string
	lines = append(lines, fmt.Sprintf("LearnedModel (feature_system=%q, τ=%v, β=%v, w_seg=%v, w_disp=%v, w_tone=%v)",
		model.FeatureSystem, model.Temperature, model.Concentration, model.SegmentWeight, model.DisplacementWeight, model.ToneWeight))

	var unconditioned, conditioned []ccCount
	for k, v := range model.SegmentTable.Counts {
		if v < opts.MinCount {
			continue
		}
		cc := model.SegmentTable.Corr[k]
		if cc.Context.ConstraintCount() == 0 {
			unconditioned = append(unconditioned, ccCount{cc, v})
		} else {
			conditioned = append(conditioned, ccCount{cc, v})
		}
	}
	sort.SliceStable(unconditioned, func(i, j int) bool {
		if unconditioned[i].count != unconditioned[j].count {
			return unconditioned[i].count > unconditioned[j].count
		}
		if unconditioned[i].cc.Src != unconditioned[j].cc.Src {
			return unconditioned[i].cc.Src < unconditioned[j].cc.Src
		}
		return unconditioned[i].cc.Tgt < unconditioned[j].cc.Tgt
	})
	topU := unconditioned
	if len(topU) > opts.TopSegments {
		topU = topU[:opts.TopSegments]
	}
	lines = append(lines, fmt.Sprintf("Segment correspondences — unconditioned (top %d of %d):", len(topU), len(unconditioned)))
	if len(topU) == 0 {
		lines = append(lines, "  (no observations yet)")
	} else {
		for _, e := range topU {
			lines = append(lines, fmt.Sprintf("  %s -> %s: %s", e.cc.Src, e.cc.Tgt, fmtCount(e.count)))
		}
	}
	if len(conditioned) > 0 {
		lines = append(lines, "")
		lines = append(lines, fmt.Sprintf("Context-conditioned splits (%d entries):", len(conditioned)))
		bySrc := map[string][]ccCount{}
		var srcOrder []string
		for _, e := range conditioned {
			if _, ok := bySrc[e.cc.Src]; !ok {
				srcOrder = append(srcOrder, e.cc.Src)
			}
			bySrc[e.cc.Src] = append(bySrc[e.cc.Src], e)
		}
		sort.Strings(srcOrder)
		for _, src := range srcOrder {
			entries := bySrc[src]
			sort.SliceStable(entries, func(i, j int) bool {
				ci, cj := entries[i].cc.Context.ConstraintCount(), entries[j].cc.Context.ConstraintCount()
				if ci != cj {
					return ci > cj
				}
				if entries[i].count != entries[j].count {
					return entries[i].count > entries[j].count
				}
				return entries[i].cc.Tgt < entries[j].cc.Tgt
			})
			for _, e := range entries {
				lines = append(lines, fmt.Sprintf("  %s -> %s%s: %s", e.cc.Src, e.cc.Tgt, compactContext(e.cc.Context), fmtCount(e.count)))
			}
		}
	}

	// Tonal correspondences.
	lines = append(lines, fmt.Sprintf("Tonal correspondences (%d):", len(model.TonalTable.Counts)))
	if len(model.TonalTable.Counts) == 0 {
		lines = append(lines, "  (none yet)")
	} else {
		type tc struct {
			key   TonalCorrespondence
			count float64
		}
		var tonals []tc
		for k, v := range model.TonalTable.Counts {
			tonals = append(tonals, tc{k, v})
		}
		sort.SliceStable(tonals, func(i, j int) bool { return tonals[i].count > tonals[j].count })
		if len(tonals) > opts.TopSegments {
			tonals = tonals[:opts.TopSegments]
		}
		for _, t := range tonals {
			tsrc := t.key.SrcTone
			if tsrc == "" {
				tsrc = "∅"
			}
			ttgt := t.key.TgtTone
			if ttgt == "" {
				ttgt = "∅"
			}
			lines = append(lines, fmt.Sprintf("  %s -> %s: %s", tsrc, ttgt, fmtCount(t.count)))
		}
	}

	// Cross-dimensional links.
	cd := model.CrossDimensionalTable.Entries
	lines = append(lines, fmt.Sprintf("Cross-dimensional links (%d):", len(cd)))
	if len(cd) == 0 {
		lines = append(lines, "  (none)")
	} else {
		sorted := append([]CrossDimensionalLink(nil), cd...)
		sort.SliceStable(sorted, func(i, j int) bool {
			a, b := sorted[i], sorted[j]
			if a.Confidence != b.Confidence {
				return a.Confidence > b.Confidence
			}
			if a.Count != b.Count {
				return a.Count > b.Count
			}
			if a.SrcFeature.Feature != b.SrcFeature.Feature {
				return a.SrcFeature.Feature < b.SrcFeature.Feature
			}
			if a.SrcPosition != b.SrcPosition {
				return a.SrcPosition < b.SrcPosition
			}
			return a.TgtValue < b.TgtValue
		})
		shown := sorted
		if len(shown) > opts.TopSegments {
			shown = shown[:opts.TopSegments]
		}
		for _, e := range shown {
			srcLabel := renderSrcPredicate(e.SrcFeature, e.SrcPosition)
			if e.SrcFeature2 != nil && e.SrcPosition2 != "" {
				srcLabel += " & " + renderSrcPredicate(*e.SrcFeature2, e.SrcPosition2)
			}
			lines = append(lines, fmt.Sprintf("  %s -> %s=%s@%s%d  count=%s/%s conf=%.2f",
				srcLabel, e.TgtDimension, e.TgtValue, plusSign(e.TgtPositionOffset), e.TgtPositionOffset,
				fmtCount(e.Count), fmtCount(e.SrcCount), e.Confidence))
		}
		if len(sorted) > opts.TopSegments {
			lines = append(lines, fmt.Sprintf("  ... (%d more)", len(sorted)-opts.TopSegments))
		}
	}

	// Displacement vectors.
	type dispCount struct {
		disp  []FeatureDisplacement
		count float64
	}
	var disps []dispCount
	for k, v := range model.DisplacementDist.Counts {
		disps = append(disps, dispCount{model.DisplacementDist.Disp[k], v})
	}
	sort.SliceStable(disps, func(i, j int) bool {
		if disps[i].count != disps[j].count {
			return disps[i].count > disps[j].count
		}
		return compactDisplacement(disps[i].disp) < compactDisplacement(disps[j].disp)
	})
	topD := disps
	if len(topD) > opts.TopDisplacements {
		topD = topD[:opts.TopDisplacements]
	}
	lines = append(lines, fmt.Sprintf("Feature displacements (top %d of %d):", len(topD), len(model.DisplacementDist.Counts)))
	if len(topD) == 0 {
		lines = append(lines, "  (none yet)")
	} else {
		for _, d := range topD {
			lines = append(lines, fmt.Sprintf("  %s: %s", compactDisplacement(d.disp), fmtCount(d.count)))
		}
	}

	// Chunk table.
	lines = append(lines, fmt.Sprintf("Promoted chunks (%d):", len(model.ChunkTable.Entries)))
	if len(model.ChunkTable.Entries) == 0 {
		lines = append(lines, "  (none promoted)")
	} else {
		reportByKey := map[string]ChunkTransparencyReport{}
		if opts.AnnotateChunks {
			for _, r := range analyzePromotedChunks(model) {
				reportByKey[chunkPairKey(r.SrcChunk, r.TgtChunk)] = r
			}
		}
		type chunkEntry struct {
			key  string
			cost float64
		}
		var entries []chunkEntry
		for k, cost := range model.ChunkTable.Entries {
			entries = append(entries, chunkEntry{k, cost})
		}
		sort.SliceStable(entries, func(i, j int) bool { return entries[i].cost < entries[j].cost })
		for _, e := range entries {
			pair := model.ChunkTable.Pairs[e.key]
			s := chunkString(pair.Src)
			t := chunkString(pair.Tgt)
			line := fmt.Sprintf("  (%s, %s): cost=%.3f", s, t, e.cost)
			if r, ok := reportByKey[e.key]; ok {
				line += fmt.Sprintf(" score=%.2f profile=%s subtype=%s conf=%.2f", r.TransparencyScore, r.ProcessProfile, r.ProcessSubtype, r.ProcessConfidence)
				if len(r.ProcessEvidence) > 0 {
					line += " evidence=" + strings.Join(r.ProcessEvidence, "; ")
				}
				if len(r.Notes) > 0 {
					line += " notes=" + strings.Join(r.Notes, "; ")
				}
				if r.TransparencyScore < opts.ChunkWarningThreshold {
					line += " WARNING"
				}
			}
			lines = append(lines, line)
		}
	}

	if opts.SummarizeChunkProcesses {
		families := summarizeChunkProcessFamilies(model)
		lines = append(lines, fmt.Sprintf("Chunk process families (%d):", len(families)))
		if len(families) == 0 {
			lines = append(lines, "  (none)")
		}
		for _, f := range families {
			line := fmt.Sprintf("  %s: count=%d support=%.2f avg_score=%.2f avg_conf=%.2f", f.ProcessProfile, f.ChunkCount, f.WeightedSupport, f.AverageTransparency, f.AverageProcessConfidence)
			if len(f.RepresentativeChunks) > 0 {
				line += " examples=" + joinChunkPairs(f.RepresentativeChunks)
			}
			if len(f.EvidenceSignatures) > 0 {
				line += " evidence=" + strings.Join(f.EvidenceSignatures, "; ")
			}
			lines = append(lines, line)
		}
	}

	if opts.SummarizeChunkSubtypes {
		subtypes := summarizeChunkProcessSubtypes(model)
		lines = append(lines, fmt.Sprintf("Chunk process subtypes (%d):", len(subtypes)))
		if len(subtypes) == 0 {
			lines = append(lines, "  (none)")
		}
		for _, s := range subtypes {
			line := fmt.Sprintf("  %s/%s: count=%d support=%.2f avg_score=%.2f avg_conf=%.2f", s.ProcessProfile, s.ProcessSubtype, s.ChunkCount, s.WeightedSupport, s.AverageTransparency, s.AverageProcessConfidence)
			if len(s.RepresentativeChunks) > 0 {
				line += " examples=" + joinChunkPairs(s.RepresentativeChunks)
			}
			if len(s.EvidenceSignatures) > 0 {
				line += " evidence=" + strings.Join(s.EvidenceSignatures, "; ")
			}
			if len(s.ContextSignatures) > 0 {
				line += " contexts=" + strings.Join(s.ContextSignatures, "; ")
			}
			lines = append(lines, line)
		}
	}

	return strings.Join(lines, "\n")
}

func joinChunkPairs(pairs [][2]string) string {
	var parts []string
	for _, p := range pairs {
		parts = append(parts, p[0]+"->"+p[1])
	}
	return strings.Join(parts, ", ")
}

func plusSign(n int) string {
	if n >= 0 {
		return "+"
	}
	return ""
}

func renderSrcPredicate(fc FeatureConstraint, positionSpec string) string {
	return fmt.Sprintf("%s=%s@%s", fc.Feature, fc.Value, positionSpec)
}

func constraintsBracket(cs []FeatureConstraint) string {
	var parts []string
	for _, c := range cs {
		parts = append(parts, c.Feature+":"+c.Value)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// compactContext renders a one-line notation for a context, or "" if empty.
func compactContext(ctx Context) string {
	if ctx.ConstraintCount() == 0 {
		return ""
	}
	hasImmediate := len(ctx.Preceding) > 0 || len(ctx.Following) > 0
	var pieces []string
	if hasImmediate {
		if len(ctx.Preceding) > 0 {
			pieces = append(pieces, constraintsBracket(ctx.Preceding))
		}
		pieces = append(pieces, "_")
		if len(ctx.Following) > 0 {
			pieces = append(pieces, constraintsBracket(ctx.Following))
		}
	}
	env := strings.Join(pieces, "")
	pos := ""
	if ctx.Position != "" {
		pos = "@" + ctx.Position
	}
	var lr []string
	for _, dc := range ctx.PrecedingAtDistance {
		lr = append(lr, fmt.Sprintf("pre@%d=[%s:%s]", dc.Offset, dc.Constraint.Feature, dc.Constraint.Value))
	}
	for _, dc := range ctx.FollowingAtDistance {
		lr = append(lr, fmt.Sprintf("fol@%d=[%s:%s]", dc.Offset, dc.Constraint.Feature, dc.Constraint.Value))
	}
	if len(ctx.SomewherePreceding) > 0 {
		lr = append(lr, "s_pre="+constraintsBracket(ctx.SomewherePreceding))
	}
	if len(ctx.SomewhereFollowing) > 0 {
		lr = append(lr, "s_fol="+constraintsBracket(ctx.SomewhereFollowing))
	}
	if len(ctx.SameSyllable) > 0 {
		lr = append(lr, "same_syl="+constraintsBracket(ctx.SameSyllable))
	}
	if len(ctx.NextSyllable) > 0 {
		lr = append(lr, "next_syl="+constraintsBracket(ctx.NextSyllable))
	}
	if len(ctx.PreviousSyllable) > 0 {
		lr = append(lr, "prev_syl="+constraintsBracket(ctx.PreviousSyllable))
	}
	lrStr := ""
	if len(lr) > 0 {
		sep := ""
		if env != "" || pos != "" {
			sep = " "
		}
		lrStr = sep + strings.Join(lr, " ")
	}
	return " / " + env + pos + lrStr
}

func compactDisplacement(disp []FeatureDisplacement) string {
	if len(disp) == 0 {
		return "identity"
	}
	var parts []string
	for _, d := range disp {
		fv := paValue(d.FromValue)
		tv := paValue(d.ToValue)
		parts = append(parts, fmt.Sprintf("%s: %s->%s", d.Feature, fv, tv))
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func paValue(v string) string {
	switch v {
	case "present":
		return "P"
	case "absent":
		return "A"
	}
	return v
}

// FormatAlignment renders an alignment as a multi-line string. show=true shows
// per-link and total costs (computed model-free under featureSystem).
func FormatAlignment(alignment Alignment, featureSystem string, showCosts, showDisplacement bool) string {
	if featureSystem == "" {
		featureSystem = "descriptive"
	}
	src := alignment.SourceForm
	tgt := alignment.TargetForm
	header := fmt.Sprintf("%s %q ~ %s %q", src.LectID, FormatSegments(src.Segments), tgt.LectID, FormatSegments(tgt.Segments))
	if showCosts {
		total := 0.0
		for _, link := range alignment.Links {
			c, _ := ScoreLink(link, featureSystem, nil)
			total += c
		}
		header += fmt.Sprintf("  (cost=%.3f)", total)
	}
	lines := []string{header}
	for _, link := range alignment.Links {
		line := "  " + FormatLink(link)
		if showCosts {
			c, _ := ScoreLink(link, featureSystem, nil)
			line += fmt.Sprintf("  [%.3f]", c)
		}
		lines = append(lines, line)
		if showDisplacement {
			for _, d := range link.FeatureDisplacement {
				lines = append(lines, fmt.Sprintf("      %s: %s -> %s", d.Feature, d.FromValue, d.ToValue))
			}
		}
	}
	return strings.Join(lines, "\n")
}

// ----- multi-lect formatting ----------------------------------------------

func formatClassSegments(klass MultiLectCorrespondenceClass) string {
	items := sortedSegmentItems(klass.Segments)
	var parts []string
	for _, it := range items {
		parts = append(parts, it.Lect+":"+it.Grapheme)
	}
	return strings.Join(parts, " ")
}

func formatClassContexts(klass MultiLectCorrespondenceClass) string {
	if klass.Contexts == nil {
		return ""
	}
	type group struct {
		ctx     Context
		members []string
	}
	var groups []group
	lects := make([]string, 0, len(klass.Contexts))
	for l := range klass.Contexts {
		lects = append(lects, l)
	}
	sort.Strings(lects)
	for _, lectID := range lects {
		ctx := klass.Contexts[lectID]
		if ctx.ConstraintCount() == 0 {
			continue
		}
		placed := false
		for i := range groups {
			if groups[i].ctx.key() == ctx.key() {
				groups[i].members = append(groups[i].members, lectID)
				placed = true
				break
			}
		}
		if !placed {
			groups = append(groups, group{ctx: ctx, members: []string{lectID}})
		}
	}
	if len(groups) == 0 {
		return ""
	}
	var pieces []string
	for _, g := range groups {
		body := strings.TrimLeft(compactContext(g.ctx), " /")
		if len(g.members) == 1 {
			pieces = append(pieces, g.members[0]+"="+body)
		} else {
			pieces = append(pieces, "{"+strings.Join(g.members, ",")+"}="+body)
		}
	}
	return " [" + strings.Join(pieces, " | ") + "]"
}

// FormatMultiLectModel renders a MultiLectModel summary.
func FormatMultiLectModel(model *MultiLectModel, topUnconditioned, topConditioned int) string {
	if topUnconditioned == 0 {
		topUnconditioned = 20
	}
	if topConditioned == 0 {
		topConditioned = 20
	}
	bar := strings.Repeat("=", 60)
	var lines []string
	lines = append(lines, bar, "MultiLectModel", bar)
	lines = append(lines, fmt.Sprintf("lects (%d): %s", len(model.LectIDs), strings.Join(model.LectIDs, ", ")))
	lines = append(lines, fmt.Sprintf("pairwise models:     %d", len(model.PairwiseModels)))
	lines = append(lines, fmt.Sprintf("unconditioned cls:   %d", len(model.UnconditionedClasses)))
	lines = append(lines, fmt.Sprintf("conditioned cls:     %d", len(model.ConditionedClasses)))
	lines = append(lines, fmt.Sprintf("cognate corpus size: %d", len(model.CognateCorpus)))
	lines = append(lines, "")

	lines = append(lines, fmt.Sprintf("--- Top %d unconditioned classes ---", topUnconditioned))
	if len(model.UnconditionedClasses) == 0 {
		lines = append(lines, "  (none)")
	}
	for i, klass := range model.UnconditionedClasses {
		if i >= topUnconditioned {
			break
		}
		lines = append(lines, fmt.Sprintf("  [%d-way] count=%5s  %s", len(klass.Segments), fmtCount(klass.Count), formatClassSegments(klass)))
	}
	if len(model.UnconditionedClasses) > topUnconditioned {
		lines = append(lines, fmt.Sprintf("  ... (%d more)", len(model.UnconditionedClasses)-topUnconditioned))
	}

	lines = append(lines, "")
	lines = append(lines, fmt.Sprintf("--- Top %d conditioned classes ---", topConditioned))
	if len(model.ConditionedClasses) == 0 {
		lines = append(lines, "  (none — class-level discovery committed no splits)")
	}
	condSorted := append([]MultiLectCorrespondenceClass(nil), model.ConditionedClasses...)
	sort.SliceStable(condSorted, func(i, j int) bool {
		a, b := condSorted[i], condSorted[j]
		if a.Confidence != b.Confidence {
			return a.Confidence > b.Confidence
		}
		if a.Count != b.Count {
			return a.Count > b.Count
		}
		return lessLectGraphemeItems(sortedSegmentItems(a.Segments), sortedSegmentItems(b.Segments))
	})
	for i, klass := range condSorted {
		if i >= topConditioned {
			break
		}
		lines = append(lines, fmt.Sprintf("  count=%5s cov=%.2f  %s%s", fmtCount(klass.Count), klass.Confidence, formatClassSegments(klass), formatClassContexts(klass)))
	}
	if len(condSorted) > topConditioned {
		lines = append(lines, fmt.Sprintf("  ... (%d more)", len(condSorted)-topConditioned))
	}

	xd := model.CrossDimensionalTable.Entries
	lines = append(lines, "")
	lines = append(lines, fmt.Sprintf("--- Multi-lect cross-dimensional rules (%d) ---", len(xd)))
	if len(xd) == 0 {
		lines = append(lines, "  (none)")
	} else {
		for _, rule := range xd {
			srcLabel := renderSrcPredicate(rule.SrcFeature, rule.SrcPosition)
			if rule.SrcFeature2 != nil && rule.SrcPosition2 != "" {
				srcLabel += " & " + renderSrcPredicate(*rule.SrcFeature2, rule.SrcPosition2)
			}
			lines = append(lines, fmt.Sprintf("  %s[%s] → %s[%s=%s@%s%d]  count=%.0f/%.0f  conf=%.2f",
				rule.SrcLect, srcLabel, rule.TgtLect, rule.TgtDimension, rule.TgtValue,
				plusSign(rule.TgtPositionOffset), rule.TgtPositionOffset, rule.Count, rule.SrcCount, rule.Confidence))
		}
	}

	return strings.Join(lines, "\n")
}

// DescribeMultiLectClass describes every class containing a (lectID, grapheme).
func DescribeMultiLectClass(model *MultiLectModel, lectID, grapheme string) string {
	var lines []string
	lines = append(lines, fmt.Sprintf("Classes with %s:%s", lectID, grapheme))
	lines = append(lines, strings.Repeat("=", 50))

	var uncond []MultiLectCorrespondenceClass
	for _, k := range model.UnconditionedClasses {
		if k.Segments[lectID] == grapheme {
			uncond = append(uncond, k)
		}
	}
	lines = append(lines, fmt.Sprintf("Unconditioned entries (%d)", len(uncond)))
	if len(uncond) == 0 {
		lines = append(lines, "  (none)")
	}
	for _, klass := range uncond {
		lines = append(lines, fmt.Sprintf("  count=%5s  %s", fmtCount(klass.Count), formatClassSegments(klass)))
	}

	lines = append(lines, "")
	var cond []MultiLectCorrespondenceClass
	for _, k := range model.ConditionedClasses {
		if k.Segments[lectID] == grapheme {
			cond = append(cond, k)
		}
	}
	lines = append(lines, fmt.Sprintf("Conditioned entries (%d)", len(cond)))
	if len(cond) == 0 {
		lines = append(lines, "  (none)")
	}
	sort.SliceStable(cond, func(i, j int) bool {
		if cond[i].Confidence != cond[j].Confidence {
			return cond[i].Confidence > cond[j].Confidence
		}
		return cond[i].Count > cond[j].Count
	})
	for _, klass := range cond {
		lines = append(lines, fmt.Sprintf("  count=%5s cov=%.2f  %s%s", fmtCount(klass.Count), klass.Confidence, formatClassSegments(klass), formatClassContexts(klass)))
	}
	return strings.Join(lines, "\n")
}
