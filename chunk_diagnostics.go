package regulae

import (
	"sort"
	"strings"
)

// Diagnostics for promoted chunk correspondences. These helpers do not change
// scoring or alignment; they inspect a trained model's promoted chunk table and
// estimate how historically transparent each chunk is. A chunk is more
// transparent when it is short, structurally balanced, decomposes cleanly
// without many gap links, and does not heavily overlap smaller promoted chunks.
//
// The describe/summarize text reporters are ported in the reporting slice; this
// file holds the data types and analyzePromotedChunks (needed by chunk
// promotion) plus its classification/scoring helpers.

var chunkProcessProfiles = []string{
	"compact_fusion",
	"residual_reduction",
	"nasal_fusion",
	"glide_or_vocalization_fusion",
	"balanced_restructuring",
	"bundled_reduction",
	"mixed_or_unclear",
}

// ChunkTransparencyReport is the interpretability report for one promoted
// chunk. TransparencyScore is a heuristic in [0, 1] (higher = more
// transparent), advisory only and independent of ProcessConfidence.
type ChunkTransparencyReport struct {
	SrcChunk           []Segment
	TgtChunk           []Segment
	PromotedCost       float64
	TransparencyScore  float64
	SubAlignment       Alignment
	Asymmetry          int
	GapRatio           float64
	MatchedCount       int
	ChangedMatchCount  int
	IdentityMatchCount int
	DeletionCount      int
	InsertionCount     int
	OverlapCount       int
	ProcessProfile     string
	ProcessSubtype     string
	ProcessConfidence  float64
	ProcessEvidence    []string
	Notes              []string
}

// ChunkProcessFamilyReport is a corpus-level summary of one inferred chunk
// process family.
type ChunkProcessFamilyReport struct {
	ProcessProfile           string
	ChunkCount               int
	WeightedSupport          float64
	AverageTransparency      float64
	AverageProcessConfidence float64
	RepresentativeChunks     [][2]string
	EvidenceSignatures       []string
}

// ChunkProcessSubtypeReport is a corpus-level summary of one inferred chunk
// process subtype.
type ChunkProcessSubtypeReport struct {
	ProcessProfile           string
	ProcessSubtype           string
	ChunkCount               int
	WeightedSupport          float64
	AverageTransparency      float64
	AverageProcessConfidence float64
	RepresentativeChunks     [][2]string
	EvidenceSignatures       []string
	ContextSignatures        []string
}

// AnalyzePromotedChunks returns transparency reports for every promoted chunk
// in the model, sorted by score ascending then stored cost then chunk strings,
// so the least transparent chunks surface first. Panics on unknown graphemes
// (internal use).
func analyzePromotedChunks(model *LearnedModel) []ChunkTransparencyReport {
	noChunks := *model
	noChunks.ChunkTable = NewChunkPhraseTable()

	var reports []ChunkTransparencyReport
	for k, promotedCost := range model.ChunkTable.Entries {
		pair := model.ChunkTable.Pairs[k]
		srcChunk, tgtChunk := pair.Src, pair.Tgt
		subAlignment := alignForms(pseudoForm("_chunk_src", srcChunk), pseudoForm("_chunk_tgt", tgtChunk), noChunks.FeatureSystem, 1, &noChunks)

		asymmetry := abs(len(srcChunk) - len(tgtChunk))
		totalLinks := len(subAlignment.Links)
		if totalLinks < 1 {
			totalLinks = 1
		}
		matchedCount := 0
		identityMatchCount := 0
		deletionCount := 0
		insertionCount := 0
		gapLinks := 0
		for _, link := range subAlignment.Links {
			hasSrc := len(link.SourceChunk) > 0
			hasTgt := len(link.TargetChunk) > 0
			if hasSrc && hasTgt {
				matchedCount++
				if segmentSlicesEqualCD(link.SourceChunk, link.TargetChunk) {
					identityMatchCount++
				}
			}
			if hasSrc && !hasTgt {
				deletionCount++
			}
			if !hasSrc && hasTgt {
				insertionCount++
			}
			if !hasSrc || !hasTgt {
				gapLinks++
			}
		}
		changedMatchCount := matchedCount - identityMatchCount
		gapRatio := float64(gapLinks) / float64(totalLinks)
		ovl := overlapCount(srcChunk, tgtChunk, model)
		processProfile, processConfidence, processEvidence := classifyChunkProcess(subAlignment, model.FeatureSystem)
		processSubtype := classifyChunkSubtype(subAlignment, processProfile, processEvidence, model.FeatureSystem)
		score := transparencyScore(srcChunk, tgtChunk, asymmetry, gapRatio, matchedCount, changedMatchCount, identityMatchCount, deletionCount, insertionCount, ovl, processProfile)
		notes := chunkNotes(srcChunk, tgtChunk, asymmetry, gapRatio, matchedCount, changedMatchCount, identityMatchCount, deletionCount, insertionCount, ovl, score, processProfile)

		reports = append(reports, ChunkTransparencyReport{
			SrcChunk:           srcChunk,
			TgtChunk:           tgtChunk,
			PromotedCost:       promotedCost,
			TransparencyScore:  score,
			SubAlignment:       subAlignment,
			Asymmetry:          asymmetry,
			GapRatio:           gapRatio,
			MatchedCount:       matchedCount,
			ChangedMatchCount:  changedMatchCount,
			IdentityMatchCount: identityMatchCount,
			DeletionCount:      deletionCount,
			InsertionCount:     insertionCount,
			OverlapCount:       ovl,
			ProcessProfile:     processProfile,
			ProcessSubtype:     processSubtype,
			ProcessConfidence:  processConfidence,
			ProcessEvidence:    processEvidence,
			Notes:              notes,
		})
	}
	sort.SliceStable(reports, func(a, b int) bool {
		ra, rb := reports[a], reports[b]
		if ra.TransparencyScore != rb.TransparencyScore {
			return ra.TransparencyScore < rb.TransparencyScore
		}
		if ra.PromotedCost != rb.PromotedCost {
			return ra.PromotedCost < rb.PromotedCost
		}
		if sa, sb := chunkString(ra.SrcChunk), chunkString(rb.SrcChunk); sa != sb {
			return sa < sb
		}
		return chunkString(ra.TgtChunk) < chunkString(rb.TgtChunk)
	})
	return reports
}

func pseudoForm(lectID string, segments []Segment) Form {
	return Form{LectID: lectID, Segments: segments}
}

func chunkString(chunk []Segment) string {
	if len(chunk) == 0 {
		return "ε"
	}
	var b strings.Builder
	for _, s := range chunk {
		b.WriteString(s.Grapheme)
	}
	return b.String()
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func segmentSlicesEqualCD(a, b []Segment) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func transparencyScore(srcChunk, tgtChunk []Segment, asymmetry int, gapRatio float64, matchedCount, changedMatchCount, identityMatchCount, deletionCount, insertionCount, overlapCount int, processProfile string) float64 {
	totalLen := len(srcChunk) + len(tgtChunk)
	score := 1.0
	score -= 0.18 * float64(asymmetry)
	score -= 0.24 * gapRatio
	score -= 0.14 * float64(maxInt(totalLen-3, 0))
	score -= 0.09 * float64(minInt(overlapCount, 3)) / 3.0
	if totalLen >= 5 && asymmetry > 0 {
		score -= 0.10
	}
	if deletionCount >= 2 {
		score -= 0.10
	}
	if insertionCount >= 2 {
		score -= 0.10
	}
	if deletionCount > 0 && insertionCount == 0 && asymmetry > 0 {
		score -= 0.10
	}
	if insertionCount > 0 && deletionCount == 0 && asymmetry > 0 {
		score -= 0.10
	}
	if totalLen <= 4 && asymmetry == 1 && matchedCount == 1 && changedMatchCount == 1 && deletionCount+insertionCount == 1 {
		score += 0.08
	}
	if totalLen <= 4 && asymmetry == 1 && matchedCount == 1 && identityMatchCount == 1 && deletionCount+insertionCount == 1 {
		score += 0.03
	}
	if asymmetry == 0 && deletionCount > 0 && insertionCount > 0 {
		score += 0.12
	}
	if matchedCount >= 2 && asymmetry <= 1 {
		score += 0.05
	}
	if asymmetry == 0 && gapRatio == 0.0 && totalLen <= 4 {
		score += 0.08
	} else if totalLen <= 4 && asymmetry <= 1 {
		score += 0.05
	}
	score += processScoreAdjustment(processProfile)
	if score < 0.0 {
		return 0.0
	}
	if score > 1.0 {
		return 1.0
	}
	return score
}

func chunkNotes(srcChunk, tgtChunk []Segment, asymmetry int, gapRatio float64, matchedCount, changedMatchCount, identityMatchCount, deletionCount, insertionCount, overlapCount int, transparencyScore float64, processProfile string) []string {
	var notes []string
	totalLen := len(srcChunk) + len(tgtChunk)
	specializedShortReduction := totalLen <= 4 && asymmetry == 1 && matchedCount == 1 && deletionCount+insertionCount == 1 && (changedMatchCount == 1 || identityMatchCount == 1)
	if asymmetry > 0 {
		notes = append(notes, "length asymmetry")
	}
	if gapRatio >= 0.34 {
		notes = append(notes, "gap-heavy decomposition")
	}
	if asymmetry == 0 && deletionCount > 0 && insertionCount > 0 {
		notes = append(notes, "balanced local restructuring")
	}
	if totalLen <= 4 && asymmetry == 1 && matchedCount == 1 && changedMatchCount == 1 && deletionCount+insertionCount == 1 {
		notes = append(notes, "compact fusional reflex")
	}
	if processProfile == "nasal_fusion" {
		notes = append(notes, "nasal fusion profile")
	}
	if processProfile == "glide_or_vocalization_fusion" {
		notes = append(notes, "glide or vocalization profile")
	}
	if totalLen <= 4 && asymmetry == 1 && matchedCount == 1 && identityMatchCount == 1 && deletionCount+insertionCount == 1 {
		notes = append(notes, "residue-preserving reduction")
	}
	if deletionCount > 0 && insertionCount == 0 && asymmetry > 0 && !specializedShortReduction {
		notes = append(notes, "reductive loss chunk")
	}
	if totalLen >= 5 {
		notes = append(notes, "large bundled chunk")
	}
	if overlapCount > 0 {
		notes = append(notes, "overlaps with smaller promoted chunks")
	}
	if matchedCount >= 2 && asymmetry <= 1 && totalLen <= 5 {
		notes = append(notes, "multi-step but locally coherent")
	}
	if len(notes) == 0 && transparencyScore >= 0.8 {
		notes = append(notes, "compact decomposition")
	}
	if transparencyScore < 0.35 || (totalLen >= 5 && asymmetry > 0) {
		notes = append(notes, "historically opaque candidate")
	}
	return notes
}

func processScoreAdjustment(processProfile string) float64 {
	switch processProfile {
	case "compact_fusion":
		return 0.03
	case "residual_reduction":
		return 0.01
	case "nasal_fusion":
		return 0.05
	case "glide_or_vocalization_fusion":
		return 0.04
	case "balanced_restructuring":
		return 0.02
	case "bundled_reduction":
		return -0.04
	default:
		return 0.0
	}
}

func classifyChunkProcess(subAlignment Alignment, featureSystem string) (string, float64, []string) {
	identityMatches, changedMatches, deletions, insertions := splitSubAlignment(subAlignment)
	totalEvents := len(changedMatches) + len(identityMatches) + len(deletions) + len(insertions)

	if looksLikeNasalFusion(changedMatches, deletions, featureSystem) {
		return "nasal_fusion", 0.92, []string{"vowel gained nasalization", "adjacent nasal absorbed"}
	}
	if looksLikeGlideOrVocalization(changedMatches, deletions, featureSystem) {
		return "glide_or_vocalization_fusion", 0.84, []string{"approximant or glide reflex", "neighboring vocalic/liquid material absorbed"}
	}
	if len(changedMatches) == 1 && len(deletions)+len(insertions) == 1 && totalEvents <= 2 {
		return "compact_fusion", 0.82, compactFusionEvidence(changedMatches[0], deletions, insertions, featureSystem)
	}
	if len(identityMatches) == 1 && len(deletions)+len(insertions) == 1 && totalEvents <= 2 {
		return "residual_reduction", 0.88, []string{"one residue segment preserved", "adjacent material reduced"}
	}
	if len(deletions) > 0 && len(insertions) > 0 && len(changedMatches)+len(identityMatches) <= 1 {
		return "balanced_restructuring", 0.68, []string{"local deletion and insertion both present"}
	}
	if totalEvents >= 3 && len(deletions)+len(insertions) > 0 && (len(changedMatches)+len(identityMatches)) > 0 {
		return "bundled_reduction", 0.86, []string{"multiple local steps required", "not historically atomic on present evidence"}
	}
	return "mixed_or_unclear", 0.45, nil
}

type segPair struct{ src, tgt Segment }

func classifyChunkSubtype(subAlignment Alignment, processProfile string, processEvidence []string, featureSystem string) string {
	identityMatches, changedMatches, deletions, insertions := splitSubAlignment(subAlignment)
	switch processProfile {
	case "compact_fusion":
		for _, item := range processEvidence {
			if strings.Contains(item, "fricative-like") {
				return "fricativizing_or_affricating_fusion"
			}
		}
		for _, item := range processEvidence {
			if strings.Contains(item, "target segment inserted") {
				return "insertional_glide_or_diphthongal_fusion"
			}
		}
		if len(changedMatches) == 1 {
			src, tgt := changedMatches[0].src, changedMatches[0].tgt
			if hasFeature(tgt, "palatal", featureSystem) {
				return "palatalizing_fusion"
			}
			if hasFeature(src, "vowel", featureSystem) && hasFeature(tgt, "vowel", featureSystem) {
				return "vocalic_absorptive_fusion"
			}
		}
		return "absorptive_fusion"
	case "residual_reduction":
		if len(insertions) > 0 {
			return "residual_expansion"
		}
		if len(identityMatches) == 1 && len(deletions) == 1 {
			src := identityMatches[0].src
			deleted := deletions[0]
			if hasFeature(src, "consonant", featureSystem) && hasFeature(deleted, "vowel", featureSystem) {
				return "consonant_residue_with_vowel_loss"
			}
			if hasFeature(src, "vowel", featureSystem) && hasFeature(deleted, "consonant", featureSystem) {
				return "vowel_residue_with_consonant_loss"
			}
		}
		return "generic_residual_reduction"
	case "nasal_fusion":
		return "vowel_nasalization_with_consonant_absorption"
	case "glide_or_vocalization_fusion":
		return "glide_formation_or_vocalization"
	case "balanced_restructuring":
		return "balanced_local_restructuring"
	case "bundled_reduction":
		return "multi_step_bundle"
	}
	return "mixed_or_unclear"
}

func compactFusionEvidence(cm segPair, deletions, insertions []Segment, featureSystem string) []string {
	var evidence []string
	disp := safeDisplacement(cm.src, cm.tgt, featureSystem)
	for _, d := range disp {
		if (d.Feature == "fricative" || d.Feature == "sibilant" || d.Feature == "affricate") && d.ToValue == "present" {
			evidence = append(evidence, "surviving segment became more fricative-like")
			break
		}
	}
	for _, d := range disp {
		if d.Feature == "palatal" && d.ToValue == "present" {
			evidence = append(evidence, "surviving segment gained palatal properties")
			break
		}
	}
	if len(deletions) > 0 {
		evidence = append(evidence, "one neighboring source segment absorbed")
	}
	if len(insertions) > 0 {
		evidence = append(evidence, "one neighboring target segment inserted")
	}
	if len(evidence) == 0 {
		evidence = append(evidence, "single changed reflex plus one local non-match")
	}
	return evidence
}

func looksLikeNasalFusion(changedMatches []segPair, deletions []Segment, featureSystem string) bool {
	if len(changedMatches) != 1 || len(deletions) != 1 {
		return false
	}
	src, tgt := changedMatches[0].src, changedMatches[0].tgt
	deleted := deletions[0]
	return hasFeature(src, "vowel", featureSystem) &&
		hasFeature(tgt, "vowel", featureSystem) &&
		gainedFeature(src, tgt, "nasalized", featureSystem) &&
		hasFeature(deleted, "nasal", featureSystem)
}

func looksLikeGlideOrVocalization(changedMatches []segPair, deletions []Segment, featureSystem string) bool {
	if len(changedMatches) != 1 || len(deletions) != 1 {
		return false
	}
	src, tgt := changedMatches[0].src, changedMatches[0].tgt
	deleted := deletions[0]
	return hasFeature(tgt, "approximant", featureSystem) &&
		(hasFeature(src, "lateral", featureSystem) ||
			hasFeature(src, "approximant", featureSystem) ||
			hasFeature(deleted, "vowel", featureSystem) ||
			hasFeature(deleted, "approximant", featureSystem))
}

func gainedFeature(src, tgt Segment, feature, featureSystem string) bool {
	srcF := getFeatures(src.Grapheme, featureSystem)
	tgtF := getFeatures(tgt.Grapheme, featureSystem)
	return srcF != nil && tgtF != nil && !srcF.has(feature) && tgtF.has(feature)
}

func hasFeature(segment Segment, feature, featureSystem string) bool {
	f := getFeatures(segment.Grapheme, featureSystem)
	return f != nil && f.has(feature)
}

func safeDisplacement(src, tgt Segment, featureSystem string) []FeatureDisplacement {
	d, ok := tryComputeDisplacement(src, tgt, featureSystem)
	if !ok {
		return nil
	}
	return d
}

func splitSubAlignment(subAlignment Alignment) (identityMatches, changedMatches []segPair, deletions, insertions []Segment) {
	for _, link := range subAlignment.Links {
		hasSrc := len(link.SourceChunk) > 0
		hasTgt := len(link.TargetChunk) > 0
		if hasSrc && hasTgt {
			src := link.SourceChunk[0]
			tgt := link.TargetChunk[0]
			if src == tgt {
				identityMatches = append(identityMatches, segPair{src, tgt})
			} else {
				changedMatches = append(changedMatches, segPair{src, tgt})
			}
		} else if hasSrc {
			deletions = append(deletions, link.SourceChunk...)
		} else if hasTgt {
			insertions = append(insertions, link.TargetChunk...)
		}
	}
	return identityMatches, changedMatches, deletions, insertions
}

func overlapCount(srcChunk, tgtChunk []Segment, model *LearnedModel) int {
	count := 0
	for k := range model.ChunkTable.Entries {
		other := model.ChunkTable.Pairs[k]
		if segmentSlicesEqualCD(other.Src, srcChunk) && segmentSlicesEqualCD(other.Tgt, tgtChunk) {
			continue
		}
		if len(other.Src) >= len(srcChunk) && len(other.Tgt) >= len(tgtChunk) {
			continue
		}
		if isContiguousSubchunk(other.Src, srcChunk) && isContiguousSubchunk(other.Tgt, tgtChunk) {
			count++
		}
	}
	return count
}

func isContiguousSubchunk(needle, haystack []Segment) bool {
	if len(needle) == 0 {
		return false
	}
	if len(needle) > len(haystack) {
		return false
	}
	for start := 0; start <= len(haystack)-len(needle); start++ {
		if segmentSlicesEqualCD(haystack[start:start+len(needle)], needle) {
			return true
		}
	}
	return false
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ----- corpus-level chunk-process summaries (used by format) ---------------

// summarizeChunkProcessFamilies aggregates promoted chunks into process
// families, sorted by weighted support desc.
func summarizeChunkProcessFamilies(model *LearnedModel) []ChunkProcessFamilyReport {
	reports := analyzePromotedChunks(model)
	byProfile := map[string][]ChunkTransparencyReport{}
	var order []string
	for _, r := range reports {
		if _, ok := byProfile[r.ProcessProfile]; !ok {
			order = append(order, r.ProcessProfile)
		}
		byProfile[r.ProcessProfile] = append(byProfile[r.ProcessProfile], r)
	}
	var families []ChunkProcessFamilyReport
	for _, profile := range order {
		members := byProfile[profile]
		weightedSupport := 0.0
		avgT := 0.0
		avgC := 0.0
		for _, r := range members {
			weightedSupport += r.TransparencyScore * r.ProcessConfidence
			avgT += r.TransparencyScore
			avgC += r.ProcessConfidence
		}
		n := float64(len(members))
		families = append(families, ChunkProcessFamilyReport{
			ProcessProfile:           profile,
			ChunkCount:               len(members),
			WeightedSupport:          weightedSupport,
			AverageTransparency:      avgT / n,
			AverageProcessConfidence: avgC / n,
			RepresentativeChunks:     representativeChunks(members),
			EvidenceSignatures:       topEvidenceSignatures(members),
		})
	}
	sort.SliceStable(families, func(i, j int) bool {
		a, b := families[i], families[j]
		if a.WeightedSupport != b.WeightedSupport {
			return a.WeightedSupport > b.WeightedSupport
		}
		if a.ChunkCount != b.ChunkCount {
			return a.ChunkCount > b.ChunkCount
		}
		if a.AverageTransparency != b.AverageTransparency {
			return a.AverageTransparency > b.AverageTransparency
		}
		return a.ProcessProfile < b.ProcessProfile
	})
	return families
}

// summarizeChunkProcessSubtypes aggregates promoted chunks into process
// subtypes, sorted by weighted support desc.
func summarizeChunkProcessSubtypes(model *LearnedModel) []ChunkProcessSubtypeReport {
	reports := analyzePromotedChunks(model)
	type key struct{ profile, subtype string }
	bySubtype := map[key][]ChunkTransparencyReport{}
	var order []key
	for _, r := range reports {
		k := key{r.ProcessProfile, r.ProcessSubtype}
		if _, ok := bySubtype[k]; !ok {
			order = append(order, k)
		}
		bySubtype[k] = append(bySubtype[k], r)
	}
	var subtypes []ChunkProcessSubtypeReport
	for _, k := range order {
		members := bySubtype[k]
		weightedSupport := 0.0
		avgT := 0.0
		avgC := 0.0
		for _, r := range members {
			weightedSupport += r.TransparencyScore * r.ProcessConfidence
			avgT += r.TransparencyScore
			avgC += r.ProcessConfidence
		}
		n := float64(len(members))
		subtypes = append(subtypes, ChunkProcessSubtypeReport{
			ProcessProfile:           k.profile,
			ProcessSubtype:           k.subtype,
			ChunkCount:               len(members),
			WeightedSupport:          weightedSupport,
			AverageTransparency:      avgT / n,
			AverageProcessConfidence: avgC / n,
			RepresentativeChunks:     representativeChunks(members),
			EvidenceSignatures:       topEvidenceSignatures(members),
			ContextSignatures:        topContextSignatures(model, members),
		})
	}
	sort.SliceStable(subtypes, func(i, j int) bool {
		a, b := subtypes[i], subtypes[j]
		if a.WeightedSupport != b.WeightedSupport {
			return a.WeightedSupport > b.WeightedSupport
		}
		if a.ChunkCount != b.ChunkCount {
			return a.ChunkCount > b.ChunkCount
		}
		if a.AverageTransparency != b.AverageTransparency {
			return a.AverageTransparency > b.AverageTransparency
		}
		if a.ProcessProfile != b.ProcessProfile {
			return a.ProcessProfile < b.ProcessProfile
		}
		return a.ProcessSubtype < b.ProcessSubtype
	})
	return subtypes
}

func representativeChunks(members []ChunkTransparencyReport) [][2]string {
	ranked := append([]ChunkTransparencyReport(nil), members...)
	sort.SliceStable(ranked, func(i, j int) bool {
		a, b := ranked[i], ranked[j]
		wa := a.TransparencyScore * a.ProcessConfidence
		wb := b.TransparencyScore * b.ProcessConfidence
		if wa != wb {
			return wa > wb
		}
		if a.TransparencyScore != b.TransparencyScore {
			return a.TransparencyScore > b.TransparencyScore
		}
		if sa, sb := chunkString(a.SrcChunk), chunkString(b.SrcChunk); sa != sb {
			return sa < sb
		}
		return chunkString(a.TgtChunk) < chunkString(b.TgtChunk)
	})
	var out [][2]string
	for i := 0; i < len(ranked) && i < 3; i++ {
		out = append(out, [2]string{chunkString(ranked[i].SrcChunk), chunkString(ranked[i].TgtChunk)})
	}
	return out
}

func topEvidenceSignatures(reports []ChunkTransparencyReport) []string {
	counts := map[string]int{}
	for _, r := range reports {
		for _, e := range r.ProcessEvidence {
			counts[e]++
		}
	}
	return topByCountThenKey(counts, 3)
}

func topContextSignatures(model *LearnedModel, reports []ChunkTransparencyReport) []string {
	seenPairs := map[[2]string]bool{}
	for _, r := range reports {
		for _, link := range r.SubAlignment.Links {
			if len(link.SourceChunk) == 0 || len(link.TargetChunk) == 0 {
				continue
			}
			seenPairs[[2]string{chunkString(link.SourceChunk), chunkString(link.TargetChunk)}] = true
		}
	}
	counts := map[string]float64{}
	for k, count := range model.SegmentTable.Counts {
		cc := model.SegmentTable.Corr[k]
		if cc.Context.ConstraintCount() == 0 {
			continue
		}
		if !seenPairs[[2]string{cc.Src, cc.Tgt}] {
			continue
		}
		sig := cc.Src + "->" + cc.Tgt + compactContext(cc.Context)
		counts[sig] += count
	}
	return topByFloatThenKey(counts, 3)
}

func topByCountThenKey(counts map[string]int, n int) []string {
	type kv struct {
		k string
		v int
	}
	items := make([]kv, 0, len(counts))
	for k, v := range counts {
		items = append(items, kv{k, v})
	}
	sort.SliceStable(items, func(i, j int) bool {
		if items[i].v != items[j].v {
			return items[i].v > items[j].v
		}
		return items[i].k < items[j].k
	})
	var out []string
	for i := 0; i < len(items) && i < n; i++ {
		out = append(out, items[i].k)
	}
	return out
}

func topByFloatThenKey(counts map[string]float64, n int) []string {
	type kv struct {
		k string
		v float64
	}
	items := make([]kv, 0, len(counts))
	for k, v := range counts {
		items = append(items, kv{k, v})
	}
	sort.SliceStable(items, func(i, j int) bool {
		if items[i].v != items[j].v {
			return items[i].v > items[j].v
		}
		return items[i].k < items[j].k
	})
	var out []string
	for i := 0; i < len(items) && i < n; i++ {
		out = append(out, items[i].k)
	}
	return out
}
