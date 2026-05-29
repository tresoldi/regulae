package regulae

// Mirror of python/tests/test_chunk_diagnostics.py
//
// Note: analyzePromotedChunks, summarizeChunkProcessFamilies,
// summarizeChunkProcessSubtypes are unexported — accessible in-package.
// The Python describe_promoted_chunk / describe_chunk_process_families /
// describe_chunk_process_subtypes text functions have no direct Go
// equivalent; their assertions are translated to field-level checks on
// the report structs (or against FormatModel output for the format test).

import (
	"fmt"
	"strings"
	"testing"
)

// chkSeg builds a []Segment from grapheme strings.
func chkSeg(graphemes ...string) []Segment {
	out := make([]Segment, len(graphemes))
	for i, g := range graphemes {
		out[i] = Segment{Grapheme: g}
	}
	return out
}

// chunkEntry pairs a (src, tgt) chunk with a cost.
type chunkEntry struct {
	src  []Segment
	tgt  []Segment
	cost float64
}

// chkMakeModel creates a LearnedModel with an explicit chunk table.
func chkMakeModel(entries []chunkEntry) *LearnedModel {
	m := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	for _, e := range entries {
		m.ChunkTable.setEntry(e.src, e.tgt, e.cost)
	}
	return m
}

func TestChunkDiagnosticsTransparencyPrefersCompactBalancedChunks(t *testing.T) {
	compact := chunkEntry{chkSeg("k", "t"), chkSeg("t", "ʃ"), -4.9}
	opaque := chunkEntry{chkSeg("a", "t", "e"), chkSeg("a", "d"), -7.4}
	model := chkMakeModel([]chunkEntry{compact, opaque})
	reports := analyzePromotedChunks(model)
	byPair := map[string]ChunkTransparencyReport{}
	for _, r := range reports {
		byPair[chunkPairKey(r.SrcChunk, r.TgtChunk)] = r
	}
	compactR := byPair[chunkPairKey(compact.src, compact.tgt)]
	opaqueR := byPair[chunkPairKey(opaque.src, opaque.tgt)]
	if compactR.TransparencyScore <= opaqueR.TransparencyScore {
		t.Errorf("compact score %v should be > opaque score %v", compactR.TransparencyScore, opaqueR.TransparencyScore)
	}
}

func TestChunkDiagnosticsFormatModelCanAnnotateChunkTransparency(t *testing.T) {
	corpus := []FormPair{
		{Src: formIPA("A", "akta"), Tgt: formIPA("B", "atʃa")},
		{Src: formIPA("A", "ikti"), Tgt: formIPA("B", "itʃi")},
		{Src: formIPA("A", "uktu"), Tgt: formIPA("B", "utʃu")},
	}
	trained := mustTrain(t, corpus)
	opts := FormatModelOptions{
		TopSegments:             15,
		TopDisplacements:        5,
		MinCount:                1,
		AnnotateChunks:          true,
		SummarizeChunkProcesses: true,
		SummarizeChunkSubtypes:  true,
	}
	out := FormatModel(trained, opts)
	if !strings.Contains(out, "score=") {
		t.Errorf("output missing 'score=': %q", out)
	}
	if !strings.Contains(out, "profile=") {
		t.Errorf("output missing 'profile=': %q", out)
	}
	if !strings.Contains(out, "subtype=") {
		t.Errorf("output missing 'subtype=': %q", out)
	}
	if !strings.Contains(out, "Chunk process families") {
		t.Errorf("output missing 'Chunk process families': %q", out)
	}
	if !strings.Contains(out, "Chunk process subtypes") {
		t.Errorf("output missing 'Chunk process subtypes': %q", out)
	}
}

func TestChunkDiagnosticsDescribePromotedChunkMentionsKeyFields(t *testing.T) {
	// Python test: describe_promoted_chunk(model, 0) contains
	// "transparency score:", "process profile:", "process subtype:",
	// "notes:", "compositional sub-alignment:".
	// In Go we check the report fields directly.
	compact := chunkEntry{chkSeg("k", "t"), chkSeg("t", "ʃ"), -4.9}
	model := chkMakeModel([]chunkEntry{compact})
	reports := analyzePromotedChunks(model)
	if len(reports) == 0 {
		t.Fatal("expected at least one chunk report")
	}
	r := reports[0]
	if r.TransparencyScore < 0.0 || r.TransparencyScore > 1.0 {
		t.Errorf("TransparencyScore %v out of [0,1]", r.TransparencyScore)
	}
	if r.ProcessProfile == "" {
		t.Error("ProcessProfile is empty")
	}
	if r.ProcessSubtype == "" {
		t.Error("ProcessSubtype is empty")
	}
	_ = r.Notes
	if len(r.SubAlignment.Links) == 0 {
		t.Error("SubAlignment has no links (expected compositional sub-alignment)")
	}
}

func TestChunkDiagnosticsTransparencyPrefersCompactFusionalReflexOverBundle(t *testing.T) {
	compact := chunkEntry{chkSeg("s", "k"), chkSeg("ʃ"), -4.9}
	bundled := chunkEntry{chkSeg("i", "h", "t"), chkSeg("aɪ", "t"), -7.4}
	model := chkMakeModel([]chunkEntry{compact, bundled})
	reports := analyzePromotedChunks(model)
	byPair := map[string]ChunkTransparencyReport{}
	for _, r := range reports {
		byPair[chunkPairKey(r.SrcChunk, r.TgtChunk)] = r
	}
	compactR := byPair[chunkPairKey(compact.src, compact.tgt)]
	bundledR := byPair[chunkPairKey(bundled.src, bundled.tgt)]
	if compactR.TransparencyScore <= bundledR.TransparencyScore {
		t.Errorf("compact_fusion score %v should be > bundled_reduction score %v", compactR.TransparencyScore, bundledR.TransparencyScore)
	}
	if compactR.ProcessProfile != "compact_fusion" {
		t.Errorf("compact ProcessProfile = %q, want compact_fusion", compactR.ProcessProfile)
	}
	hasOpaque := false
	for _, n := range bundledR.Notes {
		if strings.Contains(n, "historically opaque candidate") {
			hasOpaque = true
		}
	}
	if !hasOpaque {
		t.Errorf("bundled notes %v missing 'historically opaque candidate'", bundledR.Notes)
	}
	if bundledR.ProcessProfile != "bundled_reduction" {
		t.Errorf("bundled ProcessProfile = %q, want bundled_reduction", bundledR.ProcessProfile)
	}
}

func TestChunkDiagnosticsTransparencyDistinguishesFusionalFromResidualReduction(t *testing.T) {
	fusional := chunkEntry{chkSeg("a", "n"), chkSeg("ɑ̃"), -4.9}
	residual := chunkEntry{chkSeg("k", "w"), chkSeg("k"), -4.9}
	model := chkMakeModel([]chunkEntry{fusional, residual})
	reports := analyzePromotedChunks(model)
	byPair := map[string]ChunkTransparencyReport{}
	for _, r := range reports {
		byPair[chunkPairKey(r.SrcChunk, r.TgtChunk)] = r
	}
	fusionalR := byPair[chunkPairKey(fusional.src, fusional.tgt)]
	residualR := byPair[chunkPairKey(residual.src, residual.tgt)]
	if fusionalR.ProcessProfile != "nasal_fusion" {
		t.Errorf("fusional ProcessProfile = %q, want nasal_fusion", fusionalR.ProcessProfile)
	}
	if residualR.ProcessProfile != "residual_reduction" {
		t.Errorf("residual ProcessProfile = %q, want residual_reduction", residualR.ProcessProfile)
	}
}

func TestChunkDiagnosticsTransparencyRecognizesGlideOrVocalizationFusion(t *testing.T) {
	glide := chunkEntry{chkSeg("u", "l"), chkSeg("j"), -4.9}
	bundled := chunkEntry{chkSeg("a", "t", "e"), chkSeg("ɛ"), -7.4}
	model := chkMakeModel([]chunkEntry{glide, bundled})
	reports := analyzePromotedChunks(model)
	byPair := map[string]ChunkTransparencyReport{}
	for _, r := range reports {
		byPair[chunkPairKey(r.SrcChunk, r.TgtChunk)] = r
	}
	glideR := byPair[chunkPairKey(glide.src, glide.tgt)]
	bundledR := byPair[chunkPairKey(bundled.src, bundled.tgt)]
	if glideR.ProcessProfile != "glide_or_vocalization_fusion" {
		t.Errorf("glide ProcessProfile = %q, want glide_or_vocalization_fusion", glideR.ProcessProfile)
	}
	if bundledR.ProcessProfile != "bundled_reduction" {
		t.Errorf("bundled ProcessProfile = %q, want bundled_reduction", bundledR.ProcessProfile)
	}
	if glideR.ProcessSubtype != "glide_formation_or_vocalization" {
		t.Errorf("glide ProcessSubtype = %q, want glide_formation_or_vocalization", glideR.ProcessSubtype)
	}
}

func TestChunkDiagnosticsChunkProcessFamilySummaryAggregatesRecurrentProfiles(t *testing.T) {
	model := chkMakeModel([]chunkEntry{
		{chkSeg("a", "n"), chkSeg("ɑ̃"), -5.0},
		{chkSeg("e", "n"), chkSeg("ɛ̃"), -4.8},
		{chkSeg("k", "w"), chkSeg("k"), -4.7},
		{chkSeg("a", "t", "e"), chkSeg("ɛ"), -7.4},
	})
	families := summarizeChunkProcessFamilies(model)
	if len(families) == 0 {
		t.Fatal("expected at least one family report")
	}
	byProfile := map[string]ChunkProcessFamilyReport{}
	for _, f := range families {
		byProfile[f.ProcessProfile] = f
	}
	nasalF, hasNasal := byProfile["nasal_fusion"]
	bundledF, hasBundled := byProfile["bundled_reduction"]
	if !hasNasal {
		t.Fatalf("expected nasal_fusion in families; got: %v", families)
	}
	if !hasBundled {
		t.Fatalf("expected bundled_reduction in families; got: %v", families)
	}
	if nasalF.ChunkCount != 2 {
		t.Errorf("nasal_fusion ChunkCount = %d, want 2", nasalF.ChunkCount)
	}
	if nasalF.WeightedSupport <= bundledF.WeightedSupport {
		t.Errorf("nasal_fusion weighted_support %v should > bundled_reduction %v", nasalF.WeightedSupport, bundledF.WeightedSupport)
	}
	hasVowelNasalization := false
	for _, e := range nasalF.EvidenceSignatures {
		if strings.Contains(e, "vowel gained nasalization") {
			hasVowelNasalization = true
		}
	}
	if !hasVowelNasalization {
		t.Errorf("nasal_fusion evidence %v missing 'vowel gained nasalization'", nasalF.EvidenceSignatures)
	}

	subtypes := summarizeChunkProcessSubtypes(model)
	if len(subtypes) == 0 {
		t.Fatal("expected at least one subtype report")
	}
	type subtypeKey struct{ profile, subtype string }
	bySubtype := map[subtypeKey]ChunkProcessSubtypeReport{}
	for _, s := range subtypes {
		bySubtype[subtypeKey{s.ProcessProfile, s.ProcessSubtype}] = s
	}
	nasalSub, hasNasalSub := bySubtype[subtypeKey{"nasal_fusion", "vowel_nasalization_with_consonant_absorption"}]
	if !hasNasalSub {
		t.Fatalf("expected nasal_fusion/vowel_nasalization_with_consonant_absorption; got: %v", bySubtype)
	}
	if nasalSub.ChunkCount != 2 {
		t.Errorf("nasal_fusion/vowel_nasalization_with_consonant_absorption ChunkCount = %d, want 2", nasalSub.ChunkCount)
	}
}

func TestChunkDiagnosticsDescribeChunkProcessFamiliesMentionsProfilesAndExamples(t *testing.T) {
	model := chkMakeModel([]chunkEntry{
		{chkSeg("k", "t"), chkSeg("t", "ʃ"), -4.9},
		{chkSeg("k", "w"), chkSeg("k"), -4.8},
	})
	families := summarizeChunkProcessFamilies(model)
	if len(families) == 0 {
		t.Fatal("expected at least one family")
	}
	hasCompactOrResidual := false
	for _, f := range families {
		if f.ProcessProfile == "compact_fusion" || f.ProcessProfile == "residual_reduction" {
			hasCompactOrResidual = true
		}
	}
	if !hasCompactOrResidual {
		profiles := make([]string, len(families))
		for i, f := range families {
			profiles[i] = f.ProcessProfile
		}
		t.Errorf("expected compact_fusion or residual_reduction in %v", profiles)
	}
	hasExamples := false
	for _, f := range families {
		if len(f.RepresentativeChunks) > 0 {
			hasExamples = true
		}
	}
	if !hasExamples {
		t.Error("expected at least one family with representative chunks (examples)")
	}

	subtypes := summarizeChunkProcessSubtypes(model)
	if len(subtypes) == 0 {
		t.Fatal("expected at least one subtype")
	}
	hasSlash := false
	for _, s := range subtypes {
		if s.ProcessProfile != "" && s.ProcessSubtype != "" {
			hasSlash = true
		}
	}
	if !hasSlash {
		t.Error("expected at least one subtype with both profile and subtype")
	}

	opts := FormatModelOptions{
		TopSegments:             15,
		TopDisplacements:        5,
		MinCount:                1,
		SummarizeChunkProcesses: true,
		SummarizeChunkSubtypes:  true,
	}
	out := FormatModel(model, opts)
	if !strings.Contains(out, "Chunk process families") {
		t.Errorf("FormatModel output missing 'Chunk process families'")
	}
	if !strings.Contains(out, "Chunk process subtypes") {
		t.Errorf("FormatModel output missing 'Chunk process subtypes'")
	}
	if !strings.Contains(out, "/") {
		t.Errorf("FormatModel output missing '/' (profile/subtype separator)")
	}
}

func TestChunkDiagnosticsChunkSubtypesSplitBroadProfilesIntoNarrowerGroups(t *testing.T) {
	model := chkMakeModel([]chunkEntry{
		{chkSeg("c", "e"), chkSeg("θ"), -4.9},
		{chkSeg("i"), chkSeg("w", "a"), -4.8},
		{chkSeg("b", "o"), chkSeg("b"), -4.8},
		{chkSeg("e"), chkSeg("j", "e"), -4.8},
	})
	reports := analyzePromotedChunks(model)
	byPair := map[string]ChunkTransparencyReport{}
	for _, r := range reports {
		byPair[chunkPairKey(r.SrcChunk, r.TgtChunk)] = r
	}

	ceθKey := chunkPairKey(chkSeg("c", "e"), chkSeg("θ"))
	iwaKey := chunkPairKey(chkSeg("i"), chkSeg("w", "a"))
	boBKey := chunkPairKey(chkSeg("b", "o"), chkSeg("b"))
	ejeKey := chunkPairKey(chkSeg("e"), chkSeg("j", "e"))

	ceθR, ok1 := byPair[ceθKey]
	iwaR, ok2 := byPair[iwaKey]
	boBR, ok3 := byPair[boBKey]
	ejeR, ok4 := byPair[ejeKey]

	if !ok1 || !ok2 || !ok3 || !ok4 {
		present := make([]string, 0, len(byPair))
		for k := range byPair {
			present = append(present, fmt.Sprintf("%q", k))
		}
		t.Fatalf("missing reports; present keys: %v", present)
	}

	if ceθR.ProcessSubtype != "fricativizing_or_affricating_fusion" {
		t.Errorf("c,e->θ ProcessSubtype = %q, want fricativizing_or_affricating_fusion", ceθR.ProcessSubtype)
	}
	if iwaR.ProcessSubtype != "insertional_glide_or_diphthongal_fusion" {
		t.Errorf("i->w,a ProcessSubtype = %q, want insertional_glide_or_diphthongal_fusion", iwaR.ProcessSubtype)
	}
	if boBR.ProcessSubtype != "consonant_residue_with_vowel_loss" {
		t.Errorf("b,o->b ProcessSubtype = %q, want consonant_residue_with_vowel_loss", boBR.ProcessSubtype)
	}
	if ejeR.ProcessSubtype != "residual_expansion" {
		t.Errorf("e->j,e ProcessSubtype = %q, want residual_expansion", ejeR.ProcessSubtype)
	}
}

func TestChunkDiagnosticsChunkProcessSubtypesSurfaceDiscoveredContexts(t *testing.T) {
	chunk := chunkEntry{chkSeg("c", "e"), chkSeg("θ"), -4.9}
	model := chkMakeModel([]chunkEntry{chunk})
	// Inject a conditioned correspondence c->θ with following=[front:+].
	conditioned := ConditionedCorrespondence{
		Src: "c",
		Tgt: "θ",
		Context: Context{
			Following: []FeatureConstraint{{Feature: "front", Value: "+"}},
		},
	}
	model.SegmentTable.setCount(conditioned, 6.0)
	model.SegmentTable.SrcTotals["c"] = 6.0

	subtypes := summarizeChunkProcessSubtypes(model)
	type subtypeKey struct{ profile, subtype string }
	bySubtype := map[subtypeKey]ChunkProcessSubtypeReport{}
	for _, s := range subtypes {
		bySubtype[subtypeKey{s.ProcessProfile, s.ProcessSubtype}] = s
	}
	sk := subtypeKey{"compact_fusion", "fricativizing_or_affricating_fusion"}
	report, ok := bySubtype[sk]
	if !ok {
		t.Fatalf("expected compact_fusion/fricativizing_or_affricating_fusion; got: %v", bySubtype)
	}
	wantCtx := "c->θ / _[front:+]"
	found := false
	for _, ctx := range report.ContextSignatures {
		if ctx == wantCtx {
			found = true
		}
	}
	if !found {
		t.Errorf("context signatures %v missing %q", report.ContextSignatures, wantCtx)
	}

	opts := FormatModelOptions{
		TopSegments:            15,
		TopDisplacements:       5,
		MinCount:               1,
		SummarizeChunkSubtypes: true,
	}
	out := FormatModel(model, opts)
	if !strings.Contains(out, "contexts=") {
		t.Errorf("FormatModel output missing 'contexts=': %s", out)
	}
	if !strings.Contains(out, "c->θ / _[front:+]") {
		t.Errorf("FormatModel output missing context signature %q: %s", wantCtx, out)
	}
}
