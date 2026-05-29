package regulae

// Mirror of python/tests/test_suprasegmentals.py

import (
	"strings"
	"testing"
)

func TestSuprasegmentalsTonedSegmentsSurviveSearchUnchanged(t *testing.T) {
	// A segment entering the search with a tone annotation must exit with
	// the same tone annotation in the corresponding link.
	src := Form{
		LectID: "hmong_a",
		Segments: []Segment{
			{Grapheme: "p"},
			{Grapheme: "a", Tone: "55"},
		},
	}
	tgt := Form{
		LectID: "hmong_b",
		Segments: []Segment{
			{Grapheme: "p"},
			{Grapheme: "a", Tone: "33"},
		},
	}
	al, err := AlignForms(src, tgt, "descriptive", 0, nil)
	if err != nil {
		t.Fatalf("AlignForms: %v", err)
	}
	// Find the link where the source chunk is [a].
	var toneLink *Link
	for i := range al.Links {
		lk := &al.Links[i]
		if len(lk.SourceChunk) > 0 && lk.SourceChunk[0].Grapheme == "a" {
			toneLink = lk
			break
		}
	}
	if toneLink == nil {
		t.Fatal("could not find the a~a link in the alignment")
	}
	if toneLink.SourceChunk[0].Tone != "55" {
		t.Errorf("source tone = %q, want %q", toneLink.SourceChunk[0].Tone, "55")
	}
	if toneLink.TargetChunk[0].Tone != "33" {
		t.Errorf("target tone = %q, want %q", toneLink.TargetChunk[0].Tone, "33")
	}
}

func TestSuprasegmentalsStressAndLengthSurviveSearch(t *testing.T) {
	src := Form{
		LectID: "A",
		Segments: []Segment{
			{Grapheme: "a", Length: "long", Stress: "primary"},
			{Grapheme: "b"},
		},
	}
	tgt := Form{
		LectID: "B",
		Segments: []Segment{
			{Grapheme: "a", Length: "short"},
			{Grapheme: "b"},
		},
	}
	al, err := AlignForms(src, tgt, "descriptive", 0, nil)
	if err != nil {
		t.Fatalf("AlignForms: %v", err)
	}
	// The first link should be a~a.
	if len(al.Links) == 0 {
		t.Fatal("no links in alignment")
	}
	aLink := al.Links[0]
	if len(aLink.SourceChunk) == 0 || aLink.SourceChunk[0].Grapheme != "a" {
		t.Fatalf("first link source is not 'a': %v", aLink.SourceChunk)
	}
	if aLink.SourceChunk[0].Length != "long" {
		t.Errorf("source Length = %q, want %q", aLink.SourceChunk[0].Length, "long")
	}
	if aLink.SourceChunk[0].Stress != "primary" {
		t.Errorf("source Stress = %q, want %q", aLink.SourceChunk[0].Stress, "primary")
	}
	if len(aLink.TargetChunk) == 0 || aLink.TargetChunk[0].Grapheme != "a" {
		t.Fatalf("first link target is not 'a': %v", aLink.TargetChunk)
	}
	if aLink.TargetChunk[0].Length != "short" {
		t.Errorf("target Length = %q, want %q", aLink.TargetChunk[0].Length, "short")
	}
	if aLink.TargetChunk[0].Stress != "" {
		t.Errorf("target Stress = %q, want empty string (absent)", aLink.TargetChunk[0].Stress)
	}
}

func TestSuprasegmentalsTonedSegmentsRenderInFormatter(t *testing.T) {
	// Suprasegmentals should show up in FormatAlignment output.
	src := Form{LectID: "A", Segments: []Segment{{Grapheme: "a", Tone: "high"}}}
	tgt := Form{LectID: "B", Segments: []Segment{{Grapheme: "a", Tone: "low"}}}
	al, err := AlignForms(src, tgt, "descriptive", 0, nil)
	if err != nil {
		t.Fatalf("AlignForms: %v", err)
	}
	out := FormatAlignment(al, "descriptive", false, false)
	if !strings.Contains(out, "T=high") {
		t.Errorf("FormatAlignment missing 'T=high':\n%s", out)
	}
	if !strings.Contains(out, "T=low") {
		t.Errorf("FormatAlignment missing 'T=low':\n%s", out)
	}
}

func TestSuprasegmentalsToneDifferenceDoesNotAffectAlignmentCost(t *testing.T) {
	// ROADMAP: scoring ignores suprasegmentals. Two forms differing only in
	// tone have zero alignment cost. This will change when suprasegmentals
	// enter the scoring model.
	src := Form{LectID: "A", Segments: []Segment{{Grapheme: "a", Tone: "high"}}}
	tgt := Form{LectID: "B", Segments: []Segment{{Grapheme: "a", Tone: "low"}}}
	al, err := AlignForms(src, tgt, "descriptive", 0, nil)
	if err != nil {
		t.Fatalf("AlignForms: %v", err)
	}
	cost, err := AlignmentCost(al, "descriptive", nil)
	if err != nil {
		t.Fatalf("AlignmentCost: %v", err)
	}
	if cost != 0.0 {
		t.Errorf("cost = %v for forms differing only in tone; want 0.0", cost)
	}
}
