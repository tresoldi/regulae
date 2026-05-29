package regulae

// Mirror of python/tests/test_format.py
//
// Skipped tests (already in existing Go suite):
//   TestFormatAlignmentRendersLinks — format_test.go::TestFormatAlignmentRendersLinks
//
// Python test test_format_link_basic is mapped to TestFormatFormatLinkShowsBothChunks
// (the existing TestFormatAlignmentRendersLinks covers the overall alignment output).

import (
	"strings"
	"testing"
)

// ----- format_segments ----------------------------------------------------

func TestFormatFormatSegmentsEmptyUsesEpsilonMarker(t *testing.T) {
	out := FormatSegments(nil)
	if out != emptyChunkSymbol {
		t.Errorf("FormatSegments(nil) = %q, want %q", out, emptyChunkSymbol)
	}
}

func TestFormatFormatSegmentsConcatenatesGraphemes(t *testing.T) {
	segs := []Segment{seg("p"), seg("a"), seg("t")}
	out := FormatSegments(segs)
	if out != "pat" {
		t.Errorf("FormatSegments = %q, want %q", out, "pat")
	}
}

func TestFormatFormatSegmentsShowsToneAnnotation(t *testing.T) {
	segs := []Segment{{Grapheme: "a", Tone: "high"}}
	out := FormatSegments(segs)
	if !strings.Contains(out, "a") {
		t.Errorf("output %q missing grapheme 'a'", out)
	}
	if !strings.Contains(out, "T=high") {
		t.Errorf("output %q missing tone annotation 'T=high'", out)
	}
}

func TestFormatFormatSegmentsShowsLengthAndStress(t *testing.T) {
	segs := []Segment{{Grapheme: "a", Length: "long", Stress: "primary"}}
	out := FormatSegments(segs)
	if !strings.Contains(out, "L=long") {
		t.Errorf("output %q missing 'L=long'", out)
	}
	if !strings.Contains(out, "S=primary") {
		t.Errorf("output %q missing 'S=primary'", out)
	}
}

func TestFormatFormatSegmentsCombinesMultipleAnnotations(t *testing.T) {
	segs := []Segment{{Grapheme: "a", Tone: "high", Length: "long"}}
	out := FormatSegments(segs)
	if !strings.Contains(out, "T=high") {
		t.Errorf("output %q missing 'T=high'", out)
	}
	if !strings.Contains(out, "L=long") {
		t.Errorf("output %q missing 'L=long'", out)
	}
}

// ----- format_link --------------------------------------------------------

func TestFormatFormatLinkShowsBothChunksWithSeparator(t *testing.T) {
	// This is the "basic" link test.
	link := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("f")}, Confidence: 1.0}
	out := FormatLink(link)
	if !strings.Contains(out, "p") {
		t.Errorf("output %q missing 'p'", out)
	}
	if !strings.Contains(out, "f") {
		t.Errorf("output %q missing 'f'", out)
	}
	if !strings.Contains(out, "~") {
		t.Errorf("output %q missing '~'", out)
	}
}

func TestFormatFormatLinkEmptySourceUsesEpsilon(t *testing.T) {
	link := Link{SourceChunk: nil, TargetChunk: []Segment{seg("e")}, Confidence: 1.0}
	out := FormatLink(link)
	if !strings.Contains(out, emptyChunkSymbol) {
		t.Errorf("output %q missing epsilon symbol", out)
	}
	if !strings.Contains(out, "e") {
		t.Errorf("output %q missing 'e'", out)
	}
}

func TestFormatFormatLinkEmptyTargetUsesEpsilon(t *testing.T) {
	link := Link{SourceChunk: []Segment{seg("e")}, TargetChunk: nil, Confidence: 1.0}
	out := FormatLink(link)
	if !strings.Contains(out, emptyChunkSymbol) {
		t.Errorf("output %q missing epsilon symbol", out)
	}
}

func TestFormatFormatLinkHandlesMultiSegmentChunks(t *testing.T) {
	link := Link{
		SourceChunk: []Segment{seg("k"), seg("t")},
		TargetChunk: []Segment{seg("t"), seg("t")},
		Confidence:  1.0,
	}
	out := FormatLink(link)
	if !strings.Contains(out, "kt") {
		t.Errorf("output %q missing 'kt'", out)
	}
	if !strings.Contains(out, "tt") {
		t.Errorf("output %q missing 'tt'", out)
	}
}

// ----- format_alignment ---------------------------------------------------

func TestFormatFormatAlignmentHasHeaderWithLectIDs(t *testing.T) {
	al := mustAlign(t,
		Form{LectID: "latin", Segments: []Segment{seg("p"), seg("a")}},
		Form{LectID: "gothic", Segments: []Segment{seg("f"), seg("a")}},
		0,
	)
	out := FormatAlignment(al, "descriptive", false, false)
	if !strings.Contains(out, "latin") {
		t.Errorf("output %q missing 'latin'", out)
	}
	if !strings.Contains(out, "gothic") {
		t.Errorf("output %q missing 'gothic'", out)
	}
}

func TestFormatFormatAlignmentShowsTotalCostByDefault(t *testing.T) {
	al := mustAlign(t, formIPA("A", "p"), formIPA("B", "f"), 0)
	out := FormatAlignment(al, "descriptive", true, false)
	if !strings.Contains(strings.ToLower(out), "cost") {
		t.Errorf("output %q missing cost", out)
	}
}

func TestFormatFormatAlignmentHidesCostWhenRequested(t *testing.T) {
	al := mustAlign(t, formIPA("A", "p"), formIPA("B", "f"), 0)
	out := FormatAlignment(al, "descriptive", false, false)
	if strings.Contains(strings.ToLower(out), "cost") {
		t.Errorf("output %q unexpectedly contains 'cost' with showCosts=false", out)
	}
}

func TestFormatFormatAlignmentShowsDisplacementWhenRequested(t *testing.T) {
	al := mustAlign(t, formIPA("A", "p"), formIPA("B", "f"), 0)
	out := FormatAlignment(al, "descriptive", false, true)
	// At least one feature name with "->" arrow should appear.
	if !strings.Contains(out, "->") {
		t.Errorf("output %q missing displacement arrow '->'", out)
	}
}

func TestFormatFormatAlignmentHasOneLinePerLink(t *testing.T) {
	al := mustAlign(t, formIPA("A", "pat"), formIPA("B", "fad"), 0)
	out := FormatAlignment(al, "descriptive", false, false)
	lines := strings.Split(out, "\n")
	// header + 3 links = 4 lines
	wantLines := 1 + len(al.Links)
	if len(lines) != wantLines {
		t.Errorf("line count = %d, want %d (header + %d links)\n%s",
			len(lines), wantLines, len(al.Links), out)
	}
}

func TestFormatFormatAlignmentWithEmptyAlignment(t *testing.T) {
	// Empty alignment (both forms empty) should format without crashing.
	al := Alignment{
		SourceForm: Form{LectID: "A"},
		TargetForm: Form{LectID: "B"},
		Links:      nil,
	}
	out := FormatAlignment(al, "descriptive", false, false)
	if !strings.Contains(out, "A") {
		t.Errorf("output %q missing 'A'", out)
	}
	if !strings.Contains(out, "B") {
		t.Errorf("output %q missing 'B'", out)
	}
}
