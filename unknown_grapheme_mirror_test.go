package regulae

// Mirror of python/tests/test_unknown_grapheme.py
//
// Existing covered tests (skip):
//   TestComputeDisplacementUnknownGrapheme — already in scoring_test.go

import (
	"errors"
	"strings"
	"testing"
)

// ----- score_link on unknown graphemes -----------------------------------

func TestUnknownGraphemeScoreLinkRaisesOnUnknownSource(t *testing.T) {
	link := Link{SourceChunk: []Segment{seg("ZZZ")}, TargetChunk: []Segment{seg("p")}, Confidence: 1.0}
	_, err := ScoreLink(link, "descriptive", nil)
	if err == nil {
		t.Fatal("expected error for unknown source grapheme")
	}
	var ug *UnknownGraphemeError
	if !errors.As(err, &ug) {
		t.Fatalf("expected *UnknownGraphemeError, got %T: %v", err, err)
	}
	if ug.Grapheme != "ZZZ" {
		t.Errorf("grapheme = %q, want ZZZ", ug.Grapheme)
	}
	if !strings.Contains(err.Error(), "ZZZ") {
		t.Errorf("error message %q does not contain 'ZZZ'", err.Error())
	}
}

func TestUnknownGraphemeScoreLinkRaisesOnUnknownTarget(t *testing.T) {
	link := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("ZZZ")}, Confidence: 1.0}
	_, err := ScoreLink(link, "descriptive", nil)
	if err == nil {
		t.Fatal("expected error for unknown target grapheme")
	}
	var ug *UnknownGraphemeError
	if !errors.As(err, &ug) {
		t.Fatalf("expected *UnknownGraphemeError, got %T: %v", err, err)
	}
	if ug.Grapheme != "ZZZ" {
		t.Errorf("grapheme = %q, want ZZZ", ug.Grapheme)
	}
}

func TestUnknownGraphemeScoreLinkMentionsFeatureSystemInError(t *testing.T) {
	// The error carries the feature system name.
	link := Link{SourceChunk: []Segment{seg("ZZZ")}, TargetChunk: []Segment{seg("p")}, Confidence: 1.0}
	_, err := ScoreLink(link, "distinctive", nil)
	if err == nil {
		t.Fatal("expected error for unknown grapheme")
	}
	var ug *UnknownGraphemeError
	if !errors.As(err, &ug) {
		t.Fatalf("expected *UnknownGraphemeError, got %T", err)
	}
	if ug.FeatureSystem != "distinctive" {
		t.Errorf("feature_system = %q, want 'distinctive'", ug.FeatureSystem)
	}
}

func TestUnknownGraphemeScoreLinkFineWithValidGraphemes(t *testing.T) {
	// Regression: the unknown-grapheme check must not break the happy path.
	link := Link{SourceChunk: []Segment{seg("p")}, TargetChunk: []Segment{seg("f")}, Confidence: 1.0}
	cost, err := ScoreLink(link, "descriptive", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cost <= 0.0 {
		t.Errorf("cost = %v, want > 0", cost)
	}
}

// ----- compute_displacement on unknown graphemes -------------------------

func TestUnknownGraphemeComputeDisplacementRaisesOnUnknownSource(t *testing.T) {
	_, err := ComputeDisplacement(seg("ZZZ"), seg("p"), "descriptive")
	if err == nil {
		t.Fatal("expected error for unknown source grapheme")
	}
	var ug *UnknownGraphemeError
	if !errors.As(err, &ug) {
		t.Fatalf("expected *UnknownGraphemeError, got %T", err)
	}
	if ug.Grapheme != "ZZZ" {
		t.Errorf("grapheme = %q, want ZZZ", ug.Grapheme)
	}
}

func TestUnknownGraphemeComputeDisplacementRaisesOnUnknownTarget(t *testing.T) {
	_, err := ComputeDisplacement(seg("p"), seg("ZZZ"), "descriptive")
	if err == nil {
		t.Fatal("expected error for unknown target grapheme")
	}
	var ug *UnknownGraphemeError
	if !errors.As(err, &ug) {
		t.Fatalf("expected *UnknownGraphemeError, got %T", err)
	}
	if ug.Grapheme != "ZZZ" {
		t.Errorf("grapheme = %q, want ZZZ", ug.Grapheme)
	}
}

// ----- align_forms on unknown graphemes ----------------------------------

func TestUnknownGraphemeAlignFormsRaisesCleanly(t *testing.T) {
	// The search should propagate UnknownGraphemeError.
	src := Form{LectID: "A", Segments: []Segment{seg("p"), seg("ZZZ"), seg("t")}}
	tgt := Form{LectID: "B", Segments: []Segment{seg("p"), seg("a"), seg("t")}}
	_, err := AlignForms(src, tgt, "descriptive", 0, nil)
	if err == nil {
		t.Fatal("expected error for unknown grapheme in source form")
	}
	var ug *UnknownGraphemeError
	if !errors.As(err, &ug) {
		t.Fatalf("expected *UnknownGraphemeError, got %T: %v", err, err)
	}
	if ug.Grapheme != "ZZZ" {
		t.Errorf("grapheme = %q, want ZZZ", ug.Grapheme)
	}
}

func TestUnknownGraphemeIsAValueError(t *testing.T) {
	// COMMITMENT: UnknownGraphemeError embeds / implements error and can be
	// caught as a *UnknownGraphemeError. In Go we check that a value of
	// *UnknownGraphemeError is an error (satisfied by implementing Error()).
	// There is no ValueErrorinheritance in Go, but the commitment is that
	// callers receive it as a standard error interface value, which is
	// verified by the other tests above. This test confirms the type assertion.
	var e error = &UnknownGraphemeError{Grapheme: "Q", FeatureSystem: "descriptive"}
	var ug *UnknownGraphemeError
	if !errors.As(e, &ug) {
		t.Error("*UnknownGraphemeError should satisfy errors.As(*UnknownGraphemeError)")
	}
	if ug.Grapheme != "Q" {
		t.Errorf("grapheme = %q, want Q", ug.Grapheme)
	}
}
