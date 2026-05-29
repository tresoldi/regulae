package regulae

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "cognates.tsv")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadCognatesFromTSVBasic(t *testing.T) {
	tsv := "cognate_id\tlect_id\tsegments\tconfidence\n" +
		"c1\tA\tp a\t1.0\n" +
		"c1\tB\tf a\t1.0\n" +
		"c2\tA\tp i\t1.0\n" +
		"c2\tB\tf i\t0.5\n"
	path := writeTemp(t, tsv)
	corpus, err := LoadCognatesFromTSV(path, TSVLoadOptions{ConfidenceCol: "confidence"})
	if err != nil {
		t.Fatal(err)
	}
	if len(corpus) != 2 {
		t.Fatalf("expected 2 cognate sets, got %d", len(corpus))
	}
	// First-appearance order preserved.
	if corpus[0].CognateID != "c1" || corpus[1].CognateID != "c2" {
		t.Errorf("cognate order not preserved: %q, %q", corpus[0].CognateID, corpus[1].CognateID)
	}
	c1 := corpus[0]
	if len(c1.Forms) != 2 {
		t.Errorf("c1 should have 2 forms, got %d", len(c1.Forms))
	}
	if got := c1.Forms["A"].Segments; len(got) != 2 || got[0].Grapheme != "p" || got[1].Grapheme != "a" {
		t.Errorf("c1/A segments wrong: %v", got)
	}
	// FormsOrder records insertion order A then B.
	if len(c1.FormsOrder) != 2 || c1.FormsOrder[0] != "A" || c1.FormsOrder[1] != "B" {
		t.Errorf("FormsOrder wrong: %v", c1.FormsOrder)
	}
	// c2 confidence is the minimum across its rows (1.0, 0.5) = 0.5.
	if corpus[1].Confidence != 0.5 {
		t.Errorf("c2 confidence = %v, want 0.5 (pessimistic min)", corpus[1].Confidence)
	}
}

func TestLoadCognatesFromTSVMissingColumn(t *testing.T) {
	path := writeTemp(t, "cognate_id\tlect_id\n c1\tA\n")
	if _, err := LoadCognatesFromTSV(path, TSVLoadOptions{}); err == nil {
		t.Error("expected error for missing 'segments' column")
	}
}

func TestLoadCognatesFromTSVDropsGapsAndDashSegments(t *testing.T) {
	tsv := "cognate_id\tlect_id\tsegments\n" +
		"c1\tA\tp - a\n" +
		"c1\tB\tf a\n"
	path := writeTemp(t, tsv)
	corpus, err := LoadCognatesFromTSV(path, TSVLoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got := corpus[0].Forms["A"].Segments; len(got) != 2 {
		t.Errorf("dash gap should be dropped: got %v", got)
	}
}
