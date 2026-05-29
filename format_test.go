package regulae

import (
	"strings"
	"testing"
)

func TestFormatModelContainsCorrespondence(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pa", "B", "fa"),
		pairOf("A", "pa", "B", "fa"),
		pairOf("A", "pa", "B", "fa"),
	}
	model := mustTrain(t, corpus)
	out := FormatModel(model, DefaultFormatModelOptions())
	if !strings.Contains(out, "LearnedModel") {
		t.Error("format output missing header")
	}
	if !strings.Contains(out, "p -> f") {
		t.Errorf("format output missing the p -> f correspondence:\n%s", out)
	}
}

func TestFormatAlignmentRendersLinks(t *testing.T) {
	al := mustAlign(t, formIPA("A", "pater"), formIPA("B", "fadar"), 0)
	out := FormatAlignment(al, "descriptive", true, false)
	if !strings.Contains(out, "~") {
		t.Errorf("alignment output missing link rows:\n%s", out)
	}
	if !strings.Contains(out, "cost=") {
		t.Error("alignment output missing total cost")
	}
}

func TestFormatMultiLectModel(t *testing.T) {
	var corpus []CognateSet
	for i := 0; i < 5; i++ {
		corpus = append(corpus, csOf("c"+string(rune('0'+i)), []string{"A", "B", "C"},
			map[string]string{"A": "pa", "B": "fa", "C": "fa"}))
	}
	m := mustTrainML(t, corpus)
	out := FormatMultiLectModel(m, 20, 20)
	if !strings.Contains(out, "MultiLectModel") {
		t.Error("multi-lect format missing header")
	}
	if !strings.Contains(out, "lects (3)") {
		t.Errorf("multi-lect format missing lect count:\n%s", out)
	}
}

func TestFindCognateOutliers(t *testing.T) {
	var corpus []CognateSet
	for i := 0; i < 6; i++ {
		corpus = append(corpus, csOf("reg"+string(rune('0'+i)), []string{"A", "B"},
			map[string]string{"A": "pa", "B": "fa"}))
	}
	// One clear outlier: a pair that does not follow p->f at all.
	corpus = append(corpus, csOf("odd", []string{"A", "B"}, map[string]string{"A": "ki", "B": "wu"}))
	m := mustTrainML(t, corpus)
	reports, err := FindCognateOutliers(corpus, m, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(reports) == 0 {
		t.Fatal("expected outlier reports")
	}
	// The "odd" set should rank at or near the top (highest z-score).
	if reports[0].CognateID != "odd" {
		t.Errorf("expected 'odd' as top outlier, got %q (z=%.2f)", reports[0].CognateID, reports[0].ZScore)
	}
}
