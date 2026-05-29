package regulae

// Mirror of python/tests/test_diagnostics.py
//
// Skipped (already covered):
//   TestFindCognateOutliers — covered by the existing Go test in
//   the non-mirror test files (search the repo for the original).
//
// Actually: inspecting the skip list — TestFindCognateOutliers is listed
// in the global skip list. We port all other tests in this file.

import "testing"

func diagFormIPA(lect, word string) Form {
	segs := make([]Segment, len([]rune(word)))
	for i, c := range []rune(word) {
		segs[i] = Segment{Grapheme: string(c)}
	}
	return Form{LectID: lect, Segments: segs}
}

func diagRegularCorpus() []CognateSet {
	pairs := [][2]string{
		{"pata", "fada"},
		{"pita", "fida"},
		{"pota", "foda"},
		{"puta", "fuda"},
		{"peta", "feda"},
		{"pat", "fad"},
		{"pit", "fid"},
		{"pot", "fod"},
	}
	out := make([]CognateSet, len(pairs))
	for i, p := range pairs {
		out[i] = CognateSet{
			CognateID:  "r" + string(rune('0'+i)),
			Forms:      map[string]Form{"A": diagFormIPA("A", p[0]), "B": diagFormIPA("B", p[1])},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		}
	}
	return out
}

func TestDiagnosticsFindOutliersEmptyCorpusReturnsEmptyList(t *testing.T) {
	empty := EmptyMultiLectModel()
	reports, err := FindCognateOutliers(nil, empty, 0, 0)
	if err != nil {
		t.Fatalf("FindCognateOutliers error: %v", err)
	}
	if len(reports) != 0 {
		t.Errorf("expected empty list, got %d reports", len(reports))
	}
}

func TestDiagnosticsFindOutliersSingleCognateHasZeroZScore(t *testing.T) {
	corpus := diagRegularCorpus()[:1]
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	reports, err := FindCognateOutliers(corpus, model, 0, 0)
	if err != nil {
		t.Fatalf("FindCognateOutliers error: %v", err)
	}
	if len(reports) != 1 {
		t.Fatalf("expected 1 report, got %d", len(reports))
	}
	if reports[0].ZScore != 0.0 {
		t.Errorf("single cognate z_score = %v, want 0.0", reports[0].ZScore)
	}
}

func TestDiagnosticsFindOutliersDetectsPlantedNonCognate(t *testing.T) {
	regular := diagRegularCorpus()
	planted := CognateSet{
		CognateID:  "NON_COGNATE",
		Forms:      map[string]Form{"A": diagFormIPA("A", "xyz"), "B": diagFormIPA("B", "qmn")},
		FormsOrder: []string{"A", "B"},
		Confidence: 1.0,
	}
	corpus := append(regular, planted)
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	reports, err := FindCognateOutliers(corpus, model, 0, 0)
	if err != nil {
		t.Fatalf("FindCognateOutliers error: %v", err)
	}
	if len(reports) != len(corpus) {
		t.Errorf("expected %d reports, got %d", len(corpus), len(reports))
	}
	if reports[0].CognateID != "NON_COGNATE" {
		t.Errorf("top outlier CognateID = %q, want NON_COGNATE", reports[0].CognateID)
	}
	if reports[0].ZScore <= 0 {
		t.Errorf("top outlier z_score = %v, want > 0", reports[0].ZScore)
	}
	for _, r := range reports[1:] {
		if r.ZScore > reports[0].ZScore {
			t.Errorf("regular pair %q has z_score %v > top outlier %v", r.CognateID, r.ZScore, reports[0].ZScore)
		}
	}
}

func TestDiagnosticsFindOutliersReturnsAllReportsAsCognateOutlierReportInstances(t *testing.T) {
	corpus := diagRegularCorpus()
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	reports, err := FindCognateOutliers(corpus, model, 0, 0)
	if err != nil {
		t.Fatalf("FindCognateOutliers error: %v", err)
	}
	for _, r := range reports {
		if r.NPairs < 1 {
			t.Errorf("report %q has NPairs = %d, want >= 1", r.CognateID, r.NPairs)
		}
		// CostPerSegment can be negative (per spec).
		_ = r.CostPerSegment
		_ = r.ZScore
	}
}

func TestDiagnosticsFindOutliersTopKLimitsOutput(t *testing.T) {
	corpus := diagRegularCorpus()
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	reports, err := FindCognateOutliers(corpus, model, 3, 0)
	if err != nil {
		t.Fatalf("FindCognateOutliers error: %v", err)
	}
	if len(reports) != 3 {
		t.Errorf("top_k=3 returned %d reports, want 3", len(reports))
	}
}

func TestDiagnosticsFindOutliersIsADiagnosticNotAFilter(t *testing.T) {
	corpus := diagRegularCorpus()
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	r1, err1 := FindCognateOutliers(corpus, model, 0, 0)
	r2, err2 := FindCognateOutliers(corpus, model, 0, 0)
	if err1 != nil || err2 != nil {
		t.Fatalf("errors: %v / %v", err1, err2)
	}
	if len(r1) != len(r2) {
		t.Fatalf("run1 %d reports, run2 %d reports", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].CognateID != r2[i].CognateID {
			t.Errorf("[%d] cognate_id %q vs %q", i, r1[i].CognateID, r2[i].CognateID)
		}
		if r1[i].ZScore != r2[i].ZScore {
			t.Errorf("[%d] z_score %v vs %v", i, r1[i].ZScore, r2[i].ZScore)
		}
	}
}

func TestDiagnosticsFindOutliersIgnoresSetsWithNoTrainablePair(t *testing.T) {
	corpus := diagRegularCorpus()
	model, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	orphan := CognateSet{
		CognateID:  "ORPHAN",
		Forms:      map[string]Form{"X": diagFormIPA("X", "xyz"), "Y": diagFormIPA("Y", "zyx")},
		FormsOrder: []string{"X", "Y"},
		Confidence: 1.0,
	}
	reports, err := FindCognateOutliers([]CognateSet{orphan}, model, 0, 0)
	if err != nil {
		t.Fatalf("FindCognateOutliers error: %v", err)
	}
	if len(reports) != 0 {
		t.Errorf("expected empty list for orphan, got %d reports", len(reports))
	}
}
