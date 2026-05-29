package regulae

// Mirror of python/tests/test_multi_lect_format.py
//
// Skipped (already covered by existing Go suite):
//   TestFormatMultiLectModel — format_test.go::TestFormatMultiLectModel

import (
	"strings"
	"testing"
)

// ----- FormatMultiLectModel -----------------------------------------------

func TestMultiLectFormatFormatEmptyModel(t *testing.T) {
	// COMMITMENT: an empty model renders without crashing and shows explicit
	// 'none' markers for both class tables.
	out := FormatMultiLectModel(EmptyMultiLectModel(), 0, 0)
	if !strings.Contains(out, "MultiLectModel") {
		t.Errorf("output missing 'MultiLectModel':\n%s", out)
	}
	if !strings.Contains(out, "lects (0)") {
		t.Errorf("output missing 'lects (0)':\n%s", out)
	}
	if !strings.Contains(out, "(none") {
		t.Errorf("output missing '(none...' marker:\n%s", out)
	}
}

func TestMultiLectFormatFormatN2ModelShowsLectIDsAndClasses(t *testing.T) {
	var pairs []FormPair
	for i := 0; i < 4; i++ {
		pairs = append(pairs, pairOf("A", "pa", "B", "fa"))
	}
	corpus := CognateSetsFromPairs(pairs, [2]string{"A", "B"}, "")
	m := mustTrainML(t, corpus)
	out := FormatMultiLectModel(m, 0, 0)
	if !strings.Contains(out, "A") {
		t.Errorf("output missing 'A':\n%s", out)
	}
	if !strings.Contains(out, "B") {
		t.Errorf("output missing 'B':\n%s", out)
	}
	if !strings.Contains(out, "pairwise models:") {
		t.Errorf("output missing 'pairwise models:':\n%s", out)
	}
	if !strings.Contains(out, "unconditioned cls:") {
		t.Errorf("output missing 'unconditioned cls:':\n%s", out)
	}
}

func TestMultiLectFormatFormatTruncatesToTopN(t *testing.T) {
	// COMMITMENT: top_unconditioned caps the number of unconditioned classes
	// shown and mentions how many more exist.
	var classes []MultiLectCorrespondenceClass
	for i := 0; i < 5; i++ {
		g := string(rune('a' + i))
		classes = append(classes, MultiLectCorrespondenceClass{
			ClassID:  i,
			Segments: map[string]string{"A": g, "B": g},
			Count:    float64(10 - i),
		})
	}
	m := &MultiLectModel{
		PairwiseModels:       map[string]*LearnedModel{},
		UnconditionedClasses: classes,
		ConditionedClasses:   nil,
		CognateCorpus:        nil,
		LectIDs:              []string{"A", "B"},
	}
	out := FormatMultiLectModel(m, 2, 0)
	if !strings.Contains(out, "3 more") {
		t.Errorf("output missing '3 more' after capping at 2:\n%s", out)
	}
}

func TestMultiLectFormatFormatRendersConditionedClassesWithContexts(t *testing.T) {
	front := Context{Following: []FeatureConstraint{{Feature: "front", Value: "+"}}}
	cond := MultiLectCorrespondenceClass{
		ClassID:  0,
		Segments: map[string]string{"A": "k", "B": "s"},
		Contexts: map[string]Context{
			"A": front,
			"B": {},
		},
		Count: 8.0,
	}
	m := &MultiLectModel{
		PairwiseModels:       map[string]*LearnedModel{},
		UnconditionedClasses: nil,
		ConditionedClasses:   []MultiLectCorrespondenceClass{cond},
		CognateCorpus:        nil,
		LectIDs:              []string{"A", "B"},
	}
	out := FormatMultiLectModel(m, 0, 0)
	if !strings.Contains(out, "A:k") {
		t.Errorf("output missing 'A:k':\n%s", out)
	}
	if !strings.Contains(out, "B:s") {
		t.Errorf("output missing 'B:s':\n%s", out)
	}
	// Context appears with the pivot lect label and feature name.
	if !strings.Contains(out, "A=") {
		t.Errorf("output missing 'A=' (pivot lect context label):\n%s", out)
	}
	if !strings.Contains(out, "front") {
		t.Errorf("output missing 'front' (feature name):\n%s", out)
	}
}

func TestMultiLectFormatFormatGroupsIdenticalContextsAcrossLects(t *testing.T) {
	// COMMITMENT: when multiple lects in a conditioned class share exactly the
	// same context, the formatter collapses them into one
	// {lect1,lect2,...}=ctx entry instead of listing each lect separately.
	front := Context{Following: []FeatureConstraint{{Feature: "front", Value: "+"}}}
	cond := MultiLectCorrespondenceClass{
		ClassID:  0,
		Segments: map[string]string{"A": "ʔ", "B": "k", "C": "k", "D": "ʔ"},
		// A, C, D share the front-vowel context; B has empty.
		Contexts: map[string]Context{
			"A": front,
			"B": {},
			"C": front,
			"D": front,
		},
		Count: 6.0,
	}
	m := &MultiLectModel{
		PairwiseModels:       map[string]*LearnedModel{},
		UnconditionedClasses: nil,
		ConditionedClasses:   []MultiLectCorrespondenceClass{cond},
		CognateCorpus:        nil,
		LectIDs:              []string{"A", "B", "C", "D"},
	}
	out := FormatMultiLectModel(m, 0, 0)
	// The 3 lects sharing the context should be collapsed.
	if !strings.Contains(out, "{A,C,D}=") {
		t.Errorf("output missing '{A,C,D}=' (grouped contexts):\n%s", out)
	}
	// The empty-context lect should not appear in the display.
	if strings.Contains(out, "B=") {
		t.Errorf("output unexpectedly contains 'B=' (empty-context lect):\n%s", out)
	}
}

func TestMultiLectFormatFormatShowsPerClassConfidence(t *testing.T) {
	// COMMITMENT: format_multi_lect_model prints the per-class coverage
	// confidence alongside the count for conditioned classes.
	cond := MultiLectCorrespondenceClass{
		ClassID:  0,
		Segments: map[string]string{"A": "k", "B": "s"},
		Contexts: map[string]Context{
			"A": {Following: []FeatureConstraint{{Feature: "front", Value: "+"}}},
		},
		Count:      8.0,
		Confidence: 0.73,
	}
	m := &MultiLectModel{
		PairwiseModels:       map[string]*LearnedModel{},
		UnconditionedClasses: nil,
		ConditionedClasses:   []MultiLectCorrespondenceClass{cond},
		CognateCorpus:        nil,
		LectIDs:              []string{"A", "B"},
	}
	out := FormatMultiLectModel(m, 0, 0)
	if !strings.Contains(out, "cov=0.73") {
		t.Errorf("output missing 'cov=0.73':\n%s", out)
	}
}

// ----- DescribeMultiLectClass ---------------------------------------------

func TestMultiLectFormatDescribeClassEmptyModelShowsNoResults(t *testing.T) {
	out := DescribeMultiLectClass(EmptyMultiLectModel(), "A", "k")
	if !strings.Contains(out, "A:k") {
		t.Errorf("output missing 'A:k':\n%s", out)
	}
	if !strings.Contains(out, "(none)") {
		t.Errorf("output missing '(none)':\n%s", out)
	}
}

func TestMultiLectFormatDescribeClassShowsMatchingUnconditionedAndConditioned(t *testing.T) {
	uncond := MultiLectCorrespondenceClass{
		ClassID:  0,
		Segments: map[string]string{"A": "k", "B": "k"},
		Count:    5.0,
	}
	front := Context{Following: []FeatureConstraint{{Feature: "front", Value: "+"}}}
	cond := MultiLectCorrespondenceClass{
		ClassID:  1,
		Segments: map[string]string{"A": "k", "B": "s"},
		Contexts: map[string]Context{"A": front, "B": {}},
		Count:    3.0,
	}
	other := MultiLectCorrespondenceClass{
		ClassID:  2,
		Segments: map[string]string{"A": "p", "B": "p"},
		Count:    7.0,
	}
	m := &MultiLectModel{
		PairwiseModels:       map[string]*LearnedModel{},
		UnconditionedClasses: []MultiLectCorrespondenceClass{uncond, other},
		ConditionedClasses:   []MultiLectCorrespondenceClass{cond},
		CognateCorpus:        nil,
		LectIDs:              []string{"A", "B"},
	}
	out := DescribeMultiLectClass(m, "A", "k")
	// Matches for A:k.
	if !strings.Contains(out, "A:k") {
		t.Errorf("output missing 'A:k':\n%s", out)
	}
	if !strings.Contains(out, "B:k") {
		t.Errorf("output missing 'B:k':\n%s", out)
	}
	if !strings.Contains(out, "B:s") {
		t.Errorf("output missing 'B:s':\n%s", out)
	}
	if !strings.Contains(out, "Unconditioned entries (1)") {
		t.Errorf("output missing 'Unconditioned entries (1)':\n%s", out)
	}
	if !strings.Contains(out, "Conditioned entries (1)") {
		t.Errorf("output missing 'Conditioned entries (1)':\n%s", out)
	}
	// Other grapheme not present.
	if strings.Contains(out, "A:p") {
		t.Errorf("output unexpectedly contains 'A:p' (different grapheme):\n%s", out)
	}
}

func TestMultiLectFormatDescribeClassNoMatchesShowsZeroCounts(t *testing.T) {
	uncond := MultiLectCorrespondenceClass{
		ClassID:  0,
		Segments: map[string]string{"A": "p", "B": "p"},
		Count:    5.0,
	}
	m := &MultiLectModel{
		PairwiseModels:       map[string]*LearnedModel{},
		UnconditionedClasses: []MultiLectCorrespondenceClass{uncond},
		ConditionedClasses:   nil,
		CognateCorpus:        nil,
		LectIDs:              []string{"A", "B"},
	}
	out := DescribeMultiLectClass(m, "A", "k")
	if !strings.Contains(out, "Unconditioned entries (0)") {
		t.Errorf("output missing 'Unconditioned entries (0)':\n%s", out)
	}
	if !strings.Contains(out, "Conditioned entries (0)") {
		t.Errorf("output missing 'Conditioned entries (0)':\n%s", out)
	}
}

func TestMultiLectFormatIntegratesWithTrainedPalatalizationModel(t *testing.T) {
	// SMOKE: FormatMultiLectModel runs cleanly on a real trained MultiLectModel
	// that has both unconditioned and conditioned classes.
	type pair struct{ a, b string }
	pairs := []pair{
		{"kita", "sita"}, {"kite", "site"}, {"keta", "seta"}, {"ketu", "setu"},
		{"kata", "kata"}, {"kota", "kota"}, {"kupa", "kupa"}, {"kuma", "kuma"},
	}
	var corpus []CognateSet
	for i, p := range pairs {
		corpus = append(corpus, csOf(
			"c"+string(rune('0'+i)),
			[]string{"A", "B"},
			map[string]string{"A": p.a, "B": p.b},
		))
	}
	m := mustTrainML(t, corpus)
	out := FormatMultiLectModel(m, 0, 0)
	if !strings.Contains(out, "MultiLectModel") {
		t.Errorf("output missing 'MultiLectModel':\n%s", out)
	}
	if !strings.Contains(out, "A") {
		t.Errorf("output missing 'A':\n%s", out)
	}
	if !strings.Contains(out, "B") {
		t.Errorf("output missing 'B':\n%s", out)
	}
}
