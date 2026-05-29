package regulae

import "testing"

func csOf(id string, order []string, forms map[string]string) CognateSet {
	f := map[string]Form{}
	for lect, ipa := range forms {
		f[lect] = formIPA(lect, ipa)
	}
	return CognateSet{CognateID: id, Forms: f, FormsOrder: order, Confidence: 1.0}
}

func mustTrainML(t *testing.T, corpus []CognateSet) *MultiLectModel {
	t.Helper()
	m, err := TrainModel(corpus, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainModel error: %v", err)
	}
	return m
}

func TestMultiLectTrainBasic(t *testing.T) {
	var corpus []CognateSet
	for i := 0; i < 5; i++ {
		corpus = append(corpus, csOf(
			"c"+string(rune('0'+i)),
			[]string{"A", "B", "C"},
			map[string]string{"A": "pa", "B": "fa", "C": "fa"},
		))
	}
	m := mustTrainML(t, corpus)

	if len(m.LectIDs) != 3 {
		t.Fatalf("expected 3 lect ids, got %v", m.LectIDs)
	}
	// All three pairs should have a trained model.
	for _, pair := range [][2]string{{"A", "B"}, {"A", "C"}, {"B", "C"}} {
		if _, ok := m.PairwiseModel(pair[0], pair[1]); !ok {
			t.Errorf("missing pairwise model for %v", pair)
		}
	}
	if len(m.UnconditionedClasses) == 0 {
		t.Fatal("expected at least one unconditioned class")
	}
	// There should be a class binding A->p, B->f, C->f.
	found := false
	for _, klass := range m.UnconditionedClasses {
		if klass.Segments["A"] == "p" && klass.Segments["B"] == "f" && klass.Segments["C"] == "f" {
			found = true
			if klass.Count < 5.0 {
				t.Errorf("p/f/f class count = %v, want >= 5", klass.Count)
			}
		}
	}
	if !found {
		t.Errorf("expected a class binding A:p, B:f, C:f; got %d classes", len(m.UnconditionedClasses))
	}
}

func TestMultiLectClassesForSegment(t *testing.T) {
	var corpus []CognateSet
	for i := 0; i < 5; i++ {
		corpus = append(corpus, csOf(
			"c"+string(rune('0'+i)),
			[]string{"A", "B"},
			map[string]string{"A": "pa", "B": "fa"},
		))
	}
	m := mustTrainML(t, corpus)
	classes := m.ClassesForSegment("A", "p", true, true)
	if len(classes) == 0 {
		t.Error("expected at least one class binding A:p")
	}
	for _, k := range classes {
		if k.Segments["A"] != "p" {
			t.Errorf("ClassesForSegment returned class not binding A:p: %v", k.Segments)
		}
	}
}

func TestMultiLectDeterministic(t *testing.T) {
	build := func() []CognateSet {
		var corpus []CognateSet
		for i := 0; i < 5; i++ {
			corpus = append(corpus, csOf(
				"c"+string(rune('0'+i)),
				[]string{"A", "B", "C"},
				map[string]string{"A": "pat", "B": "fad", "C": "fat"},
			))
		}
		return corpus
	}
	m1 := mustTrainML(t, build())
	m2 := mustTrainML(t, build())
	if len(m1.UnconditionedClasses) != len(m2.UnconditionedClasses) {
		t.Fatalf("class count differs: %d vs %d", len(m1.UnconditionedClasses), len(m2.UnconditionedClasses))
	}
	for i := range m1.UnconditionedClasses {
		a, b := m1.UnconditionedClasses[i], m2.UnconditionedClasses[i]
		if a.Count != b.Count || a.ClassID != b.ClassID {
			t.Errorf("class %d differs across runs", i)
		}
		for lect, g := range a.Segments {
			if b.Segments[lect] != g {
				t.Errorf("class %d segment %s differs: %s vs %s", i, lect, g, b.Segments[lect])
			}
		}
	}
}

func TestMultiLectRejectsSingleFormInMultiLectCorpus(t *testing.T) {
	corpus := []CognateSet{
		csOf("c0", []string{"A", "B"}, map[string]string{"A": "pa", "B": "fa"}),
		csOf("c1", []string{"A"}, map[string]string{"A": "pa"}),
	}
	_, err := TrainModel(corpus, DefaultTrainOptions())
	if err == nil {
		t.Error("expected error for a single-form cognate set in a multi-lect corpus")
	}
}
