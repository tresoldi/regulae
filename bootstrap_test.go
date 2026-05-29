package regulae

import "testing"

func TestBootstrapProducesBootstrapIntervals(t *testing.T) {
	var corpus []FormPair
	for i := 0; i < 10; i++ {
		corpus = append(corpus, pairOf("A", "pa", "B", "fa"))
	}
	opts := DefaultTrainOptions()
	opts.BootstrapN = 10
	opts.BootstrapSeed = 1
	model, err := TrainPairwise(corpus, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(model.SegmentTable.Uncertainty) == 0 {
		t.Fatal("expected populated uncertainty")
	}
	for k, u := range model.SegmentTable.Uncertainty {
		if u.Method != "bootstrap" {
			t.Errorf("entry %q method = %q, want bootstrap", k, u.Method)
		}
	}
}

func TestBootstrapDeterministic(t *testing.T) {
	build := func() []FormPair {
		var c []FormPair
		for i := 0; i < 8; i++ {
			c = append(c, pairOf("A", "pa", "B", "fa"))
		}
		for i := 0; i < 4; i++ {
			c = append(c, pairOf("A", "pa", "B", "ba"))
		}
		return c
	}
	opts := DefaultTrainOptions()
	opts.BootstrapN = 20
	opts.BootstrapSeed = 7
	m1, err := TrainPairwise(build(), nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := TrainPairwise(build(), nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	for k, u1 := range m1.SegmentTable.Uncertainty {
		u2 := m2.SegmentTable.Uncertainty[k]
		if u1.Lo != u2.Lo || u1.Hi != u2.Hi {
			t.Errorf("bootstrap not deterministic for %q: [%v,%v] vs [%v,%v]", k, u1.Lo, u1.Hi, u2.Lo, u2.Hi)
		}
	}
}

func TestBootstrapMultiLectMethod(t *testing.T) {
	var corpus []CognateSet
	for i := 0; i < 8; i++ {
		corpus = append(corpus, csOf("c"+string(rune('0'+i)), []string{"A", "B"},
			map[string]string{"A": "pa", "B": "fa"}))
	}
	opts := DefaultTrainOptions()
	opts.BootstrapN = 8
	opts.BootstrapSeed = 3
	m, err := TrainModel(corpus, opts)
	if err != nil {
		t.Fatal(err)
	}
	for _, klass := range m.UnconditionedClasses {
		if klass.Uncertainty == nil || klass.Uncertainty.Method != "bootstrap" {
			t.Errorf("class uncertainty not bootstrap: %+v", klass.Uncertainty)
		}
	}
}
