package regulae

// Mirror of python/tests/test_multi_lect_model.py
//
// Skipped (already covered):
//   TestMultiLectTrainBasic              — reconciliation_test.go
//   TestMultiLectClassesForSegment       — reconciliation_test.go
//   TestMultiLectDeterministic           — reconciliation_test.go
//   TestMultiLectRejectsSingleFormInMultiLectCorpus — reconciliation_test.go
//   TestFormatMultiLectModel             — format_test.go
//
// Python tests about frozen dataclasses have no Go equivalent (Go structs are
// mutable by convention); those are noted inline and skipped.

import "testing"

// mltForm is a local alias used only by this file to avoid name collisions.
func mltFormIPA(lect, word string) Form {
	return formIPA(lect, word)
}

// ----- CognateSet ---------------------------------------------------------

func TestMultiLectModelCognateSetMinimalConstruction(t *testing.T) {
	// COMMITMENT: a CognateSet can be built from just an id and forms.
	cs := CognateSet{
		CognateID: "hand.001",
		Forms: map[string]Form{
			"latin":   mltFormIPA("latin", "manus"),
			"spanish": mltFormIPA("spanish", "mano"),
		},
		Confidence: 1.0,
	}
	if cs.CognateID != "hand.001" {
		t.Errorf("CognateID = %q, want %q", cs.CognateID, "hand.001")
	}
	if _, ok := cs.Forms["latin"]; !ok {
		t.Error("expected 'latin' in Forms")
	}
	if _, ok := cs.Forms["spanish"]; !ok {
		t.Error("expected 'spanish' in Forms")
	}
	if cs.Alignments != nil {
		t.Error("expected nil Alignments")
	}
	if cs.MorphemeBoundaries != nil {
		t.Error("expected nil MorphemeBoundaries")
	}
	if cs.Confidence != 1.0 {
		t.Errorf("Confidence = %v, want 1.0", cs.Confidence)
	}
}

func TestMultiLectModelCognateSetSingleLectIsValid(t *testing.T) {
	// COMMITMENT: N=1 is a legal degenerate case.
	cs := CognateSet{
		CognateID:  "x",
		Forms:      map[string]Form{"latin": mltFormIPA("latin", "pater")},
		Confidence: 1.0,
	}
	if len(cs.Forms) != 1 {
		t.Errorf("len(Forms) = %d, want 1", len(cs.Forms))
	}
}

func TestMultiLectModelCognateSetAbsentLectIsMissingFromForms(t *testing.T) {
	// COMMITMENT: Q6 — no sentinel value for missing lects; absence is
	// represented by the lect not being in Forms.
	cs := CognateSet{
		CognateID: "x",
		Forms: map[string]Form{
			"a": mltFormIPA("a", "pa"),
			"c": mltFormIPA("c", "fa"),
		},
		Confidence: 1.0,
	}
	if _, ok := cs.Forms["b"]; ok {
		t.Error("unexpected lect 'b' in Forms")
	}
	if _, ok := cs.Forms["a"]; !ok {
		t.Error("expected 'a' in Forms")
	}
	if _, ok := cs.Forms["c"]; !ok {
		t.Error("expected 'c' in Forms")
	}
	if len(cs.Forms) != 2 {
		t.Errorf("len(Forms) = %d, want 2", len(cs.Forms))
	}
}

func TestMultiLectModelCognateSetHoldsOptionalAlignments(t *testing.T) {
	// COMMITMENT: alignments is an optional warm-start hint.
	latinSegs := mltFormIPA("latin", "manus").Segments
	spanishSegs := mltFormIPA("spanish", "mano").Segments

	alignments := map[string][]*Segment{}
	for i := range latinSegs {
		s := latinSegs[i]
		alignments["latin"] = append(alignments["latin"], &s)
	}
	for i := range spanishSegs {
		s := spanishSegs[i]
		alignments["spanish"] = append(alignments["spanish"], &s)
	}
	// Make last spanish segment nil (Python: None at tail).
	if len(alignments["spanish"]) > 0 {
		alignments["spanish"][len(alignments["spanish"])-1] = nil
	}

	cs := CognateSet{
		CognateID: "hand",
		Forms: map[string]Form{
			"latin":   mltFormIPA("latin", "manus"),
			"spanish": mltFormIPA("spanish", "mano"),
		},
		Alignments: alignments,
		Confidence: 1.0,
	}
	if cs.Alignments == nil {
		t.Fatal("expected non-nil Alignments")
	}
	last := cs.Alignments["spanish"][len(cs.Alignments["spanish"])-1]
	if last != nil {
		t.Errorf("expected last spanish alignment entry to be nil, got %v", last)
	}
}

func TestMultiLectModelCognateSetConfidenceIsSettable(t *testing.T) {
	cs := CognateSet{
		CognateID:  "x",
		Forms:      map[string]Form{"a": mltFormIPA("a", "pa")},
		Confidence: 0.6,
	}
	if cs.Confidence != 0.6 {
		t.Errorf("Confidence = %v, want 0.6", cs.Confidence)
	}
}

func TestMultiLectModelCognateSetMorphemeBoundariesReservedField(t *testing.T) {
	// COMMITMENT: morpheme_boundaries is captured but not used in training.
	cs := CognateSet{
		CognateID:          "x",
		Forms:              map[string]Form{"a": mltFormIPA("a", "patre")},
		MorphemeBoundaries: map[string][]int{"a": {4}},
		Confidence:         1.0,
	}
	bounds := cs.MorphemeBoundaries["a"]
	if len(bounds) != 1 || bounds[0] != 4 {
		t.Errorf("MorphemeBoundaries[a] = %v, want [4]", bounds)
	}
}

// test_cognate_set_is_frozen — skipped: Go structs are not frozen.

// ----- MultiLectCorrespondenceClass ---------------------------------------

func TestMultiLectModelMultilectClassUnconditionedDefaults(t *testing.T) {
	// COMMITMENT: Contexts=nil means the class is unconditioned.
	klass := MultiLectCorrespondenceClass{
		ClassID:  0,
		Segments: map[string]string{"latin": "p", "spanish": "p", "french": "p"},
	}
	if klass.Contexts != nil {
		t.Error("expected nil Contexts for unconditioned class")
	}
	if klass.Count != 0.0 {
		t.Errorf("Count = %v, want 0.0", klass.Count)
	}
	if len(klass.SupportingCognates) != 0 {
		t.Errorf("expected empty SupportingCognates, got %v", klass.SupportingCognates)
	}
}

func TestMultiLectModelMultilectClassWithContextsIsConditioned(t *testing.T) {
	// COMMITMENT: non-nil Contexts means the class is conditioned.
	front := Context{Following: []FeatureConstraint{{Feature: "front", Value: "+"}}}
	klass := MultiLectCorrespondenceClass{
		ClassID:  7,
		Segments: map[string]string{"latin": "k", "spanish": "θ", "french": "s"},
		Contexts: map[string]Context{
			"latin":   front,
			"spanish": {},
			"french":  {},
		},
		Count:              12.0,
		SupportingCognates: []string{"caelum", "cera"},
	}
	if klass.Contexts == nil {
		t.Fatal("expected non-nil Contexts")
	}
	got := klass.Contexts["latin"]
	if len(got.Following) == 0 || got.Following[0] != (FeatureConstraint{Feature: "front", Value: "+"}) {
		t.Errorf("latin context = %v, want front context", got)
	}
	if klass.Count != 12.0 {
		t.Errorf("Count = %v, want 12.0", klass.Count)
	}
	found := false
	for _, c := range klass.SupportingCognates {
		if c == "caelum" {
			found = true
		}
	}
	if !found {
		t.Error("expected 'caelum' in SupportingCognates")
	}
}

// test_multilect_class_is_frozen — skipped: Go structs are not frozen.

// ----- MultiLectModel -----------------------------------------------------

func TestMultiLectModelEmptyFactory(t *testing.T) {
	// COMMITMENT: EmptyMultiLectModel yields a model with no pairs and no classes.
	m := EmptyMultiLectModel()
	if len(m.PairwiseModels) != 0 {
		t.Errorf("expected empty PairwiseModels, got %d", len(m.PairwiseModels))
	}
	if len(m.UnconditionedClasses) != 0 {
		t.Errorf("expected empty UnconditionedClasses, got %d", len(m.UnconditionedClasses))
	}
	if len(m.ConditionedClasses) != 0 {
		t.Errorf("expected empty ConditionedClasses, got %d", len(m.ConditionedClasses))
	}
	if len(m.CognateCorpus) != 0 {
		t.Errorf("expected empty CognateCorpus, got %d", len(m.CognateCorpus))
	}
	if len(m.LectIDs) != 0 {
		t.Errorf("expected empty LectIDs, got %d", len(m.LectIDs))
	}
}

func TestMultiLectModelPairsKeyedByUnorderedPair(t *testing.T) {
	// COMMITMENT: pairs are keyed by an unordered pair (lectPairKey) so
	// PairwiseModel("latin","spanish") == PairwiseModel("spanish","latin").
	pair := EmptyLearnedModel("descriptive", 1.0, 5.0)
	key := lectPairKey("latin", "spanish")
	m := &MultiLectModel{
		PairwiseModels: map[string]*LearnedModel{key: pair},
		LectIDs:        []string{"latin", "spanish"},
	}
	_, ok1 := m.PairwiseModel("latin", "spanish")
	_, ok2 := m.PairwiseModel("spanish", "latin")
	if !ok1 {
		t.Error("PairwiseModel(latin, spanish) not found")
	}
	if !ok2 {
		t.Error("PairwiseModel(spanish, latin) not found (order-independence failed)")
	}
}

func TestMultiLectModelHoldsClassesAndCorpus(t *testing.T) {
	cs := CognateSet{
		CognateID: "c1",
		Forms: map[string]Form{
			"a": mltFormIPA("a", "pa"),
			"b": mltFormIPA("b", "fa"),
		},
		Confidence: 1.0,
	}
	uncond := MultiLectCorrespondenceClass{
		ClassID:  0,
		Segments: map[string]string{"a": "p", "b": "f"},
		Count:    1.0,
	}
	m := &MultiLectModel{
		PairwiseModels:       map[string]*LearnedModel{lectPairKey("a", "b"): EmptyLearnedModel("descriptive", 1.0, 5.0)},
		UnconditionedClasses: []MultiLectCorrespondenceClass{uncond},
		ConditionedClasses:   nil,
		CognateCorpus:        []CognateSet{cs},
		LectIDs:              []string{"a", "b"},
	}
	if m.UnconditionedClasses[0].Segments["a"] != "p" {
		t.Errorf("UnconditionedClasses[0].Segments[a] = %q, want %q", m.UnconditionedClasses[0].Segments["a"], "p")
	}
	if m.CognateCorpus[0].CognateID != "c1" {
		t.Errorf("CognateCorpus[0].CognateID = %q, want %q", m.CognateCorpus[0].CognateID, "c1")
	}
	if len(m.LectIDs) != 2 || m.LectIDs[0] != "a" || m.LectIDs[1] != "b" {
		t.Errorf("LectIDs = %v, want [a b]", m.LectIDs)
	}
}

// test_multilect_model_is_frozen — skipped: Go structs are not frozen.
