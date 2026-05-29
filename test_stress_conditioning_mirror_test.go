package regulae

// Mirror of python/tests/test_stress_conditioning.py

import "testing"

// trnStressForm builds a Form where the character at stressPos (0-based) gets
// Stress="+". stressPos == -1 means no stress annotation.
func trnStressForm(lect, word string, stressPos int) Form {
	segs := make([]Segment, len(word))
	for i, c := range word {
		s := Segment{Grapheme: string(c)}
		if i == stressPos {
			s.Stress = "+"
		}
		segs[i] = s
	}
	return Form{LectID: lect, Segments: segs}
}

// ----- context propagation -----------------------------------------------

// TestStressConditioningSelfStressPopulatedFromSegmentStress checks that a
// stressed segment's link context carries self_stress=[stress:+].
func TestStressConditioningSelfStressPopulatedFromSegmentStress(t *testing.T) {
	// Use an empty model so context is populated (prior-only path with empty
	// model uses the model's feature system, which propagates stress).
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	src := trnStressForm("A", "pet", 1) // 'e' is stressed
	tgt := trnStressForm("B", "pet", 1)

	al, err := AlignForms(src, tgt, "descriptive", 0, model)
	if err != nil {
		t.Fatalf("AlignForms: %v", err)
	}
	// Link at position 1 is the middle ('e'→'e') link.
	if len(al.Links) < 3 {
		t.Fatalf("expected at least 3 links, got %d", len(al.Links))
	}
	midLink := al.Links[1]
	hasSelfStress := false
	for _, fc := range midLink.Context.SelfStress {
		if fc.Feature == "stress" && fc.Value == "+" {
			hasSelfStress = true
		}
	}
	if !hasSelfStress {
		t.Errorf("middle link context.SelfStress %v does not contain stress:+", midLink.Context.SelfStress)
	}
}

// TestStressConditioningPrecedingAndFollowingStressPopulated checks that
// following_stress and preceding_stress slots are populated for neighbours of
// the stressed segment.
func TestStressConditioningPrecedingAndFollowingStressPopulated(t *testing.T) {
	model := EmptyLearnedModel("descriptive", defaultTemperature, defaultConcentration)
	src := trnStressForm("A", "pet", 1)
	tgt := trnStressForm("B", "pet", 1)

	al, err := AlignForms(src, tgt, "descriptive", 0, model)
	if err != nil {
		t.Fatalf("AlignForms: %v", err)
	}
	if len(al.Links) < 3 {
		t.Fatalf("expected at least 3 links, got %d", len(al.Links))
	}

	// First link ('p'→'p'): following segment is stressed → following_stress should have stress:+
	firstLink := al.Links[0]
	hasFollowStress := false
	for _, fc := range firstLink.Context.FollowingStress {
		if fc.Feature == "stress" && fc.Value == "+" {
			hasFollowStress = true
		}
	}
	if !hasFollowStress {
		t.Errorf("first link FollowingStress %v does not contain stress:+", firstLink.Context.FollowingStress)
	}

	// Last link ('t'→'t'): preceding segment is stressed → preceding_stress should have stress:+
	lastLink := al.Links[2]
	hasPrecStress := false
	for _, fc := range lastLink.Context.PrecedingStress {
		if fc.Feature == "stress" && fc.Value == "+" {
			hasPrecStress = true
		}
	}
	if !hasPrecStress {
		t.Errorf("last link PrecedingStress %v does not contain stress:+", lastLink.Context.PrecedingStress)
	}
}

// TestStressConditioningContextSubsetHonorsStressSlots checks that Context.IsSubsetOf
// correctly treats stress slots.
func TestStressConditioningContextSubsetHonorsStressSlots(t *testing.T) {
	base := Context{}
	stressed := Context{SelfStress: []FeatureConstraint{{Feature: "stress", Value: "+"}}}
	if !base.IsSubsetOf(stressed) {
		t.Error("empty context should be a subset of stressed context")
	}
	if stressed.IsSubsetOf(base) {
		t.Error("stressed context should NOT be a subset of empty context")
	}
}

// ----- discovery on synthetic fixture ------------------------------------

// TestStressConditioningStressSyntheticCommitsVowelLoweringRules checks that
// on a stress-conditioned corpus, discovery commits e→ɛ/self_stress=[stress:+]
// and o→ɔ/self_stress=[stress:+].
func TestStressConditioningStressSyntheticCommitsVowelLoweringRules(t *testing.T) {
	vowels := map[rune]bool{'a': true, 'e': true, 'i': true, 'o': true, 'u': true, 'ɛ': true, 'ɔ': true}

	// parseStressedForm mirrors the Python `form` helper in the test.
	// ˈ marks the following vowel as stressed; - is skipped.
	parseStressedForm := func(lect, ipa string) Form {
		var segs []Segment
		pending := ""
		runes := []rune(ipa)
		for _, ch := range runes {
			if ch == 'ˈ' {
				pending = "+"
				continue
			}
			if ch == '-' {
				continue
			}
			s := Segment{Grapheme: string(ch)}
			if vowels[ch] && pending != "" {
				s.Stress = pending
				pending = ""
			}
			segs = append(segs, s)
		}
		return Form{LectID: lect, Segments: segs}
	}

	rawPairs := [][2]string{
		{"ˈpe-ta", "ˈpɛ-ta"}, {"ˈpe-ro", "ˈpɛ-ro"}, {"ˈpe-li", "ˈpɛ-li"},
		{"ˈte-na", "ˈtɛ-na"}, {"ˈte-ru", "ˈtɛ-ru"}, {"ˈte-mo", "ˈtɛ-mo"},
		{"ˈke-pa", "ˈkɛ-pa"}, {"ˈke-no", "ˈkɛ-no"}, {"ˈme-ru", "ˈmɛ-ru"},
		{"ˈme-na", "ˈmɛ-na"}, {"ˈde-po", "ˈdɛ-po"}, {"ˈde-la", "ˈdɛ-la"},
		{"ˈpo-ta", "ˈpɔ-ta"}, {"ˈpo-mi", "ˈpɔ-mi"}, {"ˈto-la", "ˈtɔ-la"},
		{"ˈto-pe", "ˈtɔ-pe"}, {"ˈko-na", "ˈkɔ-na"}, {"ˈko-ri", "ˈkɔ-ri"},
		{"ˈmo-la", "ˈmɔ-la"}, {"ˈmo-ni", "ˈmɔ-ni"}, {"ˈdo-ra", "ˈdɔ-ra"},
		{"ˈdo-mi", "ˈdɔ-mi"},
		// controls: unstressed, no change
		{"pe-ˈta", "pe-ˈta"}, {"pe-ˈro", "pe-ˈro"},
		{"te-ˈna", "te-ˈna"}, {"te-ˈru", "te-ˈru"},
		{"ke-ˈpa", "ke-ˈpa"}, {"me-ˈna", "me-ˈna"}, {"de-ˈpo", "de-ˈpo"},
		{"po-ˈta", "po-ˈta"}, {"po-ˈmi", "po-ˈmi"},
		{"to-ˈla", "to-ˈla"}, {"to-ˈpe", "to-ˈpe"},
		{"ko-ˈna", "ko-ˈna"}, {"mo-ˈla", "mo-ˈla"}, {"do-ˈra", "do-ˈra"},
	}

	fps := make([]FormPair, len(rawPairs))
	for i, p := range rawPairs {
		fps[i] = FormPair{
			Src: parseStressedForm("proto", p[0]),
			Tgt: parseStressedForm("derived", p[1]),
		}
	}
	corpus := CognateSetsFromPairs(fps, [2]string{"proto", "derived"}, "")
	multi := mustTrainML(t, corpus)

	key := lectPairKey("proto", "derived")
	pair, ok := multi.PairwiseModels[key]
	if !ok {
		t.Fatal("no pairwise model for proto/derived")
	}

	// Collect e→ɛ conditioned on self_stress=[stress:+]
	var eLowered, oLowered []float64
	for k, count := range pair.SegmentTable.Counts {
		cc, ok := pair.SegmentTable.Corr[k]
		if !ok {
			continue
		}
		hasSelfStress := false
		for _, fc := range cc.Context.SelfStress {
			if fc.Feature == "stress" && fc.Value == "+" {
				hasSelfStress = true
			}
		}
		if !hasSelfStress {
			continue
		}
		if cc.Src == "e" && cc.Tgt == "ɛ" {
			eLowered = append(eLowered, count)
		}
		if cc.Src == "o" && cc.Tgt == "ɔ" {
			oLowered = append(oLowered, count)
		}
	}

	if len(eLowered) == 0 {
		t.Error("expected e→ɛ conditioned on self_stress=[stress:+]")
	}
	if len(oLowered) == 0 {
		t.Error("expected o→ɔ conditioned on self_stress=[stress:+]")
	}

	eTotal := 0.0
	for _, c := range eLowered {
		eTotal += c
	}
	if eTotal < 10 {
		t.Errorf("e→ɛ / stress total = %v, want >= 10", eTotal)
	}

	oTotal := 0.0
	for _, c := range oLowered {
		oTotal += c
	}
	if oTotal < 8 {
		t.Errorf("o→ɔ / stress total = %v, want >= 8", oTotal)
	}
}

// ----- unchanged behavior without stress ---------------------------------

// TestStressConditioningCorpusWithoutStressAnnotationIsUnaffected checks that
// a corpus without stress annotations produces no stress-conditioned entries.
func TestStressConditioningCorpusWithoutStressAnnotationIsUnaffected(t *testing.T) {
	corpus := make([]CognateSet, 5)
	for i := range corpus {
		corpus[i] = CognateSet{
			CognateID: "c" + string(rune('0'+i)),
			Forms: map[string]Form{
				"A": {LectID: "A", Segments: []Segment{{Grapheme: "p"}, {Grapheme: "a"}}},
				"B": {LectID: "B", Segments: []Segment{{Grapheme: "f"}, {Grapheme: "a"}}},
			},
			FormsOrder: []string{"A", "B"},
			Confidence: 1.0,
		}
	}
	multi := mustTrainML(t, corpus)

	key := lectPairKey("A", "B")
	pair, ok := multi.PairwiseModels[key]
	if !ok {
		t.Fatal("no pairwise model for A/B")
	}

	for k, cc := range pair.SegmentTable.Corr {
		ctx := pair.SegmentTable.Corr[k].Context
		_ = cc
		if len(ctx.SelfStress) > 0 {
			t.Errorf("unexpected self_stress on entry %q: %v", k, ctx.SelfStress)
		}
		if len(ctx.PrecedingStress) > 0 {
			t.Errorf("unexpected preceding_stress on entry %q: %v", k, ctx.PrecedingStress)
		}
		if len(ctx.FollowingStress) > 0 {
			t.Errorf("unexpected following_stress on entry %q: %v", k, ctx.FollowingStress)
		}
	}
}
