package regulae

import "testing"

func TestContextSubsetEmptyMatchesAnything(t *testing.T) {
	empty := Context{}
	specific := Context{Position: "initial", Preceding: []FeatureConstraint{{"vowel", "present"}}}
	if !empty.IsSubsetOf(specific) {
		t.Error("empty context should be a subset of any context")
	}
	if specific.IsSubsetOf(empty) {
		t.Error("specific context should not be a subset of empty context")
	}
}

func TestContextSubsetPositionMismatch(t *testing.T) {
	a := Context{Position: "initial"}
	b := Context{Position: "final"}
	if a.IsSubsetOf(b) {
		t.Error("differing positions must not be subset")
	}
}

func TestContextSubsetConstraintMembership(t *testing.T) {
	vowel := FeatureConstraint{"vowel", "present"}
	voiced := FeatureConstraint{"voiced", "present"}
	a := Context{Preceding: []FeatureConstraint{vowel}}
	b := Context{Preceding: []FeatureConstraint{vowel, voiced}}
	if !a.IsSubsetOf(b) {
		t.Error("a's constraints are all present in b")
	}
	if b.IsSubsetOf(a) {
		t.Error("b has an extra constraint not in a")
	}
}

func TestContextConstraintCount(t *testing.T) {
	c := Context{
		Position:  "initial",
		Preceding: []FeatureConstraint{{"vowel", "present"}, {"voiced", "present"}},
		Following: []FeatureConstraint{{"nasal", "present"}},
	}
	if got := c.ConstraintCount(); got != 4 {
		t.Errorf("ConstraintCount = %d, want 4", got)
	}
}

func TestContextKeyDistinguishesAndMatches(t *testing.T) {
	a := Context{Position: "initial", Preceding: []FeatureConstraint{{"vowel", "present"}}}
	b := Context{Position: "initial", Preceding: []FeatureConstraint{{"vowel", "present"}}}
	c := Context{Position: "final", Preceding: []FeatureConstraint{{"vowel", "present"}}}
	if a.key() != b.key() {
		t.Error("identical contexts must have equal keys")
	}
	if a.key() == c.key() {
		t.Error("differing contexts must have distinct keys")
	}
	// Order sensitivity: matches Python frozen-dataclass dict-key semantics.
	d := Context{Preceding: []FeatureConstraint{{"vowel", "present"}, {"voiced", "present"}}}
	e := Context{Preceding: []FeatureConstraint{{"voiced", "present"}, {"vowel", "present"}}}
	if d.key() == e.key() {
		t.Error("constraint order should produce distinct keys")
	}
}

func TestSegmentComparableAsMapKey(t *testing.T) {
	m := map[Segment]int{}
	m[Segment{Grapheme: "p"}] = 1
	m[Segment{Grapheme: "p", Tone: "H"}] = 2
	if m[Segment{Grapheme: "p"}] != 1 {
		t.Error("plain segment key lookup failed")
	}
	if m[Segment{Grapheme: "p", Tone: "H"}] != 2 {
		t.Error("toned segment key lookup failed")
	}
	if len(m) != 2 {
		t.Errorf("expected 2 distinct keys, got %d", len(m))
	}
}
