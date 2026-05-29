package regulae

import (
	"reflect"
	"sort"
	"testing"
)

func formFromWord(word string) Form {
	runes := []rune(word)
	segs := make([]Segment, len(runes))
	for i, r := range runes {
		segs[i] = Segment{Grapheme: string(r)}
	}
	return Form{LectID: "x", Segments: segs}
}

func TestSonorityScoring(t *testing.T) {
	clearSyllabificationCaches()
	cases := []struct {
		grapheme string
		want     int
	}{
		{"a", sonorityVowel},
		{"i", sonorityVowel},
		{"p", sonorityStop},
		{"k", sonorityStop},
		{"m", sonorityNasal},
		{"n", sonorityNasal},
		{"l", sonorityLiquid}, // approximant + lateral
		{"r", sonorityLiquid}, // trill
		{"j", sonorityGlide},  // approximant, non-lateral
		{"w", sonorityGlide},
		{"", -1}, // empty grapheme = skip marker
	}
	for _, c := range cases {
		if got := sonority(Segment{Grapheme: c.grapheme}, "descriptive"); got != c.want {
			t.Errorf("sonority(%q) = %d, want %d", c.grapheme, got, c.want)
		}
	}
}

func TestSonorityUnknownGraphemeNeutral(t *testing.T) {
	score := sonorityFromFeatures(nil)
	if score < 0 || score > sonorityVowel {
		t.Errorf("sonorityFromFeatures(nil) = %d, want in [0, %d]", score, sonorityVowel)
	}
}

func TestSyllableBreaksHandChecked(t *testing.T) {
	clearSyllabificationCaches()
	cases := []struct {
		word string
		want []int
	}{
		{"a", nil},
		{"pa", nil},
		{"pata", []int{2}},
		{"kat", nil},
		{"kaspata", []int{3, 5}},
		{"strata", []int{4}},
		{"iman", []int{1}},
		{"aia", []int{1, 2}},
		{"spara", []int{3}},
		{"patask", []int{2}},
		{"pakla", []int{2}},
		{"bdl", nil},
		{"prm", nil},
	}
	for _, c := range cases {
		got := ComputeSyllableBreaks(formFromWord(c.word), "descriptive")
		if !equalIntSlices(got, c.want) {
			t.Errorf("ComputeSyllableBreaks(%q) = %v, want %v", c.word, got, c.want)
		}
	}
}

func TestSyllableBreaksUserOverride(t *testing.T) {
	form := Form{
		LectID:         "x",
		Segments:       formFromWord("strata").Segments,
		SyllableBreaks: []int{2, 4},
	}
	got := ComputeSyllableBreaks(form, "descriptive")
	if !equalIntSlices(got, []int{2, 4}) {
		t.Errorf("override not respected: got %v, want [2 4]", got)
	}
}

func TestSyllableBreaksDeterministic(t *testing.T) {
	form := formFromWord("kaspata")
	r1 := ComputeSyllableBreaks(form, "descriptive")
	r2 := ComputeSyllableBreaks(form, "descriptive")
	r3 := ComputeSyllableBreaks(form, "descriptive")
	if !equalIntSlices(r1, r2) || !equalIntSlices(r2, r3) {
		t.Errorf("non-deterministic: %v %v %v", r1, r2, r3)
	}
}

func TestSyllableBreaksCacheClears(t *testing.T) {
	_ = ComputeSyllableBreaks(formFromWord("pata"), "descriptive")
	clearSyllabificationCaches()
	if got := ComputeSyllableBreaks(formFromWord("pata"), "descriptive"); !equalIntSlices(got, []int{2}) {
		t.Errorf("after cache clear: got %v, want [2]", got)
	}
}

func TestSyllableBreaksStrictlyIncreasing(t *testing.T) {
	for _, word := range []string{"pata", "kaspata", "strata", "iman", "aia", "spara", "patask", "pakla"} {
		breaks := ComputeSyllableBreaks(formFromWord(word), "descriptive")
		n := len([]rune(word))
		seen := map[int]bool{}
		for _, b := range breaks {
			if b <= 0 || b >= n {
				t.Errorf("%q: break %d out of range (0, %d)", word, b, n)
			}
			if seen[b] {
				t.Errorf("%q: duplicate break %d", word, b)
			}
			seen[b] = true
		}
		if !sort.IntsAreSorted(breaks) {
			t.Errorf("%q: breaks %v not sorted", word, breaks)
		}
	}
}

func TestSyllableBreaksEmptyForm(t *testing.T) {
	empty := Form{LectID: "x"}
	if got := ComputeSyllableBreaks(empty, "descriptive"); len(got) != 0 {
		t.Errorf("empty form: got %v, want empty", got)
	}
}

func TestSyllableBreaksToneOnlySkipped(t *testing.T) {
	segs := []Segment{{Grapheme: "p"}, {Grapheme: "a"}, {Grapheme: ""}, {Grapheme: "t"}, {Grapheme: "a"}}
	form := Form{LectID: "x", Segments: segs}
	if got := ComputeSyllableBreaks(form, "descriptive"); !equalIntSlices(got, []int{3}) {
		t.Errorf("tone-only skip: got %v, want [3]", got)
	}
}

func equalIntSlices(a, b []int) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}
