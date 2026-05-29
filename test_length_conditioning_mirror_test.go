package regulae

// Mirror of python/tests/test_length_conditioning.py
//
// SKIPPED: test_oe_modern_english_preserves_long_short_distinction — requires
// python/experiments/oe_english/cognates.tsv which is a data file not expected
// to be present in CI; the Python test itself skips if it's missing.

import (
	"testing"
	"unicode/utf8"
)

// trnLenParseIPA parses an IPA string, treating multi-character tokens in
// `multi` as single segments. Mirrors the Python _parse helper.
func trnLenParseIPA(ipa string, multi []string) []Segment {
	var out []Segment
	i := 0
	for i < len(ipa) {
		matched := false
		for _, m := range multi {
			if len(ipa[i:]) >= len(m) && ipa[i:i+len(m)] == m {
				out = append(out, Segment{Grapheme: m})
				i += len(m)
				matched = true
				break
			}
		}
		if !matched {
			r, sz := utf8.DecodeRuneInString(ipa[i:])
			out = append(out, Segment{Grapheme: string(r)})
			i += sz
		}
	}
	return out
}

// trnLenSyntheticCorpus builds the length-conditioning synthetic corpus.
// b → β exclusively after long vowels.
func trnLenSyntheticCorpus() []CognateSet {
	multi := []string{"aː", "eː", "iː", "oː", "uː"}

	longPairs := [][2]string{
		{"aːba", "aːβa"}, {"eːba", "eːβa"}, {"iːba", "iːβa"},
		{"oːba", "oːβa"}, {"uːba", "uːβa"},
		{"aːbo", "aːβo"}, {"eːbo", "eːβo"}, {"iːbo", "iːβo"},
		{"oːbe", "oːβe"}, {"uːbo", "uːβo"},
		{"aːbe", "aːβe"}, {"eːbe", "eːβe"},
	}
	shortPairs := [][2]string{
		{"aba", "aba"}, {"eba", "eba"}, {"iba", "iba"},
		{"oba", "oba"}, {"uba", "uba"},
		{"abo", "abo"}, {"ebo", "ebo"}, {"ibo", "ibo"},
		{"obe", "obe"}, {"ubo", "ubo"},
		{"abe", "abe"}, {"ebe", "ebe"},
	}
	controls := [][2]string{
		{"ata", "ata"}, {"aːta", "aːta"}, {"eta", "eta"},
		{"eːta", "eːta"}, {"iti", "iti"}, {"iːti", "iːti"},
	}

	allPairs := append(append(longPairs, shortPairs...), controls...)
	fps := make([]FormPair, len(allPairs))
	for i, p := range allPairs {
		fps[i] = FormPair{
			Src: Form{LectID: "proto", Segments: trnLenParseIPA(p[0], multi)},
			Tgt: Form{LectID: "derived", Segments: trnLenParseIPA(p[1], multi)},
		}
	}
	return CognateSetsFromPairs(fps, [2]string{"proto", "derived"}, "")
}

// TestLengthConditioningLengthConditionedRuleIsCommitted checks that the
// discovery layer commits b→β / [long:+] _ as a conditioned correspondence.
func TestLengthConditioningLengthConditionedRuleIsCommitted(t *testing.T) {
	corpus := trnLenSyntheticCorpus()
	multi := mustTrainML(t, corpus)

	key := lectPairKey("proto", "derived")
	pair, ok := multi.PairwiseModels[key]
	if !ok {
		t.Fatal("no pairwise model for proto/derived")
	}

	var longConditioned []struct {
		cc    ConditionedCorrespondence
		count float64
	}
	for k, count := range pair.SegmentTable.Counts {
		cc, ok := pair.SegmentTable.Corr[k]
		if !ok || cc.Src != "b" || cc.Tgt != "β" {
			continue
		}
		for _, fc := range cc.Context.Preceding {
			if fc.Feature == "long" && fc.Value == "+" {
				longConditioned = append(longConditioned, struct {
					cc    ConditionedCorrespondence
					count float64
				}{cc, count})
				break
			}
		}
	}

	if len(longConditioned) == 0 {
		t.Error("expected at least one b→β entry conditioned on preceding [long:+], got none")
	}

	total := 0.0
	for _, e := range longConditioned {
		total += e.count
	}
	if total < 10 {
		t.Errorf("b→β / [long:+] total count = %v, want >= 10", total)
	}
}
