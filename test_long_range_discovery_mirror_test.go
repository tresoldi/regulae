package regulae

// Mirror of python/tests/test_long_range_discovery.py

import "testing"

// trnLRForm builds a Form from an IPA string.
func trnLRForm(lect, ipa string) Form {
	return formIPA(lect, ipa)
}

// trnLRPairs converts string-pair slices into []FormPair with lects "A" and "B".
func trnLRPairs(raw [][2]string) []FormPair {
	out := make([]FormPair, len(raw))
	for i, p := range raw {
		out[i] = FormPair{
			Src: trnLRForm("A", p[0]),
			Tgt: trnLRForm("B", p[1]),
		}
	}
	return out
}

// trnLRTrainPair converts a []FormPair to a MultiLectModel pairwise on ("A","B").
func trnLRTrainPair(t *testing.T, corpus []FormPair) *LearnedModel {
	t.Helper()
	sets := CognateSetsFromPairs(corpus, [2]string{"A", "B"}, "")
	multi := mustTrainML(t, sets)
	lm, ok := multi.PairwiseModel("A", "B")
	if !ok {
		t.Fatal("no pairwise model for A/B")
	}
	return lm
}

// lrEntriesForSource returns all long-range conditioned entries whose Src == src.
func lrEntriesForSource(model *LearnedModel, src string) []struct {
	cc    ConditionedCorrespondence
	count float64
} {
	var out []struct {
		cc    ConditionedCorrespondence
		count float64
	}
	for k, count := range model.SegmentTable.Counts {
		cc, ok := model.SegmentTable.Corr[k]
		if !ok || cc.Src != src {
			continue
		}
		ctx := cc.Context
		lr := len(ctx.PrecedingAtDistance) +
			len(ctx.FollowingAtDistance) +
			len(ctx.SomewherePreceding) +
			len(ctx.SomewhereFollowing) +
			len(ctx.SameSyllable) +
			len(ctx.NextSyllable) +
			len(ctx.PreviousSyllable)
		if lr > 0 {
			out = append(out, struct {
				cc    ConditionedCorrespondence
				count float64
			}{cc, count})
		}
	}
	return out
}

func trnLRUmlautCorpus() []FormPair {
	umlaut := [][2]string{
		{"pati", "pæti"}, {"bani", "bæni"}, {"kati", "kæti"},
		{"dani", "dæni"}, {"gati", "gæti"},
		{"pabe", "pæbe"}, {"bade", "bæde"}, {"kame", "kæme"},
		{"dape", "dæpe"}, {"gabe", "gæbe"},
	}
	control := [][2]string{
		{"pato", "pato"}, {"bado", "bado"}, {"kako", "kako"},
		{"dago", "dago"}, {"gamo", "gamo"},
		{"papu", "papu"}, {"babu", "babu"}, {"kamu", "kamu"},
		{"datu", "datu"}, {"gapu", "gapu"},
	}
	nonhit := [][2]string{
		{"piti", "piti"}, {"biki", "biki"}, {"kiti", "kiti"},
		{"gidi", "gidi"}, {"pito", "pito"},
	}
	all := append(append(umlaut, control...), nonhit...)
	return trnLRPairs(all)
}

func trnLRHarmonyCorpus() []FormPair {
	hit := [][2]string{
		{"puka", "puko"}, {"buka", "buko"}, {"tuma", "tumo"},
		{"duna", "duno"}, {"kuba", "kubo"}, {"gupa", "gupo"},
		{"poka", "poko"}, {"boma", "bomo"}, {"toka", "toko"},
		{"doba", "dobo"}, {"nupa", "nupo"}, {"muka", "muko"},
		{"gopa", "gopo"}, {"noma", "nomo"}, {"kopa", "kopo"},
	}
	nonhit := [][2]string{
		{"pika", "pika"}, {"bika", "bika"}, {"tima", "tima"},
		{"dina", "dina"}, {"kiba", "kiba"}, {"gipa", "gipa"},
		{"peka", "peka"}, {"beka", "beka"}, {"tema", "tema"},
		{"dena", "dena"},
	}
	return trnLRPairs(append(hit, nonhit...))
}

func trnLRNoiseCorpus() []FormPair {
	words := []string{
		"pata", "kata", "tapa", "paka", "maku", "kuma", "puki", "piki",
		"toke", "keto", "buba", "dudu", "gaga", "kiki", "papa", "tata",
		"nana", "mama", "gogo", "pipi", "tutu", "bubu", "dada", "nene",
		"mimi", "kuki", "guga", "toto", "dodo", "baba", "giga", "tiki",
		"mako", "kumi", "pitu", "bika", "doke", "noma", "kupu",
	}
	// "takö" contains ö which may be unknown; replace with "taku" to keep
	// the corpus all-ASCII-IPA.
	words = append(words, "taku")
	pairs := make([]FormPair, len(words))
	for i, w := range words {
		pairs[i] = FormPair{
			Src: trnLRForm("A", w),
			Tgt: trnLRForm("B", w),
		}
	}
	return pairs
}

// TestLongRangeRecoveryUmlautRule checks that umlaut discovery commits an
// a→æ rule with next_syllable=[front:+].
func TestLongRangeRecoveryUmlautRule(t *testing.T) {
	model := trnLRTrainPair(t, trnLRUmlautCorpus())
	entries := lrEntriesForSource(model, "a")
	// Look for an a→æ entry with next_syllable containing front:+.
	var hits []struct {
		cc    ConditionedCorrespondence
		count float64
	}
	for _, e := range entries {
		if e.cc.Tgt != "æ" {
			continue
		}
		for _, fc := range e.cc.Context.NextSyllable {
			if fc.Feature == "front" && fc.Value == "+" {
				hits = append(hits, e)
				break
			}
		}
	}
	if len(hits) == 0 {
		t.Fatalf("no umlaut rule committed (a→æ / next_syllable=[front:+]); got %v", entries)
	}
	// The rule should capture most of the 10 umlaut firings.
	if hits[0].count < 8 {
		t.Errorf("umlaut rule count = %v, want >= 8", hits[0].count)
	}
}

// TestLongRangeRecoveryHarmonyRule checks that harmony discovery commits a
// previous_syllable-conditioned entry for source "a".
func TestLongRangeRecoveryHarmonyRule(t *testing.T) {
	model := trnLRTrainPair(t, trnLRHarmonyCorpus())
	entries := lrEntriesForSource(model, "a")
	if len(entries) == 0 {
		t.Fatal("no long-range rule on source 'a'")
	}
	// Either a→o / prev_syl[back:+] or a→a / prev_syl[front:+] is acceptable.
	var hits []struct {
		cc    ConditionedCorrespondence
		count float64
	}
	for _, e := range entries {
		ctx := e.cc.Context
		if len(ctx.PreviousSyllable) == 0 {
			continue
		}
		for _, fc := range ctx.PreviousSyllable {
			if (fc.Feature == "front" || fc.Feature == "back") && fc.Value == "+" {
				hits = append(hits, e)
				break
			}
		}
	}
	if len(hits) == 0 {
		t.Fatalf("no prev_syl harmony rule committed; got %v", entries)
	}
	// At least one commit should carry a substantive count.
	maxCount := 0.0
	for _, h := range hits {
		if h.count > maxCount {
			maxCount = h.count
		}
	}
	if maxCount < 10 {
		t.Errorf("max harmony rule count = %v, want >= 10", maxCount)
	}
}

// TestLongRangeCommitsNothingOnNoiseCorpus checks that no long-range entries
// are committed on an identity noise corpus.
func TestLongRangeCommitsNothingOnNoiseCorpus(t *testing.T) {
	model := trnLRTrainPair(t, trnLRNoiseCorpus())
	for k, count := range model.SegmentTable.Counts {
		cc, ok := model.SegmentTable.Corr[k]
		if !ok {
			continue
		}
		ctx := cc.Context
		lr := len(ctx.PrecedingAtDistance) +
			len(ctx.FollowingAtDistance) +
			len(ctx.SomewherePreceding) +
			len(ctx.SomewhereFollowing) +
			len(ctx.SameSyllable) +
			len(ctx.NextSyllable) +
			len(ctx.PreviousSyllable)
		if lr > 0 {
			t.Errorf("unexpected long-range commit on noise: cc=%+v count=%v", cc, count)
		}
	}
}

// TestLongRangeIsDeterministic checks that repeated training on the same corpus
// produces identical long-range entries.
func TestLongRangeIsDeterministic(t *testing.T) {
	corpus := trnLRUmlautCorpus()
	m1 := trnLRTrainPair(t, corpus)
	m2 := trnLRTrainPair(t, corpus)

	if len(m1.SegmentTable.Counts) != len(m2.SegmentTable.Counts) {
		t.Errorf("segment table count lengths differ: %d vs %d",
			len(m1.SegmentTable.Counts), len(m2.SegmentTable.Counts))
	}
	for k, v1 := range m1.SegmentTable.Counts {
		if v2 := m2.SegmentTable.Counts[k]; v1 != v2 {
			t.Errorf("count[%q]: %v vs %v", k, v1, v2)
		}
	}
}

// TestLongRangeMultiLectHelperFiresOnCleanSignal calls
// commitMultiLectLongRangeSplitsForPivot directly to verify it commits at
// least one entry when given a clean long-range signal (12 front-ctx + 12 plain).
func TestLongRangeMultiLectHelperFiresOnCleanSignal(t *testing.T) {
	frontCtx := Context{
		NextSyllable: []FeatureConstraint{{Feature: "front", Value: "+"}},
	}
	plainCtx := Context{}

	// Build sisterByKey: each key maps to a []lectGrapheme.
	sisterFront := []lectGrapheme{{Lect: "D1", Grapheme: "æ"}, {Lect: "D2", Grapheme: "æ"}}
	sisterPlain := []lectGrapheme{{Lect: "D1", Grapheme: "a"}, {Lect: "D2", Grapheme: "a"}}
	sisterByKey := map[string][]lectGrapheme{
		"front_key": sisterFront,
		"plain_key": sisterPlain,
	}

	// Build observations: 12 front + 12 plain, weight 1.0 each.
	obs := make([]pivotObs, 0, 24)
	for i := 0; i < 12; i++ {
		obs = append(obs, pivotObs{sisterKey: "front_key", ctx: frontCtx, weight: 1.0})
	}
	for i := 0; i < 12; i++ {
		obs = append(obs, pivotObs{sisterKey: "plain_key", ctx: plainCtx, weight: 1.0})
	}

	var committed []committedSplit
	commitMultiLectLongRangeSplitsForPivot(
		"PROTO", "a", obs, sisterByKey, &committed, 0.5, DefaultBICConfig(),
	)

	if len(committed) == 0 {
		t.Fatal("no long-range commit on clean signal")
	}
	c := committed[0]
	if c.pivotLect != "PROTO" {
		t.Errorf("pivotLect = %q, want %q", c.pivotLect, "PROTO")
	}
	if c.pivotGrapheme != "a" {
		t.Errorf("pivotGrapheme = %q, want %q", c.pivotGrapheme, "a")
	}
	hasFront := false
	for _, fc := range c.ctx.NextSyllable {
		if fc.Feature == "front" && fc.Value == "+" {
			hasFront = true
		}
	}
	if !hasFront {
		t.Errorf("committed context NextSyllable %v does not contain front:+", c.ctx.NextSyllable)
	}
	// Sister should be the front set.
	if len(c.sister) != 2 || c.sister[0].Grapheme != "æ" {
		t.Errorf("committed sister %v, want [{D1 æ} {D2 æ}]", c.sister)
	}
	if c.count != 12 {
		t.Errorf("committed count = %v, want 12", c.count)
	}
	if c.bucketSize != 24 {
		t.Errorf("bucket size = %v, want 24", c.bucketSize)
	}
}

// TestLongRangeMultiLectHelperSilentOnNoise checks that the helper commits
// nothing when all observations carry the same sister tuple.
func TestLongRangeMultiLectHelperSilentOnNoise(t *testing.T) {
	frontCtx := Context{
		NextSyllable: []FeatureConstraint{{Feature: "front", Value: "+"}},
	}
	sameSister := []lectGrapheme{{Lect: "D1", Grapheme: "x"}, {Lect: "D2", Grapheme: "x"}}
	sisterByKey := map[string][]lectGrapheme{"only": sameSister}

	obs := make([]pivotObs, 0, 20)
	for i := 0; i < 10; i++ {
		obs = append(obs, pivotObs{sisterKey: "only", ctx: frontCtx, weight: 1.0})
	}
	for i := 0; i < 10; i++ {
		obs = append(obs, pivotObs{sisterKey: "only", ctx: Context{}, weight: 1.0})
	}

	var committed []committedSplit
	commitMultiLectLongRangeSplitsForPivot(
		"PROTO", "a", obs, sisterByKey, &committed, 0.5, DefaultBICConfig(),
	)
	if len(committed) != 0 {
		t.Errorf("expected no commits on noise, got %d", len(committed))
	}
}
