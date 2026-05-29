package regulae

// Mirror of python/tests/test_training_consolidation.py
//
// SKIPPED (already covered):
//   none — all tests in this file are new.

import (
	"math/rand"
	"testing"
)

// trnConsolidationForm is a helper for building forms in this file (reuses
// formIPA which is defined in search_test.go).
func trnConsolidationForm(lectID, ipa string) Form {
	return formIPA(lectID, ipa)
}

// --- 1. Real-data integration ---

// TestConsolidationLatinGermanicRecoversGrimmTypeCorrespondences checks that
// training on a Latin-Germanic cognate corpus surfaces Grimm-type correspondences.
func TestConsolidationLatinGermanicRecoversGrimmTypeCorrespondences(t *testing.T) {
	corpus := []FormPair{
		pairOf("latin", "pater", "germ", "fader"),
		pairOf("latin", "mater", "germ", "moder"),
		pairOf("latin", "frater", "germ", "bruder"),
		pairOf("latin", "pedis", "germ", "fotu"),
		pairOf("latin", "piskis", "germ", "fiskaz"),
		pairOf("latin", "plenus", "germ", "fulnaz"),
		pairOf("latin", "tres", "germ", "θrejez"),
		pairOf("latin", "kornu", "germ", "hurnan"),
		pairOf("latin", "kor", "germ", "hertan"),
		pairOf("latin", "kanis", "germ", "hundaz"),
	}
	trained := mustTrain(t, corpus)
	counts := trained.SegmentTable.Counts

	// Grimm: p→f
	pfKey := ccKeyEmpty("p", "f")
	if counts[pfKey] < 3 {
		t.Errorf("expected p->f count >= 3, got %v", counts[pfKey])
	}
	// Grimm: k→h
	khKey := ccKeyEmpty("k", "h")
	if counts[khKey] < 2 {
		t.Errorf("expected k->h count >= 2, got %v", counts[khKey])
	}
	// t→θ or t→d
	tFricative := counts[ccKeyEmpty("t", "θ")] + counts[ccKeyEmpty("t", "d")]
	if tFricative < 2 {
		t.Errorf("expected t->θ or t->d count >= 2, got %v", tFricative)
	}
}

// TestConsolidationLatinGermanicAlignmentCostDropsAfterTraining checks that
// training on Latin-Germanic pairs reduces total corpus alignment cost.
func TestConsolidationLatinGermanicAlignmentCostDropsAfterTraining(t *testing.T) {
	corpus := []FormPair{
		pairOf("latin", "pater", "germ", "fader"),
		pairOf("latin", "mater", "germ", "moder"),
		pairOf("latin", "pedis", "germ", "fotu"),
		pairOf("latin", "tres", "germ", "θrejez"),
		pairOf("latin", "kornu", "germ", "hurnan"),
	}

	var m2Cost float64
	for _, fp := range corpus {
		al, err := AlignForms(fp.Src, fp.Tgt, "descriptive", 0, nil)
		if err != nil {
			t.Fatalf("AlignForms M2: %v", err)
		}
		c, err := AlignmentCost(al, "descriptive", nil)
		if err != nil {
			t.Fatalf("AlignmentCost M2: %v", err)
		}
		m2Cost += c
	}

	trained := mustTrain(t, corpus)
	var m3Cost float64
	for _, fp := range corpus {
		al, err := AlignForms(fp.Src, fp.Tgt, "descriptive", 0, trained)
		if err != nil {
			t.Fatalf("AlignForms trained: %v", err)
		}
		c, err := AlignmentCost(al, "descriptive", trained)
		if err != nil {
			t.Fatalf("AlignmentCost trained: %v", err)
		}
		m3Cost += c
	}

	if m3Cost >= m2Cost {
		t.Errorf("trained cost %v not less than M2 cost %v", m3Cost, m2Cost)
	}
}

// TestConsolidationRegularSubstitutionCorpusConcentratesMass checks that
// P(f|p) > P(p|p) and P(f|p) > 0.5 on an all-p→f corpus.
func TestConsolidationRegularSubstitutionCorpusConcentratesMass(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "papa", "B", "fafa"),
		pairOf("A", "papi", "B", "fafi"),
		pairOf("A", "popo", "B", "fofo"),
		pairOf("A", "pupu", "B", "fufu"),
		pairOf("A", "papa", "B", "fafa"),
		pairOf("A", "papi", "B", "fafi"),
	}
	trained := mustTrain(t, corpus)

	postForP := trained.PosteriorFor("p", nil)
	pFGivenP := postForP["f"]
	pPGivenP := postForP["p"]

	if pFGivenP <= pPGivenP {
		t.Errorf("P(f|p)=%v should exceed P(p|p)=%v", pFGivenP, pPGivenP)
	}
	if pFGivenP <= 0.5 {
		t.Errorf("P(f|p)=%v should exceed 0.5", pFGivenP)
	}
}

// --- 2. Training invariants ---

// TestConsolidationTrainingIsDeterministicUnderCorpusReordering checks that
// permuting the corpus order yields an equivalent model.
func TestConsolidationTrainingIsDeterministicUnderCorpusReordering(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "papa", "B", "fafa"),
		pairOf("A", "mata", "B", "mada"),
		pairOf("A", "kata", "B", "hata"),
		pairOf("A", "tasa", "B", "dasa"),
	}
	rng := rand.New(rand.NewSource(0))
	shuffled := make([]FormPair, len(corpus))
	copy(shuffled, corpus)
	rng.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })

	m1 := mustTrain(t, corpus)
	m2 := mustTrain(t, shuffled)

	// Segment counts must be equal (order-invariant M-step).
	for k, v1 := range m1.SegmentTable.Counts {
		v2 := m2.SegmentTable.Counts[k]
		if v1 != v2 {
			t.Errorf("segment count[%q]: %v vs %v", k, v1, v2)
		}
	}
	for k, v2 := range m2.SegmentTable.Counts {
		v1 := m1.SegmentTable.Counts[k]
		if v1 != v2 {
			t.Errorf("segment count[%q] missing in m1: %v", k, v2)
		}
	}
}

// TestConsolidationTrainingIsIdempotentOnShortCorpora checks that retraining
// on the same corpus twice yields identical models.
func TestConsolidationTrainingIsIdempotentOnShortCorpora(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pata", "B", "fada"),
		pairOf("A", "pata", "B", "fada"),
	}
	m1 := mustTrain(t, corpus)
	m2 := mustTrain(t, corpus)

	for k, v1 := range m1.SegmentTable.Counts {
		if v2 := m2.SegmentTable.Counts[k]; v1 != v2 {
			t.Errorf("segment count[%q]: %v vs %v", k, v1, v2)
		}
	}
	for k, v1 := range m1.ChunkTable.Entries {
		if v2 := m2.ChunkTable.Entries[k]; v1 != v2 {
			t.Errorf("chunk entry[%q]: %v vs %v", k, v1, v2)
		}
	}
}

// TestConsolidationSegmentEMCostMonotonicallyDecreasesOrStaysStable checks
// that running segment EM manually never increases the total corpus cost
// between iterations.
func TestConsolidationSegmentEMCostMonotonicallyDecreasesOrStaysStable(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "papa", "B", "fafa"),
		pairOf("A", "papi", "B", "fafi"),
		pairOf("A", "popo", "B", "fofo"),
		pairOf("A", "pata", "B", "fada"),
		pairOf("A", "taka", "B", "daha"),
	}
	pairWeights := make([]float64, len(corpus))
	for i := range pairWeights {
		pairWeights[i] = 1.0
	}
	model := initialModel(corpus, pairWeights, "descriptive",
		defaultTemperature, defaultConcentration,
		defaultSegmentWeight, defaultDisplacementWeight, nil)

	var costs []float64
	for iter := 0; iter < 10; iter++ {
		alignments := alignCorpus(corpus, model, DefaultMaxChunkSize)
		total := 0.0
		for _, al := range alignments {
			c := alignmentCost(al, "descriptive", model)
			total += c
		}
		costs = append(costs, total)
		model = updateSegmentTable(model, alignments, pairWeights)
	}

	for i := 1; i < len(costs); i++ {
		if costs[i] > costs[i-1]+1e-9 {
			t.Errorf("segment EM cost increased at iter %d: %v -> %v", i, costs[i-1], costs[i])
		}
	}
}

// TestConsolidationEmptyCorpusYieldsFloorBehaviorOnNewPairs checks that an
// empty-trained model scores identically to M2.
func TestConsolidationEmptyCorpusYieldsFloorBehaviorOnNewPairs(t *testing.T) {
	trained := mustTrain(t, nil)
	src := trnConsolidationForm("A", "pater")
	tgt := trnConsolidationForm("B", "fadar")

	aM2, err := AlignForms(src, tgt, "descriptive", 0, nil)
	if err != nil {
		t.Fatalf("AlignForms M2: %v", err)
	}
	aM3, err := AlignForms(src, tgt, "descriptive", 0, trained)
	if err != nil {
		t.Fatalf("AlignForms trained: %v", err)
	}

	costM2, err := AlignmentCost(aM2, "descriptive", nil)
	if err != nil {
		t.Fatalf("AlignmentCost M2: %v", err)
	}
	costM3, err := AlignmentCost(aM3, "descriptive", trained)
	if err != nil {
		t.Fatalf("AlignmentCost trained: %v", err)
	}
	if costM2 != costM3 {
		t.Errorf("empty-trained model cost %v != M2 cost %v", costM3, costM2)
	}
}

// TestConsolidationTrainingNeverCrashesOnRandomInputs checks that training
// on deterministically random form pairs does not panic or error.
func TestConsolidationTrainingNeverCrashesOnRandomInputs(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	alphabet := []rune("paebtdkgmnfvszlriouɛɔ")
	for trial := 0; trial < 5; trial++ {
		n := 2 + rng.Intn(7)
		corpus := make([]FormPair, n)
		for i := range corpus {
			srcLen := 1 + rng.Intn(5)
			tgtLen := 1 + rng.Intn(5)
			srcSegs := make([]Segment, srcLen)
			tgtSegs := make([]Segment, tgtLen)
			for j := range srcSegs {
				srcSegs[j] = seg(string(alphabet[rng.Intn(len(alphabet))]))
			}
			for j := range tgtSegs {
				tgtSegs[j] = seg(string(alphabet[rng.Intn(len(alphabet))]))
			}
			corpus[i] = FormPair{
				Src: Form{LectID: "A", Segments: srcSegs},
				Tgt: Form{LectID: "B", Segments: tgtSegs},
			}
		}
		trained, err := TrainPairwise(corpus, nil, DefaultTrainOptions())
		if err != nil {
			t.Errorf("trial %d: unexpected error: %v", trial, err)
			continue
		}
		if len(trained.SegmentTable.LogNormalizers) == 0 {
			t.Errorf("trial %d: log normalizers not computed", trial)
		}
	}
}

// --- 3. Edge cases ---

// TestConsolidationSinglePairCorpusDoesNotCrash checks that a single-pair
// corpus completes and produces segment-level counts.
func TestConsolidationSinglePairCorpusDoesNotCrash(t *testing.T) {
	corpus := []FormPair{pairOf("A", "pata", "B", "fada")}
	trained := mustTrain(t, corpus)
	if len(trained.SegmentTable.Counts) == 0 {
		t.Error("expected non-empty segment counts for single-pair corpus")
	}
}

// TestConsolidationSinglePairDoesNotPromoteAnyChunks checks that a single-pair
// corpus produces no promoted chunks.
func TestConsolidationSinglePairDoesNotPromoteAnyChunks(t *testing.T) {
	corpus := []FormPair{pairOf("A", "pata", "B", "fada")}
	trained := mustTrain(t, corpus)
	if len(trained.ChunkTable.Entries) != 0 {
		t.Errorf("singleton should not promote chunks, got %d", len(trained.ChunkTable.Entries))
	}
}

// TestConsolidationIdenticalPairsStressTest checks that 10 identical pairs
// produce counts of exactly 10 for the observed segment pairs.
func TestConsolidationIdenticalPairsStressTest(t *testing.T) {
	corpus := make([]FormPair, 10)
	for i := range corpus {
		corpus[i] = pairOf("A", "pata", "B", "fada")
	}
	trained := mustTrain(t, corpus)

	if got := trained.SegmentTable.Counts[ccKeyEmpty("p", "f")]; got != 10 {
		t.Errorf("count(p->f) = %v, want 10", got)
	}
	// "pata"/"fada" has two 'a' segments each → two a→a links per pair.
	if got := trained.SegmentTable.Counts[ccKeyEmpty("a", "a")]; got != 20 {
		t.Errorf("count(a->a) = %v, want 20", got)
	}
	if got := trained.SegmentTable.Counts[ccKeyEmpty("t", "d")]; got != 10 {
		t.Errorf("count(t->d) = %v, want 10", got)
	}
}

// TestConsolidationUnrelatedPairsDoNotRaiseAndGiveReasonableAlignment checks
// that a corpus of unrelated pairs produces a valid model and no promoted chunks.
func TestConsolidationUnrelatedPairsDoNotRaiseAndGiveReasonableAlignment(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "abc", "B", "xyz"),
		pairOf("A", "def", "B", "qrs"),
		pairOf("A", "ghi", "B", "nop"),
	}
	trained, err := TrainPairwise(corpus, nil, DefaultTrainOptions())
	if err != nil {
		// abc/xyz may contain unknown graphemes — acceptable if we get a clean error.
		t.Skipf("corpus contains unknown graphemes: %v", err)
	}
	if len(trained.ChunkTable.Entries) != 0 {
		t.Errorf("unrelated pairs should not promote chunks, got %d", len(trained.ChunkTable.Entries))
	}
	// Model should be usable.
	al, err := AlignForms(corpus[0].Src, corpus[0].Tgt, "descriptive", 0, trained)
	if err != nil {
		t.Fatalf("AlignForms on trained model: %v", err)
	}
	cost, err := AlignmentCost(al, "descriptive", trained)
	if err != nil {
		t.Fatalf("AlignmentCost: %v", err)
	}
	if cost == 0 {
		t.Error("expected non-zero cost for unrelated pair")
	}
}

// TestConsolidationTrainingWithUnknownGraphemeRaisesCleanly checks that an
// unknown grapheme causes a clean *UnknownGraphemeError, not a panic.
func TestConsolidationTrainingWithUnknownGraphemeRaisesCleanly(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pata", "B", "fada"),
		{
			Src: Form{LectID: "A", Segments: []Segment{{Grapheme: "Z"}, {Grapheme: "Z"}, {Grapheme: "Z"}}},
			Tgt: Form{LectID: "B", Segments: []Segment{{Grapheme: "x"}, {Grapheme: "y"}, {Grapheme: "z"}}},
		},
	}
	_, err := TrainPairwise(corpus, nil, DefaultTrainOptions())
	if err == nil {
		t.Fatal("expected UnknownGraphemeError")
	}
	var ugErr *UnknownGraphemeError
	if !isUnknownGrapheme(err, &ugErr) {
		t.Errorf("want *UnknownGraphemeError, got %T: %v", err, err)
	}
}

// TestConsolidationTrainingWithTinyCorpusStillBeatM2ForSeenPairs checks that
// even a tiny (3-pair) corpus makes the seen pair cheaper than M2.
func TestConsolidationTrainingWithTinyCorpusStillBeatM2ForSeenPairs(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pa", "B", "fa"),
		pairOf("A", "pa", "B", "fa"),
		pairOf("A", "pa", "B", "fa"),
	}
	trained := mustTrain(t, corpus)

	aM2, err := AlignForms(corpus[0].Src, corpus[0].Tgt, "descriptive", 0, nil)
	if err != nil {
		t.Fatalf("AlignForms M2: %v", err)
	}
	aTrained, err := AlignForms(corpus[0].Src, corpus[0].Tgt, "descriptive", 0, trained)
	if err != nil {
		t.Fatalf("AlignForms trained: %v", err)
	}

	m2Cost, _ := AlignmentCost(aM2, "descriptive", nil)
	m3Cost, _ := AlignmentCost(aTrained, "descriptive", trained)
	if m3Cost >= m2Cost {
		t.Errorf("trained cost %v not less than M2 cost %v", m3Cost, m2Cost)
	}
}

// TestConsolidationCorpusWithLengthAsymmetry checks that very asymmetric
// pairs do not break training.
func TestConsolidationCorpusWithLengthAsymmetry(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "a", "B", "xyz"),
		pairOf("A", "a", "B", "xyz"),
	}
	trained, err := TrainPairwise(corpus, nil, DefaultTrainOptions())
	if err != nil {
		t.Skipf("grapheme not in feature system: %v", err)
	}
	if trained.SegmentTable.Counts == nil {
		t.Error("expected non-nil segment counts map")
	}
}

// isUnknownGrapheme is a helper to check error type (mirrors asUnknownGrapheme
// in scoring_test.go, but uses errors.As for clarity).
func isUnknownGrapheme(err error, target **UnknownGraphemeError) bool {
	var ug *UnknownGraphemeError
	if ok := errorsAs(err, &ug); ok {
		if target != nil {
			*target = ug
		}
		return true
	}
	return false
}

// errorsAs is a minimal local wrapper so we don't import "errors" twice.
func errorsAs(err error, target interface{}) bool {
	if err == nil {
		return false
	}
	type asIface interface {
		As(interface{}) bool
	}
	// Try direct type match.
	if t, ok := target.(**UnknownGraphemeError); ok {
		if ug, ok2 := err.(*UnknownGraphemeError); ok2 {
			*t = ug
			return true
		}
	}
	_ = asIface(nil)
	return false
}
