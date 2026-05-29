package regulae

import (
	"reflect"
	"testing"
)

func pairOf(srcLect, srcIPA, tgtLect, tgtIPA string) FormPair {
	return FormPair{Src: formIPA(srcLect, srcIPA), Tgt: formIPA(tgtLect, tgtIPA)}
}

func mustTrain(t *testing.T, corpus []FormPair) *LearnedModel {
	t.Helper()
	m, err := TrainPairwise(corpus, nil, DefaultTrainOptions())
	if err != nil {
		t.Fatalf("TrainPairwise error: %v", err)
	}
	return m
}

func ccKeyEmpty(src, tgt string) string {
	return ConditionedCorrespondence{Src: src, Tgt: tgt, Context: Context{}}.key()
}

func TestTrainEmptyCorpusReturnsEmptyModel(t *testing.T) {
	m, err := TrainPairwise(nil, nil, DefaultTrainOptions())
	if err != nil {
		t.Fatal(err)
	}
	if len(m.SegmentTable.Counts) != 0 || len(m.ChunkTable.Entries) != 0 {
		t.Error("empty corpus should yield an empty model")
	}
}

func TestRegularCorrespondenceRecovery(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pap", "B", "faf"),
		pairOf("A", "pip", "B", "fif"),
		pairOf("A", "pop", "B", "fof"),
		pairOf("A", "pup", "B", "fuf"),
		pairOf("A", "pep", "B", "fef"),
		pairOf("A", "pap", "B", "faf"),
		pairOf("A", "pip", "B", "fif"),
		pairOf("A", "pop", "B", "fof"),
		pairOf("A", "pup", "B", "fuf"),
		pairOf("A", "pep", "B", "fef"),
	}
	trained := mustTrain(t, corpus)

	if got := trained.SegmentTable.Counts[ccKeyEmpty("p", "f")]; got < 10.0 {
		t.Errorf("count(p->f) = %v, want >= 10", got)
	}

	learnedCost := segmentPairCostWithModel(Segment{Grapheme: "p"}, Segment{Grapheme: "f"}, trained, Context{})
	if learnedCost >= 0.625 {
		t.Errorf("learned cost(p,f) = %v, want < 0.625 (the M2 baseline)", learnedCost)
	}
}

func TestRegularCorrespondenceMakesAlignmentCheaper(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pap", "B", "faf"),
		pairOf("A", "pap", "B", "faf"),
		pairOf("A", "pap", "B", "faf"),
	}
	trained := mustTrain(t, corpus)
	aM2, _ := AlignForms(corpus[0].Src, corpus[0].Tgt, "descriptive", 0, nil)
	aTrained, _ := AlignForms(corpus[0].Src, corpus[0].Tgt, "descriptive", 0, trained)
	costTrained, _ := AlignmentCost(aTrained, "descriptive", trained)
	costM2, _ := AlignmentCost(aM2, "descriptive", nil)
	if costTrained >= costM2 {
		t.Errorf("trained cost %v not less than M2 cost %v", costTrained, costM2)
	}
}

func TestDisplacementLayerPopulated(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pa", "B", "fa"),
		pairOf("A", "pa", "B", "fa"),
		pairOf("A", "ta", "B", "sa"),
		pairOf("A", "ta", "B", "sa"),
		pairOf("A", "ka", "B", "xa"),
		pairOf("A", "ka", "B", "xa"),
	}
	trained := mustTrain(t, corpus)
	if len(trained.DisplacementDist.Counts) == 0 || trained.DisplacementDist.Total <= 0 {
		t.Error("displacement distribution should be populated")
	}
}

func TestChunkPromotionForRecurringCluster(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "akta", "B", "atta"),
		pairOf("A", "ikti", "B", "itti"),
		pairOf("A", "okto", "B", "otto"),
		pairOf("A", "ukte", "B", "utte"),
		pairOf("A", "akta", "B", "atta"),
		pairOf("A", "ikti", "B", "itti"),
		pairOf("A", "okto", "B", "otto"),
		pairOf("A", "ukte", "B", "utte"),
		pairOf("A", "akto", "B", "atto"),
		pairOf("A", "ikti", "B", "itti"),
		pairOf("A", "ukte", "B", "utte"),
		pairOf("A", "okta", "B", "otta"),
		pairOf("A", "akte", "B", "atte"),
		pairOf("A", "ikto", "B", "itto"),
		pairOf("A", "uktu", "B", "uttu"),
	}
	trained := mustTrain(t, corpus)
	if len(trained.ChunkTable.Entries) == 0 {
		t.Error("expected at least one promoted chunk for the recurring kt->tt cluster")
	}
}

func TestChunkPromotionRejectsSingletons(t *testing.T) {
	corpus := []FormPair{pairOf("A", "akta", "B", "atta")}
	trained := mustTrain(t, corpus)
	if len(trained.ChunkTable.Entries) != 0 {
		t.Errorf("singleton chunk should not be promoted, got %d entries", len(trained.ChunkTable.Entries))
	}
}

func TestTrainDeterministic(t *testing.T) {
	corpus := []FormPair{
		pairOf("A", "pater", "B", "fadar"),
		pairOf("A", "mater", "B", "madar"),
	}
	m1 := mustTrain(t, corpus)
	m2 := mustTrain(t, corpus)
	if !reflect.DeepEqual(m1.SegmentTable.Counts, m2.SegmentTable.Counts) {
		t.Error("segment counts not deterministic across runs")
	}
	if !reflect.DeepEqual(m1.ChunkTable.Entries, m2.ChunkTable.Entries) {
		t.Error("chunk entries not deterministic across runs")
	}
}

func TestPosteriorForLearnedCorrespondence(t *testing.T) {
	corpus := make([]FormPair, 0, 8)
	for i := 0; i < 8; i++ {
		corpus = append(corpus, pairOf("A", "pa", "B", "fa"))
	}
	trained := mustTrain(t, corpus)
	post := trained.PosteriorFor("p", nil)
	if post["f"] <= 0.5 {
		t.Errorf("P(f|p) = %v, want > 0.5 after training on p->f", post["f"])
	}
}
