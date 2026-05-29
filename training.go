package regulae

import "fmt"

// Training orchestration. The full TrainModel entry point and the multi-lect
// pipeline are assembled in the reconciliation/training slice; this file
// currently holds the shared training defaults, the internal pair-corpus
// representation, and the alignCorpus convenience used by the EM and discovery
// stages.

// Training defaults, matching the Python module-level constants.
const (
	defaultTemperature        = 1.0
	defaultConcentration      = 5.0
	defaultMaxIter            = 30
	defaultConvergenceEps     = 1e-4
	defaultSegmentWeight      = 0.7
	defaultDisplacementWeight = 0.3
	defaultToneWeight         = 1.0
)

// FormPair is the internal pairwise-corpus element: a directed (source,
// target) form pair. The Python pipeline used tuple[Form, Form].
type FormPair struct {
	Src Form
	Tgt Form
}

// alignCorpus aligns every pair in a corpus under a model, returning one
// Alignment per input pair in corpus order. Panics on unknown graphemes
// (internal use; public boundaries recover).
func alignCorpus(corpus []FormPair, model *LearnedModel, maxChunkSize int) []Alignment {
	out := make([]Alignment, len(corpus))
	for i, p := range corpus {
		out[i] = alignForms(p.Src, p.Tgt, model.FeatureSystem, maxChunkSize, model)
	}
	return out
}

// TrainOptions bundles the tunable hyperparameters for training. Use
// DefaultTrainOptions for the standard configuration and override fields as
// needed. The zero value is NOT valid — always start from DefaultTrainOptions.
type TrainOptions struct {
	FeatureSystem        string
	MaxChunkSize         int
	Temperature          float64
	Concentration        float64
	MaxIter              int
	ConvergenceEps       float64
	SegmentWeight        float64
	DisplacementWeight   float64
	ToneWeight           float64
	ChunkMinTransparency float64
	TypologicalPrior     TypologicalPrior
	BICConfig            BICConfig
	// BootstrapN > 0 replaces closed-form Wilson intervals with bootstrap
	// percentile intervals over BootstrapN resampled trainings; BootstrapSeed
	// seeds the resampler. Training time scales as (1 + BootstrapN) × base.
	BootstrapN    int
	BootstrapSeed int
}

// DefaultTrainOptions returns the standard training configuration, matching the
// Python train_model defaults.
func DefaultTrainOptions() TrainOptions {
	return TrainOptions{
		FeatureSystem:        "descriptive",
		MaxChunkSize:         DefaultMaxChunkSize,
		Temperature:          defaultTemperature,
		Concentration:        defaultConcentration,
		MaxIter:              defaultMaxIter,
		ConvergenceEps:       defaultConvergenceEps,
		SegmentWeight:        defaultSegmentWeight,
		DisplacementWeight:   defaultDisplacementWeight,
		ToneWeight:           defaultToneWeight,
		ChunkMinTransparency: 0.0,
		TypologicalPrior:     nil,
		BICConfig:            DefaultBICConfig(),
	}
}

// TrainPairwise trains a layered LearnedModel on a directed pairwise corpus.
// pairWeights (one per pair; nil = all 1.0) are confidence weights. Returns an
// *UnknownGraphemeError if a grapheme is unrecognised.
func TrainPairwise(corpus []FormPair, pairWeights []float64, opts TrainOptions) (model *LearnedModel, err error) {
	defer catchUnknownGrapheme(&err)
	if len(corpus) == 0 {
		return EmptyLearnedModel(opts.FeatureSystem, opts.Temperature, opts.Concentration), nil
	}
	model = trainPairwiseLegacy(corpus, pairWeights, opts)
	if opts.BootstrapN > 0 {
		model = bootstrapPairwiseUncertainty(model, corpus, opts.BootstrapN, opts.BootstrapSeed, opts)
	}
	return model, nil
}

// TrainModel trains a multi-lect alignment model on a corpus of cognate sets:
// per-pair learned models plus reconciled multi-lect correspondence classes.
// This is the canonical entry point. Returns an error on invalid cognate sets
// or an unrecognised grapheme.
func TrainModel(corpus []CognateSet, opts TrainOptions) (model *MultiLectModel, err error) {
	defer catchUnknownGrapheme(&err)
	if len(corpus) == 0 {
		return EmptyMultiLectModel(), nil
	}
	if err := validateCognateSets(corpus); err != nil {
		return nil, err
	}
	model = trainMultiLect(corpus, opts)
	if opts.BootstrapN > 0 {
		model = bootstrapMultiLectUncertainty(model, corpus, opts.BootstrapN, opts.BootstrapSeed, opts)
	}
	return model, nil
}

// validateCognateSets checks cognate sets for common structural errors.
func validateCognateSets(corpus []CognateSet) error {
	allLects := map[string]bool{}
	for _, cs := range corpus {
		for l := range cs.Forms {
			allLects[l] = true
		}
	}
	nLects := len(allLects)
	for _, cs := range corpus {
		if nLects >= 2 && len(cs.Forms) < 2 {
			return fmt.Errorf("CognateSet %q has %d form(s); at least 2 are required for multi-lect training", cs.CognateID, len(cs.Forms))
		}
		for lectID, form := range cs.Forms {
			if len(form.Segments) == 0 {
				return fmt.Errorf("CognateSet %q, lect %q: form has no segments", cs.CognateID, lectID)
			}
		}
		if cs.Confidence < 0.0 || cs.Confidence > 1.0 {
			return fmt.Errorf("CognateSet %q: confidence must be in [0, 1], got %v", cs.CognateID, cs.Confidence)
		}
	}
	return nil
}

// CognateSetsFromPairs converts a list of pairwise form tuples to cognate sets,
// rewriting each form's lect id to match lectIDs and recording the (src, tgt)
// ordering in FormsOrder so the canonical lect direction is preserved.
func CognateSetsFromPairs(pairs []FormPair, lectIDs [2]string, cognateIDPrefix string) []CognateSet {
	if cognateIDPrefix == "" {
		cognateIDPrefix = "pair"
	}
	srcLect, tgtLect := lectIDs[0], lectIDs[1]
	out := make([]CognateSet, 0, len(pairs))
	for i, p := range pairs {
		srcForm := Form{LectID: srcLect, Segments: p.Src.Segments, SyllableBreaks: p.Src.SyllableBreaks, MorphemeBreaks: p.Src.MorphemeBreaks}
		tgtForm := Form{LectID: tgtLect, Segments: p.Tgt.Segments, SyllableBreaks: p.Tgt.SyllableBreaks, MorphemeBreaks: p.Tgt.MorphemeBreaks}
		out = append(out, CognateSet{
			CognateID:  fmt.Sprintf("%s.%05d", cognateIDPrefix, i),
			Forms:      map[string]Form{srcLect: srcForm, tgtLect: tgtForm},
			FormsOrder: []string{srcLect, tgtLect},
			Confidence: 1.0,
		})
	}
	return out
}

// AlignCorpus aligns every pair in a corpus under a trained model, returning
// one Alignment per input pair in corpus order. Returns an *UnknownGraphemeError
// on an unrecognised grapheme.
func AlignCorpus(corpus []FormPair, model *LearnedModel, maxChunkSize int) (out []Alignment, err error) {
	defer catchUnknownGrapheme(&err)
	if maxChunkSize == 0 {
		maxChunkSize = DefaultMaxChunkSize
	}
	out = alignCorpus(corpus, model, maxChunkSize)
	return out, nil
}

// trainPairwiseLegacy runs the staged pairwise training pipeline:
// initial prior -> segment EM -> displacement aggregation -> context discovery
// -> chunk promotion -> tonal aggregation -> cross-dimensional discovery ->
// long-range context discovery.
func trainPairwiseLegacy(corpus []FormPair, pairWeights []float64, opts TrainOptions) *LearnedModel {
	if pairWeights == nil {
		pairWeights = make([]float64, len(corpus))
		for i := range pairWeights {
			pairWeights[i] = 1.0
		}
	}

	initial := initialModel(corpus, pairWeights, opts.FeatureSystem, opts.Temperature, opts.Concentration, opts.SegmentWeight, opts.DisplacementWeight, opts.TypologicalPrior)
	afterEM := segmentEM(corpus, pairWeights, initial, opts.MaxChunkSize, opts.MaxIter, opts.ConvergenceEps)
	afterDisplacement := displacementAggregation(corpus, pairWeights, afterEM, opts.MaxChunkSize)
	afterContext := contextDiscovery(corpus, pairWeights, afterDisplacement, opts.MaxChunkSize, opts.BICConfig)
	afterChunks := chunkPromotion(corpus, pairWeights, afterContext, opts.MaxChunkSize, opts.ChunkMinTransparency, opts.BICConfig)
	afterTonal := tonalAggregation(corpus, pairWeights, afterChunks, opts.MaxChunkSize)
	afterCrossDim := crossDimensionalDiscovery(corpus, pairWeights, afterTonal, 0, opts.BICConfig)
	afterLongRange := longRangeDiscovery(corpus, pairWeights, afterCrossDim, opts.MaxChunkSize, opts.BICConfig)
	return afterLongRange
}
