package regulae

import (
	"sort"
	"strings"
)

// Data types for the learned model. This file holds only data types and the
// pure-lookup methods on them — no training or scoring logic (those live in
// training.go and scoring.go), keeping the model's structure easy to inspect.
//
// Several Python tables were keyed by frozen dataclasses or segment tuples,
// which are not Go-comparable. Those tables are represented here as maps keyed
// by a canonical string (the structured key's key() method), with a parallel
// map recovering the structured key wherever deterministic iteration needs it
// (e.g. Context for subset matching). Comparable keys (TonalCorrespondence)
// remain direct map keys.

// ConditionedCorrespondence identifies one segment correspondence together
// with its conditioning context. Context-free correspondences use the empty
// Context{}.
type ConditionedCorrespondence struct {
	Src     string
	Tgt     string
	Context Context
}

func (c ConditionedCorrespondence) key() string {
	return c.Src + fieldSep + c.Tgt + fieldSep + c.Context.key()
}

// TonalCorrespondence identifies one tone-to-tone correspondence. Tones are
// strings on Segment; the empty string represents an untoned segment (Python's
// None). This struct is comparable and used directly as a map key.
type TonalCorrespondence struct {
	SrcTone string
	TgtTone string
}

// TonalCorrespondenceTable holds learned tone-to-tone correspondence counts
// with a Dirichlet prior. An empty table contributes zero cost at scoring
// time, staying inert for non-tonal corpora.
type TonalCorrespondenceTable struct {
	Counts            map[TonalCorrespondence]float64
	PriorPseudoCounts map[TonalCorrespondence]float64
	SrcTotals         map[string]float64
	Uncertainty       map[TonalCorrespondence]UncertaintyEstimate
}

func NewTonalCorrespondenceTable() TonalCorrespondenceTable {
	return TonalCorrespondenceTable{
		Counts:            map[TonalCorrespondence]float64{},
		PriorPseudoCounts: map[TonalCorrespondence]float64{},
		SrcTotals:         map[string]float64{},
		Uncertainty:       map[TonalCorrespondence]UncertaintyEstimate{},
	}
}

// CrossDimensionalLink is a learned correspondence that crosses phonological
// dimensions — e.g. a source voiced initial consonant predicting a target tone
// (tonogenesis). SrcPosition / SrcPosition2 are "relative_-1", "relative_0",
// "relative_+1". TgtDimension is currently only "tone". A nil SrcFeature2 /
// empty SrcPosition2 means a single-predictor rule. Uncertainty is a pointer
// (nil = unpopulated); it is excluded from equality.
type CrossDimensionalLink struct {
	SrcFeature        FeatureConstraint
	SrcPosition       string
	TgtDimension      string
	TgtValue          string
	TgtPositionOffset int
	Count             float64
	SrcCount          float64
	Confidence        float64

	SrcFeature2  *FeatureConstraint
	SrcPosition2 string

	Uncertainty *UncertaintyEstimate
}

// CrossDimensionalLinkTable holds committed CrossDimensionalLink rules. An
// empty table contributes zero cost at scoring time.
type CrossDimensionalLinkTable struct {
	Entries []CrossDimensionalLink
}

// MultiLectCrossDimensionalLink is the multi-lect analogue of
// CrossDimensionalLink, with explicit SrcLect / TgtLect labels. The multi-lect
// table is built by projecting per-pair commits.
type MultiLectCrossDimensionalLink struct {
	SrcLect           string
	TgtLect           string
	SrcFeature        FeatureConstraint
	SrcPosition       string
	TgtDimension      string
	TgtValue          string
	TgtPositionOffset int
	Count             float64
	SrcCount          float64
	Confidence        float64

	SrcFeature2  *FeatureConstraint
	SrcPosition2 string

	Uncertainty *UncertaintyEstimate
}

// MultiLectCrossDimensionalLinkTable holds committed
// MultiLectCrossDimensionalLink rules, populated by projecting per-pair
// commits after the multi-lect class-discovery stage.
type MultiLectCrossDimensionalLinkTable struct {
	Entries []MultiLectCrossDimensionalLink
}

// SegmentCorrespondenceTable holds observed segment-pair counts plus Dirichlet
// prior pseudo-counts. The posterior predictive of tgt given src is
//
//	P(t | s) = (alpha(t | s) + N(t | s)) / (beta + N(s))
//
// where Counts is N(t|s), PriorPseudoCounts is alpha(t|s), SrcTotals is N(s),
// and beta (concentration) lives on the parent LearnedModel. LogNormalizers
// caches log(Z(s)) for the prior partition function, subtracted from -log
// P_post at scoring so that at zero observations the cost equals the merkmal
// distance exactly.
//
// The CC-keyed maps are keyed by ConditionedCorrespondence.key(); Corr
// recovers the structured key for iteration (scoring needs Context).
type SegmentCorrespondenceTable struct {
	Counts            map[string]float64
	PriorPseudoCounts map[string]float64
	SrcTotals         map[string]float64
	LogNormalizers    map[string]float64
	Uncertainty       map[string]UncertaintyEstimate
	Corr              map[string]ConditionedCorrespondence
}

func NewSegmentCorrespondenceTable() SegmentCorrespondenceTable {
	return SegmentCorrespondenceTable{
		Counts:            map[string]float64{},
		PriorPseudoCounts: map[string]float64{},
		SrcTotals:         map[string]float64{},
		LogNormalizers:    map[string]float64{},
		Uncertainty:       map[string]UncertaintyEstimate{},
		Corr:              map[string]ConditionedCorrespondence{},
	}
}

// setCount records a count for a correspondence, registering its structured key.
func (t *SegmentCorrespondenceTable) setCount(cc ConditionedCorrespondence, v float64) {
	k := cc.key()
	t.Counts[k] = v
	t.Corr[k] = cc
}

// setPrior records a prior pseudo-count for a correspondence, registering its
// structured key.
func (t *SegmentCorrespondenceTable) setPrior(cc ConditionedCorrespondence, v float64) {
	k := cc.key()
	t.PriorPseudoCounts[k] = v
	t.Corr[k] = cc
}

// cloneCorr returns a shallow copy of the structured-key registry.
func cloneCorr(src map[string]ConditionedCorrespondence) map[string]ConditionedCorrespondence {
	out := make(map[string]ConditionedCorrespondence, len(src))
	for k, v := range src {
		out[k] = v
	}
	return out
}

// sortedCorrKeys returns the count keys in deterministic order.
func (t SegmentCorrespondenceTable) sortedCorrKeys() []string {
	keys := make([]string, 0, len(t.Counts))
	for k := range t.Counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// displacementKey builds a canonical key for a displacement tuple. The tuple
// is assumed already canonicalised (sorted) at insertion, matching Python.
func displacementKey(ds []FeatureDisplacement) string {
	var b strings.Builder
	for i, d := range ds {
		if i > 0 {
			b.WriteString(keySep)
		}
		b.WriteString(d.Feature)
		b.WriteString("|")
		b.WriteString(d.FromValue)
		b.WriteString("|")
		b.WriteString(d.ToValue)
	}
	return b.String()
}

// DisplacementDistribution is a categorical distribution over feature
// displacement vectors. Keys are canonical (sorted) displacement tuples; Disp
// recovers the structured tuple. An empty distribution contributes nothing to
// scoring.
type DisplacementDistribution struct {
	Counts           map[string]float64
	Total            float64
	PriorPseudoCount float64
	Uncertainty      map[string]UncertaintyEstimate
	Disp             map[string][]FeatureDisplacement
}

func NewDisplacementDistribution() DisplacementDistribution {
	return DisplacementDistribution{
		Counts:           map[string]float64{},
		PriorPseudoCount: 1.0,
		Uncertainty:      map[string]UncertaintyEstimate{},
		Disp:             map[string][]FeatureDisplacement{},
	}
}

func (d *DisplacementDistribution) setCount(ds []FeatureDisplacement, v float64) {
	k := displacementKey(ds)
	d.Counts[k] = v
	d.Disp[k] = ds
}

// ChunkPair is the structured key for a chunk-table entry.
type ChunkPair struct {
	Src []Segment
	Tgt []Segment
}

func chunkPairKey(src, tgt []Segment) string {
	return segmentsKey(src) + fieldSep + segmentsKey(tgt)
}

// ChunkPhraseTable holds promoted multi-segment correspondences, each mapping
// a (source chunk, target chunk) pair to a direct cost. Only chunks that
// passed BIC promotion appear. Entries override the compositional fallback for
// their chunk pair at scoring time. Maps are keyed by chunkPairKey; Pairs
// recovers the structured chunk pair.
type ChunkPhraseTable struct {
	Entries           map[string]float64
	ObservationCounts map[string]float64
	Uncertainty       map[string]UncertaintyEstimate
	Diagnostics       map[string]ChunkTransparencyReport
	Pairs             map[string]ChunkPair
}

func NewChunkPhraseTable() ChunkPhraseTable {
	return ChunkPhraseTable{
		Entries:           map[string]float64{},
		ObservationCounts: map[string]float64{},
		Uncertainty:       map[string]UncertaintyEstimate{},
		Diagnostics:       map[string]ChunkTransparencyReport{},
		Pairs:             map[string]ChunkPair{},
	}
}

func (t *ChunkPhraseTable) setEntry(src, tgt []Segment, cost float64) {
	k := chunkPairKey(src, tgt)
	t.Entries[k] = cost
	t.Pairs[k] = ChunkPair{Src: src, Tgt: tgt}
}

// LearnedModel is a trained learned model: the layered tables plus
// hyperparameters. Passed to ScoreLink and AlignForms as an optional
// parameter; a nil model means prior-only scoring (merkmal distances).
//
// Hyperparameters: Temperature (tau) controls how sharply merkmal distance
// becomes a prior; Concentration (beta) is the Dirichlet concentration;
// SegmentWeight and DisplacementWeight are the log-linear layer weights (should
// sum to 1.0, not enforced).
type LearnedModel struct {
	SegmentTable          SegmentCorrespondenceTable
	DisplacementDist      DisplacementDistribution
	ChunkTable            ChunkPhraseTable
	TonalTable            TonalCorrespondenceTable
	FeatureSystem         string
	Temperature           float64
	Concentration         float64
	SegmentWeight         float64
	DisplacementWeight    float64
	ToneWeight            float64
	CrossDimensionalTable CrossDimensionalLinkTable
}

// EmptyLearnedModel creates an empty model (no learned data). Scoring under it
// falls back to merkmal distances for everything — the prior-only fallback
// used before any training data is available.
func EmptyLearnedModel(featureSystem string, temperature, concentration float64) *LearnedModel {
	if featureSystem == "" {
		featureSystem = "descriptive"
	}
	return &LearnedModel{
		SegmentTable:          NewSegmentCorrespondenceTable(),
		DisplacementDist:      NewDisplacementDistribution(),
		ChunkTable:            NewChunkPhraseTable(),
		TonalTable:            NewTonalCorrespondenceTable(),
		CrossDimensionalTable: CrossDimensionalLinkTable{},
		FeatureSystem:         featureSystem,
		Temperature:           temperature,
		Concentration:         concentration,
		SegmentWeight:         0.7,
		DisplacementWeight:    0.3,
		ToneWeight:            1.0,
	}
}

// PosteriorFor returns P(tgt | src, context) for every target grapheme the
// model could emit for src. If any conditioned entry matches the context, only
// entries at maximum specificity are used; unconditioned entries are the
// fallback only when no conditioned entry matches. Probabilities are
// Dirichlet-smoothed with the winning entries' pseudo-counts and the model's
// Concentration. A nil/empty context queries the unconditioned posterior.
// Returns an empty map when src has no entries; does not include the merkmal
// fallback used at scoring time for unknown pairs.
func (m *LearnedModel) PosteriorFor(src string, context *Context) map[string]float64 {
	linkContext := Context{}
	if context != nil {
		linkContext = *context
	}

	matches := map[string]ConditionedCorrespondence{} // tgt -> winning key
	maxSpecificity := -1
	for _, k := range m.SegmentTable.sortedCorrKeys() {
		key := m.SegmentTable.Corr[k]
		if key.Src != src {
			continue
		}
		if !key.Context.IsSubsetOf(linkContext) {
			continue
		}
		spec := key.Context.ConstraintCount()
		if spec > maxSpecificity {
			maxSpecificity = spec
			matches = map[string]ConditionedCorrespondence{}
		}
		if spec == maxSpecificity {
			existing, ok := matches[key.Tgt]
			if !ok {
				matches[key.Tgt] = key
			} else if key.Context.ConstraintCount() > existing.Context.ConstraintCount() {
				matches[key.Tgt] = key
			}
		}
	}
	if maxSpecificity < 0 {
		// No entry matches; fall back to unconditioned prior pseudo-counts.
		priorKeys := make([]string, 0, len(m.SegmentTable.PriorPseudoCounts))
		for k := range m.SegmentTable.PriorPseudoCounts {
			priorKeys = append(priorKeys, k)
		}
		sort.Strings(priorKeys)
		for _, k := range priorKeys {
			key := m.SegmentTable.Corr[k]
			if key.Src == src && key.Context.ConstraintCount() == 0 {
				matches[key.Tgt] = key
			}
		}
	}

	countSum := 0.0
	alphaSum := 0.0
	for _, key := range matches {
		k := key.key()
		countSum += m.SegmentTable.Counts[k]
		alphaSum += m.SegmentTable.PriorPseudoCounts[k]
	}
	denominator := alphaSum + countSum
	if denominator <= 0.0 {
		return map[string]float64{}
	}
	out := map[string]float64{}
	for tgt, key := range matches {
		k := key.key()
		alpha := m.SegmentTable.PriorPseudoCounts[k]
		n := m.SegmentTable.Counts[k]
		out[tgt] = (alpha + n) / denominator
	}
	return out
}

// ----- multi-lect types ---------------------------------------------------

// CognateSet is a set of cognate forms across N lects (N=1 single-lect, N=2
// pairwise, N>=3 multi-lect). Cognate membership is explicit user input; the
// framework never infers cognacy and never drops sets during training.
//
// Alignments and MorphemeBoundaries are reserved (captured by loaders when
// present but not currently consumed). Confidence in [0, 1] is an evidence
// weight: lower-confidence sets still align and stay in the corpus for
// auditability but contribute proportionally less mass.
type CognateSet struct {
	CognateID          string
	Forms              map[string]Form
	Alignments         map[string][]*Segment
	MorphemeBoundaries map[string][]int
	Confidence         float64
	// FormsOrder records the lect-id insertion order of Forms, which Go maps
	// do not preserve. It determines the canonical lect ordering (and thus
	// pair-training direction) the way Python's ordered dict did. When nil,
	// the sorted form keys are used as a fallback.
	FormsOrder []string
}

// orderedLects returns the lect ids of the cognate set in canonical order:
// FormsOrder if set (filtered to present forms), otherwise sorted form keys.
func (cs CognateSet) orderedLects() []string {
	if len(cs.FormsOrder) > 0 {
		out := make([]string, 0, len(cs.FormsOrder))
		seen := map[string]bool{}
		for _, l := range cs.FormsOrder {
			if _, ok := cs.Forms[l]; ok && !seen[l] {
				out = append(out, l)
				seen[l] = true
			}
		}
		return out
	}
	out := make([]string, 0, len(cs.Forms))
	for l := range cs.Forms {
		out = append(out, l)
	}
	sort.Strings(out)
	return out
}

// MultiLectCorrespondenceClass binds one grapheme per participating lect,
// optionally with per-lect contexts. A nil Contexts means unconditioned; a
// non-nil Contexts is context-conditioned (lects without a constraint hold the
// empty Context{}). Confidence is the pivot-bucket coverage for conditioned
// classes, 1.0 for unconditioned. Uncertainty (nil = unpopulated) is excluded
// from equality.
type MultiLectCorrespondenceClass struct {
	ClassID            int
	Segments           map[string]string
	Contexts           map[string]Context
	Count              float64
	SupportingCognates []string
	Confidence         float64
	Uncertainty        *UncertaintyEstimate
}

// MultiLectModel is a trained multi-lect alignment model: the per-pair learned
// models, reconciled correspondence classes (unconditioned and conditioned),
// and the retained corpus. PairwiseModels is keyed by lectPairKey(a, b); use
// PairwiseModel to look up by lect-id pair.
type MultiLectModel struct {
	PairwiseModels        map[string]*LearnedModel
	UnconditionedClasses  []MultiLectCorrespondenceClass
	ConditionedClasses    []MultiLectCorrespondenceClass
	CognateCorpus         []CognateSet
	LectIDs               []string
	CrossDimensionalTable MultiLectCrossDimensionalLinkTable
}

// lectPairKey returns the canonical key for an unordered pair of lect ids.
func lectPairKey(a, b string) string {
	if a <= b {
		return a + fieldSep + b
	}
	return b + fieldSep + a
}

// EmptyMultiLectModel creates an empty multi-lect model.
func EmptyMultiLectModel() *MultiLectModel {
	return &MultiLectModel{
		PairwiseModels:       map[string]*LearnedModel{},
		UnconditionedClasses: nil,
		ConditionedClasses:   nil,
		CognateCorpus:        nil,
		LectIDs:              nil,
	}
}

// PairwiseModel returns the learned model trained on the unordered lect pair
// (a, b), or nil/false if there was no shared cognate data for it.
func (m *MultiLectModel) PairwiseModel(a, b string) (*LearnedModel, bool) {
	lm, ok := m.PairwiseModels[lectPairKey(a, b)]
	return lm, ok
}

// ClassByID returns the class with the given ID (searching both tables), or
// nil if none. ClassIDs are unique within a model.
func (m *MultiLectModel) ClassByID(classID int) *MultiLectCorrespondenceClass {
	for i := range m.UnconditionedClasses {
		if m.UnconditionedClasses[i].ClassID == classID {
			return &m.UnconditionedClasses[i]
		}
	}
	for i := range m.ConditionedClasses {
		if m.ConditionedClasses[i].ClassID == classID {
			return &m.ConditionedClasses[i]
		}
	}
	return nil
}

// ClassesForSegment returns every class that binds grapheme to lectID,
// unconditioned matches first then conditioned, preserving source-table order.
func (m *MultiLectModel) ClassesForSegment(lectID, grapheme string, includeUnconditioned, includeConditioned bool) []MultiLectCorrespondenceClass {
	var out []MultiLectCorrespondenceClass
	if includeUnconditioned {
		for _, klass := range m.UnconditionedClasses {
			if klass.Segments[lectID] == grapheme {
				out = append(out, klass)
			}
		}
	}
	if includeConditioned {
		for _, klass := range m.ConditionedClasses {
			if klass.Segments[lectID] == grapheme {
				out = append(out, klass)
			}
		}
	}
	return out
}

// CognateSetByID returns the cognate set with the given ID from the retained
// corpus, or nil if not present.
func (m *MultiLectModel) CognateSetByID(cognateID string) *CognateSet {
	for i := range m.CognateCorpus {
		if m.CognateCorpus[i].CognateID == cognateID {
			return &m.CognateCorpus[i]
		}
	}
	return nil
}
