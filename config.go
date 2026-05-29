package regulae

import "fmt"

// BICConfig collects the tunable thresholds for the BIC-gated discovery
// stages into one struct. The defaults (see DefaultBICConfig) match the
// historical module-level constants used across discovery, chunks,
// cross-dimensional, and reconciliation.
//
// Note: unlike Python, the Go zero value of BICConfig is NOT the default
// configuration (the defaults are non-zero). Callers wanting defaults must use
// DefaultBICConfig(); training treats a zero-valued config as "use defaults".
//
// Tighten to reject more commits on noisy data; loosen to commit more rules on
// sparse data.
type BICConfig struct {
	// DeltaBICThreshold is the safety buffer on ΔBIC for immediate-neighbour
	// context splits. A split commits only when its BIC delta is more
	// negative than this. -1.0 rejects near-zero spurious splits.
	DeltaBICThreshold float64

	// MinSplitObservations is the minimum weighted observation count for a
	// split branch. Prevents degenerate 1-observation partitions.
	MinSplitObservations int

	// MaxSplitDepth caps context splits committed on the same source
	// grapheme, preventing pathological multi-level splits.
	MaxSplitDepth int

	// MinChunkObservations is the minimum chunk observations for promotion
	// consideration. Below this, Laplace-smoothed MLE gives P=1 from a single
	// observation and BIC cannot reject the spurious promotion.
	MinChunkObservations int

	// LongRangeDeltaBICThreshold is the ΔBIC threshold for long-range context
	// splits. Stricter than immediate-neighbour because the candidate space is
	// ~4x larger.
	LongRangeDeltaBICThreshold float64

	// LongRangeMinSplitObservations is the minimum weighted observation count
	// per branch for long-range splits.
	LongRangeMinSplitObservations int

	// LongRangeMinDominantFraction is the minimum fraction of YES observations
	// that must carry the same target outcome for a long-range split to
	// commit. Rejects grab-bag splits that lump heterogeneous rare outcomes.
	LongRangeMinDominantFraction float64

	// CrossDimMaxIterations caps sequential-greedy iterations in
	// cross-dimensional discovery.
	CrossDimMaxIterations int

	// CrossDimMinRuleCount is the minimum support count for a committed
	// cross-dimensional rule. BIC alone admits count=1/N noise on corpora with
	// many rare tonal outcomes.
	CrossDimMinRuleCount int

	// CrossDimMinRuleConfidence is the minimum confidence (count/src_count)
	// for a committed cross-dimensional rule.
	CrossDimMinRuleConfidence float64

	// MultiLectBICSmallSampleCorrection enables an AICc-style small-sample
	// correction on the multi-lect class-discovery BIC penalty.
	MultiLectBICSmallSampleCorrection bool

	// MultiLectMinCommitScale scales the adaptive floor for per-sister-tuple
	// emissions in the multi-lect class-discovery loop. Floor is
	// max(2, ceil(scale * log2(n+1))); 0.0 disables it entirely.
	MultiLectMinCommitScale float64
}

// DefaultBICConfig returns the default configuration, equivalent to a
// default-constructed Python BICConfig().
func DefaultBICConfig() BICConfig {
	return BICConfig{
		DeltaBICThreshold:                 -1.0,
		MinSplitObservations:              2,
		MaxSplitDepth:                     3,
		MinChunkObservations:              2,
		LongRangeDeltaBICThreshold:        -5.0,
		LongRangeMinSplitObservations:     5,
		LongRangeMinDominantFraction:      0.6,
		CrossDimMaxIterations:             5,
		CrossDimMinRuleCount:              3,
		CrossDimMinRuleConfidence:         0.5,
		MultiLectBICSmallSampleCorrection: true,
		MultiLectMinCommitScale:           0.5,
	}
}

// Validate checks field bounds, failing fast on nonsense values. Mirrors the
// Python __post_init__ validation.
func (c BICConfig) Validate() error {
	if c.MinSplitObservations < 1 {
		return fmt.Errorf("MinSplitObservations must be >= 1, got %d", c.MinSplitObservations)
	}
	if c.MaxSplitDepth < 1 {
		return fmt.Errorf("MaxSplitDepth must be >= 1, got %d", c.MaxSplitDepth)
	}
	if c.MinChunkObservations < 1 {
		return fmt.Errorf("MinChunkObservations must be >= 1, got %d", c.MinChunkObservations)
	}
	if c.LongRangeMinSplitObservations < 1 {
		return fmt.Errorf("LongRangeMinSplitObservations must be >= 1, got %d", c.LongRangeMinSplitObservations)
	}
	if c.LongRangeMinDominantFraction < 0.0 || c.LongRangeMinDominantFraction > 1.0 {
		return fmt.Errorf("LongRangeMinDominantFraction must be in [0, 1], got %v", c.LongRangeMinDominantFraction)
	}
	if c.CrossDimMaxIterations < 1 {
		return fmt.Errorf("CrossDimMaxIterations must be >= 1, got %d", c.CrossDimMaxIterations)
	}
	if c.CrossDimMinRuleCount < 1 {
		return fmt.Errorf("CrossDimMinRuleCount must be >= 1, got %d", c.CrossDimMinRuleCount)
	}
	if c.CrossDimMinRuleConfidence < 0.0 || c.CrossDimMinRuleConfidence > 1.0 {
		return fmt.Errorf("CrossDimMinRuleConfidence must be in [0, 1], got %v", c.CrossDimMinRuleConfidence)
	}
	if c.MultiLectMinCommitScale < 0.0 {
		return fmt.Errorf("MultiLectMinCommitScale must be >= 0, got %v", c.MultiLectMinCommitScale)
	}
	return nil
}
