package regulae

import "testing"

func TestDefaultBICConfigValues(t *testing.T) {
	c := DefaultBICConfig()
	if c.DeltaBICThreshold != -1.0 ||
		c.MinSplitObservations != 2 ||
		c.MaxSplitDepth != 3 ||
		c.MinChunkObservations != 2 ||
		c.LongRangeDeltaBICThreshold != -5.0 ||
		c.LongRangeMinSplitObservations != 5 ||
		c.LongRangeMinDominantFraction != 0.6 ||
		c.CrossDimMaxIterations != 5 ||
		c.CrossDimMinRuleCount != 3 ||
		c.CrossDimMinRuleConfidence != 0.5 ||
		c.MultiLectBICSmallSampleCorrection != true ||
		c.MultiLectMinCommitScale != 0.5 {
		t.Errorf("default config does not match historical values: %+v", c)
	}
	if err := c.Validate(); err != nil {
		t.Errorf("default config should validate, got %v", err)
	}
}

func TestInvalidBICConfigValuesRejected(t *testing.T) {
	base := DefaultBICConfig()
	mutate := []func(*BICConfig){
		func(c *BICConfig) { c.MinSplitObservations = 0 },
		func(c *BICConfig) { c.MaxSplitDepth = 0 },
		func(c *BICConfig) { c.MinChunkObservations = 0 },
		func(c *BICConfig) { c.LongRangeMinSplitObservations = 0 },
		func(c *BICConfig) { c.LongRangeMinDominantFraction = -0.1 },
		func(c *BICConfig) { c.LongRangeMinDominantFraction = 1.1 },
		func(c *BICConfig) { c.CrossDimMaxIterations = 0 },
		func(c *BICConfig) { c.CrossDimMinRuleCount = 0 },
		func(c *BICConfig) { c.CrossDimMinRuleConfidence = -0.1 },
		func(c *BICConfig) { c.CrossDimMinRuleConfidence = 1.1 },
		func(c *BICConfig) { c.MultiLectMinCommitScale = -0.1 },
	}
	for i, m := range mutate {
		c := base
		m(&c)
		if err := c.Validate(); err == nil {
			t.Errorf("mutation %d: expected validation error, got nil", i)
		}
	}
}
