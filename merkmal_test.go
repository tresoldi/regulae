package regulae

import "testing"

func TestMerkmalBridgeSmoke(t *testing.T) {
	feats := getFeatures("p", "descriptive")
	if feats == nil {
		t.Fatal(`getFeatures("p", "descriptive") returned nil; expected a known grapheme`)
	}
	if len(feats) == 0 {
		t.Fatal(`getFeatures("p") returned an empty feature set`)
	}

	if got := getFeatures("QQZZ", "descriptive"); got != nil {
		t.Errorf("expected nil features for an unknown grapheme, got %v", got)
	}

	if d := segmentDistance("p", "p", "descriptive"); d != 0 {
		t.Errorf("distance(p, p) = %v, want 0", d)
	}
	dpb := segmentDistance("p", "b", "descriptive")
	if dpb <= 0 || dpb > 1 {
		t.Errorf("distance(p, b) = %v, want a value in (0, 1]", dpb)
	}
}
