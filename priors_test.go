package regulae

import "testing"

func TestUniformPriorIsZero(t *testing.T) {
	p := Uniform()
	if got := p("p", "f"); got != 0.0 {
		t.Errorf("uniform prior = %v, want 0", got)
	}
}

func TestCombinePriorsSum(t *testing.T) {
	a := func(src, tgt string) float64 {
		if src == "p" && tgt == "f" {
			return 0.5
		}
		return 0.0
	}
	b := func(src, tgt string) float64 {
		if src == "p" && tgt == "f" {
			return 0.25
		}
		return 0.0
	}
	p := Combine(Uniform(), a, b)
	if got := p("p", "f"); got != 0.75 {
		t.Errorf("combined prior = %v, want 0.75", got)
	}
	if got := p("k", "x"); got != 0.0 {
		t.Errorf("combined prior on unmatched pair = %v, want 0", got)
	}
}
