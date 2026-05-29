package regulae

import (
	"math"
	"testing"
)

func TestWilsonZeroNFullInterval(t *testing.T) {
	u := WilsonInterval(0, 0, DefaultAlpha)
	if u.Lo != 0.0 || u.Hi != 1.0 {
		t.Errorf("got [%v, %v], want [0, 1]", u.Lo, u.Hi)
	}
	if u.Method != "wilson" {
		t.Errorf("method = %q, want wilson", u.Method)
	}
}

func TestWilsonFullCertaintyNarrowHigh(t *testing.T) {
	u := WilsonInterval(100, 100, DefaultAlpha)
	if u.Lo <= 0.95 {
		t.Errorf("lo = %v, want > 0.95", u.Lo)
	}
	if u.Hi != 1.0 {
		t.Errorf("hi = %v, want 1.0", u.Hi)
	}
}

func TestWilsonZeroSuccessesNarrowLow(t *testing.T) {
	u := WilsonInterval(0, 100, DefaultAlpha)
	if u.Lo >= 1e-10 {
		t.Errorf("lo = %v, want < 1e-10", u.Lo)
	}
	if u.Hi >= 0.05 {
		t.Errorf("hi = %v, want < 0.05", u.Hi)
	}
}

func TestWilsonHalfHalfCentered(t *testing.T) {
	u := WilsonInterval(50, 100, DefaultAlpha)
	if math.Abs((u.Lo+u.Hi)/2-0.5) >= 0.01 {
		t.Errorf("center = %v, want ~0.5", (u.Lo+u.Hi)/2)
	}
	if u.Hi-u.Lo >= 0.25 {
		t.Errorf("width = %v, want < 0.25", u.Hi-u.Lo)
	}
}

func TestWilsonWidensAsNShrinks(t *testing.T) {
	narrow := WilsonInterval(50, 100, DefaultAlpha)
	wide := WilsonInterval(5, 10, DefaultAlpha)
	if (wide.Hi - wide.Lo) <= (narrow.Hi - narrow.Lo) {
		t.Errorf("wide width %v not greater than narrow width %v", wide.Hi-wide.Lo, narrow.Hi-narrow.Lo)
	}
}

func TestWilsonFractionalCounts(t *testing.T) {
	u := WilsonInterval(2.5, 5.0, DefaultAlpha)
	if !(u.Lo > 0.0 && u.Lo < 0.3) {
		t.Errorf("lo = %v, want in (0, 0.3)", u.Lo)
	}
	if !(u.Hi > 0.7 && u.Hi < 1.0) {
		t.Errorf("hi = %v, want in (0.7, 1.0)", u.Hi)
	}
}

func TestWilsonRejectsUnsupportedAlpha(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic on unsupported alpha")
		}
	}()
	WilsonInterval(5, 10, 0.07)
}

func TestPercentileIntervalEmpty(t *testing.T) {
	u := PercentileInterval(nil, 0, DefaultAlpha)
	if u.Lo != 0.0 || u.Hi != 1.0 {
		t.Errorf("got [%v, %v], want [0, 1]", u.Lo, u.Hi)
	}
}

func TestPercentileIntervalBracketsMiddle(t *testing.T) {
	samples := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	u := PercentileInterval(samples, 0, DefaultAlpha)
	if u.Lo > 0.2 {
		t.Errorf("lo = %v, want <= 0.2", u.Lo)
	}
	if u.Hi < 0.8 {
		t.Errorf("hi = %v, want >= 0.8", u.Hi)
	}
}
