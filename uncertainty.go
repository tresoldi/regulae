package regulae

import (
	"fmt"
	"math"
	"sort"
)

// Uncertainty quantification for count-bearing entries. Every count in a
// trained model is an estimate from finite observations; downstream consumers
// need an interval on each count. The interval is always on a rate in [0, 1] —
// the conditional probability the count represents.

// DefaultAlpha is the significance level for a 95% interval.
const DefaultAlpha = 0.05

// zByAlpha maps a significance level to its two-sided z-score. Hard-coded so
// there is no dependency on a statistics library.
var zByAlpha = map[float64]float64{
	0.10: 1.6448536269514722,
	0.05: 1.959963984540054,
	0.01: 2.5758293035489004,
}

// UncertaintyEstimate is a two-sided interval on a rate, plus provenance.
// Lo and Hi bracket the rate (a conditional probability in [0, 1]); the
// interval's width reflects how much data backs that rate. Method records how
// the interval was produced ("wilson" or "bootstrap"). N is the denominator
// used; Alpha is the significance level (0.05 = 95% CI).
type UncertaintyEstimate struct {
	Lo     float64
	Hi     float64
	Method string
	N      float64
	Alpha  float64
}

// supportedAlphas returns the sorted list of supported alpha values, for error
// messages.
func supportedAlphas() []float64 {
	out := make([]float64, 0, len(zByAlpha))
	for a := range zByAlpha {
		out = append(out, a)
	}
	sort.Float64s(out)
	return out
}

// WilsonInterval computes the Wilson score interval on the rate k/n. It
// handles n <= 0 by returning [0, 1], and accepts fractional k and n because
// confidence-weighted training produces non-integer counts. It panics on an
// unsupported alpha, which is a programmer error (mirrors Python's ValueError).
func WilsonInterval(k, n, alpha float64) UncertaintyEstimate {
	z, ok := zByAlpha[alpha]
	if !ok {
		panic(fmt.Sprintf("alpha %v not supported; expected one of %v", alpha, supportedAlphas()))
	}
	if n <= 0.0 {
		return UncertaintyEstimate{Lo: 0.0, Hi: 1.0, Method: "wilson", N: 0.0, Alpha: alpha}
	}
	pHat := math.Max(0.0, math.Min(1.0, k/n))
	denom := 1.0 + z*z/n
	center := (pHat + z*z/(2.0*n)) / denom
	inner := pHat*(1.0-pHat)/n + z*z/(4.0*n*n)
	half := (z / denom) * math.Sqrt(math.Max(inner, 0.0))
	lo := math.Max(0.0, center-half)
	hi := math.Min(1.0, center+half)
	return UncertaintyEstimate{Lo: lo, Hi: hi, Method: "wilson", N: n, Alpha: alpha}
}

// PercentileInterval computes a distribution-free percentile interval over a
// set of samples (each the rate or count from one resampled training). The
// returned Lo/Hi are the alpha/2 and 1-alpha/2 quantiles. Empty samples return
// [0, 1]. N records the denominator associated with the point estimate, not
// the sample size.
func PercentileInterval(samples []float64, n, alpha float64) UncertaintyEstimate {
	if len(samples) == 0 {
		return UncertaintyEstimate{Lo: 0.0, Hi: 1.0, Method: "bootstrap", N: n, Alpha: alpha}
	}
	ordered := append([]float64(nil), samples...)
	sort.Float64s(ordered)
	lo := quantile(ordered, alpha/2.0)
	hi := quantile(ordered, 1.0-alpha/2.0)
	return UncertaintyEstimate{Lo: lo, Hi: hi, Method: "bootstrap", N: n, Alpha: alpha}
}

// quantile computes a linear-interpolated quantile on a pre-sorted slice.
// q is in [0, 1].
func quantile(ordered []float64, q float64) float64 {
	if len(ordered) == 0 {
		return 0.0
	}
	if len(ordered) == 1 {
		return ordered[0]
	}
	if q <= 0.0 {
		return ordered[0]
	}
	if q >= 1.0 {
		return ordered[len(ordered)-1]
	}
	pos := q * float64(len(ordered)-1)
	loIdx := int(math.Floor(pos))
	hiIdx := int(math.Ceil(pos))
	if loIdx == hiIdx {
		return ordered[loIdx]
	}
	frac := pos - float64(loIdx)
	return ordered[loIdx]*(1.0-frac) + ordered[hiIdx]*frac
}

// bootstrapRateInterval computes a bootstrap percentile interval on a rate.
// samples is the distribution of rates (each computed within its own
// resample's denominator); n is the base denominator, stored for reference.
// Each sample is clamped to [0, 1]. Empty samples return [0, 1].
func bootstrapRateInterval(samples []float64, n, alpha float64) UncertaintyEstimate {
	if len(samples) == 0 {
		return UncertaintyEstimate{Lo: 0.0, Hi: 1.0, Method: "bootstrap", N: n, Alpha: alpha}
	}
	ordered := make([]float64, len(samples))
	for i, s := range samples {
		ordered[i] = math.Max(0.0, math.Min(1.0, s))
	}
	sort.Float64s(ordered)
	lo := quantile(ordered, alpha/2.0)
	hi := quantile(ordered, 1.0-alpha/2.0)
	return UncertaintyEstimate{Lo: lo, Hi: hi, Method: "bootstrap", N: n, Alpha: alpha}
}
