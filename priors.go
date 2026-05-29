package regulae

// TypologicalPrior is a log-prior adjustment in nats on a directed
// segment-pair correspondence. Given (srcGrapheme, tgtGrapheme) it returns a
// value added to the merkmal feature-distance logit before the softmax that
// produces the Dirichlet pseudo-counts in the initial model:
//
//	logit(s, t) = -temperature * d(s, t) + prior(s, t)
//
// Positive returns boost the prior probability of t given s; negative suppress
// it; zero leaves the merkmal-distance prior unchanged.
//
// Implementations must be pure: identical inputs must produce identical
// outputs across calls so training stays deterministic.
//
// regulae ships no opinionated priors. The framework deliberately does not
// encode typological asymmetries (e.g. "lenition is more likely than
// fortition") as defaults — that knowledge belongs with the user.
type TypologicalPrior func(src, tgt string) float64

// Uniform returns the no-op prior: zero adjustment for every pair. It
// preserves merkmal-only behaviour and is the default for TrainModel.
func Uniform() TypologicalPrior {
	return func(_, _ string) float64 { return 0.0 }
}

// Combine sums multiple priors, returning a new prior that adds each
// contributor's adjustment. Useful for composing a sparse user table with a
// structural prior, or stacking family-level and pair-specific priors.
func Combine(priors ...TypologicalPrior) TypologicalPrior {
	return func(src, tgt string) float64 {
		total := 0.0
		for _, p := range priors {
			total += p(src, tgt)
		}
		return total
	}
}
