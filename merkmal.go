package regulae

import (
	"fmt"
	"sync"

	merkmal "github.com/tresoldi/merkmal/go"
)

// This file is the sole bridge to the merkmal feature library. The Python
// package reached merkmal through two module-level helpers,
// ``merkmal.get_features(grapheme, system=...)`` and
// ``merkmal.distance(a, b, system=...)``; everything else in regulae is built
// on top of those two. We reproduce exactly that surface here, backed by
// merkmal's bundled default registry, and cache the resolved systems so
// repeated lookups don't re-parse the embedded model files.

var (
	merkmalOnce     sync.Once
	merkmalRegistry *merkmal.Registry
	merkmalRegErr   error

	merkmalMu      sync.Mutex
	merkmalSystems = map[string]merkmal.System{}
)

// merkmalSystem resolves a named merkmal feature system from the bundled
// default registry, caching the result. An empty name falls back to the
// "descriptive" system, matching the Python default for Lect.feature_system.
func merkmalSystem(name string) (merkmal.System, error) {
	if name == "" {
		name = "descriptive"
	}
	merkmalOnce.Do(func() {
		merkmalRegistry, merkmalRegErr = merkmal.NewDefaultRegistry()
	})
	if merkmalRegErr != nil {
		return nil, fmt.Errorf("regulae: loading merkmal registry: %w", merkmalRegErr)
	}

	merkmalMu.Lock()
	defer merkmalMu.Unlock()
	if sys, ok := merkmalSystems[name]; ok {
		return sys, nil
	}
	sys, err := merkmalRegistry.Get(name)
	if err != nil {
		return nil, fmt.Errorf("regulae: resolving merkmal system %q: %w", name, err)
	}
	merkmalSystems[name] = sys
	return sys, nil
}

// featureSet is regulae's representation of a grapheme's phonological feature
// bundle: an unordered set of feature-name strings. It mirrors the
// ``frozenset[str]`` returned by ``merkmal.get_features``. A nil featureSet
// means the grapheme is unknown to the system (Python's ``None``); an empty
// non-nil set means a known grapheme with no features.
type featureSet map[string]struct{}

func (fs featureSet) has(feature string) bool {
	_, ok := fs[feature]
	return ok
}

// getFeatures returns the feature bundle for a grapheme under the named
// system, or nil if the grapheme is unknown (Python's get_features -> None).
func getFeatures(grapheme, system string) featureSet {
	sys, err := merkmalSystem(system)
	if err != nil {
		panic(err)
	}
	feats, ok := sys.GraphemeToFeatures(grapheme)
	if !ok {
		return nil
	}
	out := make(featureSet, len(feats))
	for f := range feats {
		out[f] = struct{}{}
	}
	return out
}

// segmentDistance returns merkmal's symmetric distance between two graphemes
// under the named system, reproducing ``merkmal.distance(a, b, system=...)``
// with default options.
func segmentDistance(a, b, system string) float64 {
	sys, err := merkmalSystem(system)
	if err != nil {
		panic(err)
	}
	return sys.SegmentDistance(a, b)
}
