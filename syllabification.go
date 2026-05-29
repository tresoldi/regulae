package regulae

import "sync"

// Minimal sonority-based syllabification via the maximum onset principle under
// the sonority sequencing principle (SSP). Deliberately language-agnostic;
// callers with language-specific phonotactics should populate
// Form.SyllableBreaks externally, and ComputeSyllableBreaks respects any such
// pre-populated value. Provides the syllable-structural infrastructure used by
// long-range context predicates.

const (
	sonorityStop      = 1
	sonorityFricative = 2
	sonorityNasal     = 3
	sonorityLiquid    = 4
	sonorityGlide     = 5
	sonorityVowel     = 6
	sonorityUnknown   = 3 // neutral middle value
)

type sonorityCacheKey struct {
	grapheme      string
	featureSystem string
}

type breaksCacheKey struct {
	segments      string
	featureSystem string
}

var (
	sonorityMu    sync.Mutex
	sonorityCache = map[sonorityCacheKey]int{}

	breaksMu    sync.Mutex
	breaksCache = map[breaksCacheKey][]int{}
)

// sonorityFromFeatures converts a merkmal feature set to a sonority score,
// applying stop < fricative < nasal < liquid < glide < vowel. Non-lateral
// approximants (j, w) are glides; lateral approximants (l) and trills/taps (r)
// are liquids. A nil feature set (unknown grapheme) scores neutral.
func sonorityFromFeatures(features featureSet) int {
	if features == nil {
		return sonorityUnknown
	}
	if features.has("vowel") {
		return sonorityVowel
	}
	if features.has("approximant") {
		if features.has("lateral") {
			return sonorityLiquid
		}
		return sonorityGlide
	}
	if features.has("trill") || features.has("tap") {
		return sonorityLiquid
	}
	if features.has("nasal") {
		return sonorityNasal
	}
	if features.has("fricative") {
		return sonorityFricative
	}
	if features.has("stop") || features.has("affricate") {
		return sonorityStop
	}
	return sonorityUnknown
}

// sonority returns the sonority score for a segment, or -1 for tone-only
// segments (empty grapheme) so callers can skip them.
func sonority(segment Segment, featureSystem string) int {
	grapheme := segment.Grapheme
	if grapheme == "" {
		return -1
	}
	key := sonorityCacheKey{grapheme: grapheme, featureSystem: featureSystem}
	sonorityMu.Lock()
	if cached, ok := sonorityCache[key]; ok {
		sonorityMu.Unlock()
		return cached
	}
	sonorityMu.Unlock()

	score := sonorityFromFeatures(getFeatures(grapheme, featureSystem))

	sonorityMu.Lock()
	sonorityCache[key] = score
	sonorityMu.Unlock()
	return score
}

// findNuclei returns sorted positions of syllable nuclei. Any vowel-sonority
// segment is a nucleus; additionally a strict sonority peak (strictly greater
// than both neighbours) with sonority >= liquid is a nucleus (syllabic
// liquids). Tone-only positions (score < 0) are skipped. If no nuclei are
// found, the single highest-sonority position becomes the default nucleus so
// every non-empty form has at least one syllable.
func findNuclei(scores []int) []int {
	nuclei := []int{}
	n := len(scores)
	for i := 0; i < n; i++ {
		s := scores[i]
		if s < 0 {
			continue
		}
		if s >= sonorityVowel {
			nuclei = append(nuclei, i)
			continue
		}
		if s >= sonorityLiquid {
			leftLess := i == 0 || scores[i-1] < 0 || scores[i-1] < s
			rightLess := i == n-1 || scores[i+1] < 0 || scores[i+1] < s
			if leftLess && rightLess {
				nuclei = append(nuclei, i)
			}
		}
	}

	if len(nuclei) == 0 {
		best := -1
		bestScore := -1
		for i, s := range scores {
			if s < 0 {
				continue
			}
			if s > bestScore {
				bestScore = s
				best = i
			}
		}
		if best >= 0 {
			nuclei = append(nuclei, best)
		}
	}
	return nuclei
}

// placeBreakBetween returns the syllable-break position between two nuclei,
// applying the maximum onset principle subject to the SSP. The onset cluster
// of the right syllable must have non-decreasing sonority from its leftmost
// segment toward the nucleus; the break is placed as far left as possible.
// Tone-only positions are skipped.
func placeBreakBetween(leftNucleus, rightNucleus int, scores []int) int {
	breakPos := rightNucleus // default: no onset
	lastIncluded := -1       // -1 sentinel for "none yet"
	haveLast := false
	for j := rightNucleus - 1; j > leftNucleus; j-- {
		s := scores[j]
		if s < 0 {
			continue
		}
		if !haveLast {
			lastIncluded = s
			haveLast = true
			breakPos = j
		} else if s <= lastIncluded {
			lastIncluded = s
			breakPos = j
		} else {
			break
		}
	}
	return breakPos
}

// ComputeSyllableBreaks computes syllable break indices for a form. The result
// is the set of positions where new syllables start; the first syllable is
// implied at index 0 and not listed (so (3, 5) describes three syllables over
// [0:3], [3:5], [5:]). If form.SyllableBreaks is non-empty it is returned
// unchanged (the escape hatch for language-specific phonotactics). Forms with
// 0 or 1 segments, or no identifiable sonority, return nil.
func ComputeSyllableBreaks(form Form, featureSystem string) []int {
	if featureSystem == "" {
		featureSystem = "descriptive"
	}
	if len(form.SyllableBreaks) > 0 {
		return form.SyllableBreaks
	}

	segments := form.Segments
	n := len(segments)
	if n <= 1 {
		return nil
	}

	cacheKey := breaksCacheKey{segments: segmentsKey(segments), featureSystem: featureSystem}
	breaksMu.Lock()
	if cached, ok := breaksCache[cacheKey]; ok {
		breaksMu.Unlock()
		return cached
	}
	breaksMu.Unlock()

	scores := make([]int, n)
	for i, seg := range segments {
		scores[i] = sonority(seg, featureSystem)
	}
	nuclei := findNuclei(scores)

	var result []int
	if len(nuclei) > 1 {
		breaks := make([]int, 0, len(nuclei)-1)
		for idx := 0; idx < len(nuclei)-1; idx++ {
			breaks = append(breaks, placeBreakBetween(nuclei[idx], nuclei[idx+1], scores))
		}
		result = breaks
	}

	breaksMu.Lock()
	breaksCache[cacheKey] = result
	breaksMu.Unlock()
	return result
}

// clearSyllabificationCaches clears the module's internal caches. Intended for
// tests that exercise the cold-cache path.
func clearSyllabificationCaches() {
	sonorityMu.Lock()
	sonorityCache = map[sonorityCacheKey]int{}
	sonorityMu.Unlock()
	breaksMu.Lock()
	breaksCache = map[breaksCacheKey][]int{}
	breaksMu.Unlock()
}
