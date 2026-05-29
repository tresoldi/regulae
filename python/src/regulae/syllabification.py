"""Minimal sonority-based syllabification.

Computes syllable break indices from segment sonority using the
maximum onset principle under the sonority sequencing principle
(SSP). Deliberately language-agnostic and simple: users with
language-specific phonotactics should populate
``Form.syllable_breaks`` externally, and this module respects any
such pre-populated value.

Provides the syllable-structural infrastructure used by long-range
context predicates.

Algorithm summary:

1. Each segment gets a sonority score from its merkmal features:
   stop/affricate = 1, fricative = 2, nasal = 3, liquid = 4,
   glide = 5, vowel = 6. Unknown graphemes fall back to 3.
   Tone-only segments (empty grapheme) are skipped.
2. Syllable nuclei are all vowels (sonority 6) plus sonority
   peaks (segments strictly greater than both neighbours with
   sonority ≥ 4 — captures syllabic liquids).
3. Between two consecutive nuclei, the syllable break is placed
   at the leftmost position whose onset cluster toward the
   right nucleus has non-decreasing sonority. This is the
   maximum onset principle + SSP.
4. Word-initial and word-final consonant clusters are absorbed
   into the adjacent syllable regardless of SSP violations
   (there is no alternative home for them).

Known limitations (v1, deliberately not handled):

- **Diphthongs**: two adjacent vowels are always treated as
  separate nuclei. ``kapakau`` syllabifies as ``ka.pa.ka.u``
  rather than ``ka.pa.kau``. A human analysis of Romance or
  Austronesian phonology would group ``au`` as a single
  diphthong nucleus. Users who need diphthong-aware
  syllabification should pre-populate ``Form.syllable_breaks``
  externally.
- **Language-specific phonotactics**: the algorithm is
  universal and doesn't know which clusters are legal in a
  given language. Max-onset under SSP is a common default but
  not always the right analysis.
- **Syllabic nasals**: a nasal (sonority 3) between two stops
  does not qualify as a sonority peak under the liquid-or-
  higher rule, so ``bdn̩`` is NOT split as expected. Users with
  syllabic-nasal languages should pre-populate
  ``syllable_breaks``.
- **Ambisyllabicity**: every segment belongs to exactly one
  syllable. Ambisyllabic consonants (common in English
  prosodic analysis) are not representable in the flat
  break-index data model.
"""

import merkmal

from regulae.types import Form, Segment


SONORITY_STOP: int = 1
SONORITY_FRICATIVE: int = 2
SONORITY_NASAL: int = 3
SONORITY_LIQUID: int = 4
SONORITY_GLIDE: int = 5
SONORITY_VOWEL: int = 6
SONORITY_UNKNOWN: int = 3  # neutral middle value

# Per-process cache of (grapheme, feature_system) -> sonority score.
# Module-global so repeated calls don't re-query merkmal.
_SONORITY_CACHE: dict[tuple[str, str], int] = {}

# Per-form cache of computed syllable breaks, keyed by
# (form segments tuple, feature_system). Keeps repeated calls
# during training cheap.
_BREAKS_CACHE: dict[tuple[tuple[Segment, ...], str], tuple[int, ...]] = {}


def _sonority_from_features(features: frozenset[str] | None) -> int:
    """Convert a merkmal feature set to a sonority score.

    Applies the hierarchy stop < fricative < nasal < liquid <
    glide < vowel. Non-lateral approximants (j, w) are
    classified as glides; lateral approximants (l) and trills
    (r) are classified as liquids.
    """
    if features is None:
        return SONORITY_UNKNOWN
    if "vowel" in features:
        return SONORITY_VOWEL
    if "approximant" in features:
        if "lateral" in features:
            return SONORITY_LIQUID
        return SONORITY_GLIDE
    if "trill" in features or "tap" in features:
        return SONORITY_LIQUID
    if "nasal" in features:
        return SONORITY_NASAL
    if "fricative" in features:
        return SONORITY_FRICATIVE
    if "stop" in features or "affricate" in features:
        return SONORITY_STOP
    return SONORITY_UNKNOWN


def _sonority(segment: Segment, feature_system: str) -> int:
    """Sonority score for a segment. Returns ``-1`` for tone-only
    segments (empty grapheme) so callers can skip them."""
    grapheme = segment.grapheme
    if not grapheme:
        return -1
    cache_key = (grapheme, feature_system)
    cached = _SONORITY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        features = merkmal.get_features(grapheme, system=feature_system)
    except KeyError:
        features = None
    score = _sonority_from_features(features)
    _SONORITY_CACHE[cache_key] = score
    return score


def _find_nuclei(scores: list[int]) -> list[int]:
    """Return sorted positions of syllable nuclei.

    Any segment whose sonority equals :data:`SONORITY_VOWEL` is a
    nucleus. Additionally, any segment that is a **strict sonority
    peak** (strictly greater than both neighbours) with sonority
    ≥ :data:`SONORITY_LIQUID` is a nucleus — this catches syllabic
    liquids between consonants.

    Tone-only positions (``score < 0``) are skipped entirely.

    If no nuclei are found by the rules above, the single
    highest-sonority position in the sequence becomes the default
    nucleus. This guarantees every non-empty form has at least
    one syllable.
    """
    nuclei: list[int] = []
    n = len(scores)
    for i in range(n):
        s = scores[i]
        if s < 0:
            continue
        if s >= SONORITY_VOWEL:
            nuclei.append(i)
            continue
        if s >= SONORITY_LIQUID:
            left_less = i == 0 or scores[i - 1] < 0 or scores[i - 1] < s
            right_less = i == n - 1 or scores[i + 1] < 0 or scores[i + 1] < s
            if left_less and right_less:
                nuclei.append(i)

    if not nuclei:
        # Fallback: word has no vowels and no liquid peaks. Pick
        # the absolute sonority maximum as the single nucleus so
        # we don't return a syllabification with zero syllables.
        best: int = -1
        best_score: int = -1
        for i, s in enumerate(scores):
            if s < 0:
                continue
            if s > best_score:
                best_score = s
                best = i
        if best >= 0:
            nuclei.append(best)
    return nuclei


def _place_break_between(
    left_nucleus: int,
    right_nucleus: int,
    scores: list[int],
) -> int:
    """Return the syllable-break position between two nuclei.

    Applies the maximum onset principle subject to the sonority
    sequencing principle (SSP): the onset cluster of the right
    syllable must have non-decreasing sonority from its leftmost
    segment toward the nucleus. The break is placed as far left
    as possible consistent with that constraint.

    Tone-only positions between the nuclei are skipped; they
    don't affect the break position.
    """
    break_pos = right_nucleus  # default: no onset
    last_included_score: int | None = None
    for j in range(right_nucleus - 1, left_nucleus, -1):
        s = scores[j]
        if s < 0:
            # Tone-only position: skip without affecting SSP.
            continue
        if last_included_score is None:
            # The consonant immediately left of the nucleus always
            # joins its onset.
            last_included_score = s
            break_pos = j
        else:
            # SSP check: the sonority sequence from the new left
            # position to the nucleus must be non-decreasing. That
            # means any consonant we're adding to the left of the
            # current onset must have sonority ≤ the consonant
            # immediately to its right.
            if s <= last_included_score:
                last_included_score = s
                break_pos = j
            else:
                break
    return break_pos


def compute_syllable_breaks(
    form: Form,
    *,
    feature_system: str = "descriptive",
) -> tuple[int, ...]:
    """Compute syllable break indices for a form.

    Returns a tuple of positions where new syllables start. The
    first syllable is implied to start at index 0 and is therefore
    not listed: a return value of ``(3, 5)`` describes a form
    with three syllables occupying segments ``[0:3]``, ``[3:5]``,
    and ``[5:]``.

    Respects user overrides: if ``form.syllable_breaks`` is
    non-empty, that value is returned unchanged. This is the
    escape hatch for language-specific phonotactics — any caller
    with better knowledge than this sonority heuristic can
    pre-populate the field and the framework will use it
    instead.

    For forms with zero or one segment, returns an empty tuple.
    For forms whose segments are all tone-only or otherwise
    have no identifiable sonority, returns an empty tuple
    (single-syllable fallback).
    """
    if form.syllable_breaks:
        return form.syllable_breaks

    segments = form.segments
    n = len(segments)
    if n <= 1:
        return ()

    cache_key = (segments, feature_system)
    cached = _BREAKS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    scores = [_sonority(seg, feature_system) for seg in segments]
    nuclei = _find_nuclei(scores)
    if len(nuclei) <= 1:
        result: tuple[int, ...] = ()
    else:
        breaks: list[int] = []
        for idx in range(len(nuclei) - 1):
            breaks.append(
                _place_break_between(nuclei[idx], nuclei[idx + 1], scores)
            )
        result = tuple(breaks)

    _BREAKS_CACHE[cache_key] = result
    return result


def clear_caches() -> None:
    """Clear the module's internal caches.

    Intended for tests that want to exercise the cold-cache path
    or for long-running processes that need to reclaim memory.
    """
    _SONORITY_CACHE.clear()
    _BREAKS_CACHE.clear()
