"""Alignment search via dynamic programming.

Given two forms, find the lowest-cost ``Alignment`` — a sequence of
``Link`` objects that exhaustively covers both forms.

The search is a dynamic program over pairs of positions in the two
forms. At each cell ``(i, j)``, ``cost[i][j]`` holds the minimum total
cost to align ``source[:i]`` against ``target[:j]``. Transitions allow:

* 1-to-1 links (the diagonal move)
* 1-to-0 links (deletion — the down move)
* 0-to-1 links (insertion — the right move)
* N-to-M links (multi-segment chunks) up to a configurable maximum
  chunk size, scored compositionally or via the learned phrase table

The result is the single minimum-cost alignment. Ties are broken by
preferring smaller link sizes, biasing toward compositional
interpretations over chunk promotion.

When a trained model is supplied, scoring uses the learned segment
table (with context-conditioned entries), the displacement
distribution, the chunk phrase table, and cross-dimensional links.
Without a model, scoring falls back to raw merkmal distances.

Each link in the returned alignment carries its conditioning context
(preceding and following features, word position, long-range
predicates), computed from the form during the DP.

Complexity: for forms of length ``n`` and ``m`` with maximum chunk
size ``C``, the search examines at most ``(n + 1) * (m + 1) * C**2``
candidate transitions. For phonology-scale inputs (n, m up to a few
dozen; C <= 3) this is fast enough for interactive use.
"""

import merkmal

from regulae.model import LearnedModel
from regulae.scoring import (
    apply_cross_dimensional_adjustments,
    compute_displacement,
    score_link,
)
from regulae.types import (
    Alignment,
    Context,
    FeatureConstraint,
    Form,
    Link,
    Segment,
)

# Small, hand-picked inventory of features used when computing a link's
# conditioning context. Larger inventories would be more expressive but
# slower at the DP inner loop; the experiments showed that this set
# covers the real conditioning dimensions (vowel, height, frontness,
# voicing, sonorant, manner).
_CONTEXT_FEATURES: frozenset[str] = frozenset(
    (
        "vowel",
        "consonant",
        "front",
        "back",
        "close",
        "open",
        "voiced",
        "voiceless",
        "stop",
        "fricative",
        "nasal",
        "sonorant",
        "long",
    )
)

# Default maximum chunk size on either side of a link.
# C=3 covers almost all real phonological clusters. Set higher if you
# know you have genuinely long fused correspondences.
DEFAULT_MAX_CHUNK_SIZE: int = 3

# Per-step tie-breaking penalty added to every DP transition to
# enforce a deterministic cost-based preference among alignments
# that would otherwise tie exactly. Formula per link:
#
#     penalty = _CHUNK_COMPLEXITY_PENALTY * (k + l - 2)
#
# so a 1-to-1 link pays 0, a 2-to-1 or 1-to-2 link pays 1 unit,
# a 2-to-2 pays 2 units, a 3-to-3 pays 4, and so on. A gap link
# (k=0 or l=0) pays (k+l-2) which can be -1 — that's fine,
# insertions/deletions are already penalized elsewhere via
# DEFAULT_GAP_COST and we don't want the tiebreaker to incentivize
# them.
#
# The subtraction by 2 matters: without it, ``(k + l) * penalty``
# would pay ``(n + m) * penalty`` on every complete alignment
# regardless of how chunks are decomposed — total source and
# target length are fixed. Using ``(k + l - 2)`` per link only
# charges the "excess" beyond a 1-to-1 link, so decompositions
# into fewer, larger chunks pay strictly more than decompositions
# into more 1-to-1 links, breaking ties in favor of atomic
# alignments.
#
# The constant is 1e-9 so the penalty is ~20 orders of magnitude
# below any real alignment cost (which sits in the 0-10 range in
# typical use). No real cost comparison gets flipped by this term;
# it ONLY decides among exact-tied alternatives.
#
# Without this, the choice between two tied alignments depends on
# the (k, l) iteration order of the DP, which is deterministic but
# opaque: a future reorder of the inner loops would silently
# change picked alignments for tied cases. With the explicit
# penalty the preference is "smallest total chunk excess wins"
# and that preference is transparent in the arithmetic.
_CHUNK_COMPLEXITY_PENALTY: float = 1e-9


def align_forms(
    source: Form,
    target: Form,
    *,
    feature_system: str = "descriptive",
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    model: LearnedModel | None = None,
) -> Alignment:
    """Find the lowest-cost alignment between two forms.

    The returned ``Alignment`` holds a tuple of ``Link`` s whose source
    chunks concatenate to ``source.segments`` and whose target chunks
    concatenate to ``target.segments`` (the coverage invariant).

    For 1-to-1 links, the feature displacement vector is populated via
    ``compute_displacement``. For multi-segment links it is left empty
    — computing a displacement over a chunk requires a compositional
    interpretation that is not yet provided.

    Ties in the DP are broken by iteration order, which prefers smaller
    chunk sizes on each side.

    ``max_chunk_size`` must be at least 1. With ``max_chunk_size = 1``
    the search reduces to classical Needleman-Wunsch over the
    prior-only scoring.

    If ``model`` is provided, the search uses the layered learned
    scoring (segment-correspondence posterior + displacement
    distribution + chunk phrase table). When ``model`` is ``None``,
    the prior-only behavior is preserved exactly.
    """
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be at least 1")

    # When a model is provided, its feature system takes precedence over
    # the keyword argument to ensure consistency between prior,
    # displacement computation, and scoring.
    if model is not None:
        feature_system = model.feature_system

    n = len(source.segments)
    m = len(target.segments)

    # Precompute per-position feature constraints for the source form.
    # Each entry is the tuple of FeatureConstraint that a link's
    # ``preceding`` or ``following`` slot would get if it abutted that
    # position. Having this array lets context construction at each
    # DP cell be O(1) dict-and-tuple work, rather than an O(features)
    # merkmal lookup.
    source_features: list[tuple[FeatureConstraint, ...]]
    # Long-range auxiliary arrays. All empty when no model.
    left_cum: list[tuple[FeatureConstraint, ...]] = []
    right_cum: list[tuple[FeatureConstraint, ...]] = []
    syl_of: list[int] = []
    syl_starts: list[int] = []
    syl_features: list[tuple[FeatureConstraint, ...]] = []
    same_syl_excl: list[tuple[FeatureConstraint, ...]] = []
    if model is not None:
        source_features = [
            _features_for_neighbor(source, idx, feature_system)
            for idx in range(n)
        ]
        (
            left_cum,
            right_cum,
            syl_of,
            syl_starts,
            syl_features,
            same_syl_excl,
        ) = _compute_long_range_data(source, source_features, feature_system)
    else:
        source_features = []

    # cost[i][j] = minimum cost to align source[:i] with target[:j]
    cost: list[list[float]] = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0

    # back[i][j] = (prev_i, prev_j, link) that achieved cost[i][j], or None
    back: list[list[tuple[int, int, Link] | None]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            # Try every transition (k, l) that ends at (i, j).
            # k = source chunk length, l = target chunk length.
            # We iterate smaller chunk sizes first so ties prefer compact links.
            for k in range(min(max_chunk_size, i) + 1):
                for l in range(min(max_chunk_size, j) + 1):
                    if k == 0 and l == 0:
                        continue
                    prev = cost[i - k][j - l]
                    if prev == float("inf"):
                        continue

                    src_chunk = source.segments[i - k : i]
                    tgt_chunk = target.segments[j - l : j]

                    # Populate feature displacement only for 1-to-1 links.
                    displacement: tuple = ()
                    if k == 1 and l == 1:
                        displacement = compute_displacement(
                            src_chunk[0],
                            tgt_chunk[0],
                            feature_system=feature_system,
                        )

                    # Compute the link's conditioning context from the
                    # surrounding form segments. Empty when no model is
                    # provided (M2 behavior), since M2 doesn't use context.
                    if model is not None:
                        src_start = i - k
                        src_end = i
                        preceding = (
                            source_features[src_start - 1]
                            if src_start > 0 else ()
                        )
                        following = (
                            source_features[src_end]
                            if src_end < n else ()
                        )
                        if src_start == 0:
                            position: str | None = "initial"
                        elif src_end == n:
                            position = "final"
                        else:
                            position = "medial"
                        # Long-range fields. Distance offsets {2, 3}
                        # only (offset 1 is the immediate `preceding`/
                        # `following` slots above). Existential and
                        # syllable-structural fields use precomputed arrays.
                        # Syllable-structural fields are only populated
                        # for 1-to-1 links (k == l == 1) — chunks span
                        # multiple positions and the "same/next/previous
                        # syllable" notion gets ambiguous, so they are
                        # left empty.
                        pre_at: list[tuple[int, FeatureConstraint]] = []
                        fol_at: list[tuple[int, FeatureConstraint]] = []
                        for d in (2, 3):
                            li = src_start - d
                            if li >= 0:
                                for fc in source_features[li]:
                                    pre_at.append((d, fc))
                            ri = src_end + (d - 1)
                            if ri < n:
                                for fc in source_features[ri]:
                                    fol_at.append((d, fc))
                        somewhere_pre = left_cum[src_start] if src_start > 0 else ()
                        somewhere_fol = right_cum[src_end] if src_end < n else ()
                        same_syl: tuple[FeatureConstraint, ...] = ()
                        next_syl: tuple[FeatureConstraint, ...] = ()
                        prev_syl: tuple[FeatureConstraint, ...] = ()
                        if k == 1 and l == 1 and n > 0:
                            seg_idx = src_start
                            same_syl = same_syl_excl[seg_idx]
                            s_idx = syl_of[seg_idx]
                            if s_idx + 1 < len(syl_features):
                                next_syl = syl_features[s_idx + 1]
                            if s_idx > 0:
                                prev_syl = syl_features[s_idx - 1]
                        # Stress predicates are populated from
                        # ``Segment.stress`` (user-supplied), not
                        # merkmal features. Only 1-to-1 links carry
                        # these; chunks span multiple positions and
                        # the notion of "own stress" is ambiguous.
                        self_stress: tuple[FeatureConstraint, ...] = ()
                        preceding_stress: tuple[FeatureConstraint, ...] = ()
                        following_stress: tuple[FeatureConstraint, ...] = ()
                        if k == 1 and l == 1:
                            own = source.segments[src_start].stress
                            if own is not None:
                                self_stress = (FeatureConstraint("stress", own),)
                            if src_start > 0:
                                pre_s = source.segments[src_start - 1].stress
                                if pre_s is not None:
                                    preceding_stress = (FeatureConstraint("stress", pre_s),)
                            if src_end < n:
                                fol_s = source.segments[src_end].stress
                                if fol_s is not None:
                                    following_stress = (FeatureConstraint("stress", fol_s),)
                        link_context = Context(
                            position=position,
                            preceding=preceding,
                            following=following,
                            preceding_at_distance=tuple(pre_at),
                            following_at_distance=tuple(fol_at),
                            somewhere_preceding=somewhere_pre,
                            somewhere_following=somewhere_fol,
                            same_syllable=same_syl,
                            next_syllable=next_syl,
                            previous_syllable=prev_syl,
                            self_stress=self_stress,
                            preceding_stress=preceding_stress,
                            following_stress=following_stress,
                        )
                    else:
                        link_context = Context()

                    link = Link(
                        source_chunk=src_chunk,
                        target_chunk=tgt_chunk,
                        context=link_context,
                        feature_displacement=displacement,
                    )
                    link_cost = score_link(link, feature_system=feature_system, model=model)
                    # Add the chunk-complexity tie-breaker. See the
                    # _CHUNK_COMPLEXITY_PENALTY comment at module top:
                    # ``(k + l - 2)`` pays 0 for 1-to-1 links and
                    # increases monotonically with chunk size, so
                    # on exactly-tied substantive costs the DP
                    # picks the decomposition with fewer/smaller
                    # multi-segment chunks.
                    total = prev + link_cost + _CHUNK_COMPLEXITY_PENALTY * (k + l - 2)

                    if total < cost[i][j]:
                        cost[i][j] = total
                        back[i][j] = (i - k, j - l, link)

    # Traceback from (n, m) to (0, 0), collecting links in reverse order.
    links: list[Link] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = back[i][j]
        if step is None:
            # Only possible if both forms are empty, handled below.
            break
        pi, pj, link = step
        links.append(link)
        i, j = pi, pj
    links.reverse()

    return Alignment(source_form=source, target_form=target, links=tuple(links))


def _compute_link_context(
    source_form: Form,
    target_form: Form,
    src_start: int,
    src_end: int,
    tgt_start: int,
    tgt_end: int,
    feature_system: str,
) -> Context:
    """Build a ``Context`` from the segments surrounding the proposed link.

    * ``preceding`` reflects the features of the source segment at
      ``src_start - 1`` (empty tuple if the link starts at the form edge).
    * ``following`` reflects the features of the source segment at
      ``src_end`` (empty tuple if the link ends at the form edge).
    * ``position`` is ``"initial"`` when the link starts at index 0,
      ``"final"`` when it ends at the last segment, ``"medial"``
      otherwise.

    Also populates the long-range fields
    (``preceding_at_distance``, ``following_at_distance``,
    ``somewhere_preceding``, ``somewhere_following``,
    ``same_syllable``, ``next_syllable``, ``previous_syllable``).
    Syllable-structural fields are only populated for 1-segment
    source spans.
    """
    n = len(source_form.segments)
    preceding = _features_for_neighbor(source_form, src_start - 1, feature_system)
    following = _features_for_neighbor(source_form, src_end, feature_system)
    if src_start == 0:
        position: str | None = "initial"
    elif src_end == n:
        position = "final"
    else:
        position = "medial"
    source_features = [
        _features_for_neighbor(source_form, idx, feature_system)
        for idx in range(n)
    ]
    (
        left_cum,
        right_cum,
        syl_of,
        _syl_starts,
        syl_features,
        same_syl_excl,
    ) = _compute_long_range_data(source_form, source_features, feature_system)
    pre_at: list[tuple[int, FeatureConstraint]] = []
    fol_at: list[tuple[int, FeatureConstraint]] = []
    for d in (2, 3):
        li = src_start - d
        if li >= 0:
            for fc in source_features[li]:
                pre_at.append((d, fc))
        ri = src_end + (d - 1)
        if ri < n:
            for fc in source_features[ri]:
                fol_at.append((d, fc))
    somewhere_pre = left_cum[src_start] if src_start > 0 else ()
    somewhere_fol = right_cum[src_end] if src_end < n else ()
    same_syl: tuple[FeatureConstraint, ...] = ()
    next_syl: tuple[FeatureConstraint, ...] = ()
    prev_syl: tuple[FeatureConstraint, ...] = ()
    if (src_end - src_start) == 1 and n > 0:
        seg_idx = src_start
        same_syl = same_syl_excl[seg_idx]
        s_idx = syl_of[seg_idx]
        if s_idx + 1 < len(syl_features):
            next_syl = syl_features[s_idx + 1]
        if s_idx > 0:
            prev_syl = syl_features[s_idx - 1]
    self_stress: tuple[FeatureConstraint, ...] = ()
    preceding_stress: tuple[FeatureConstraint, ...] = ()
    following_stress: tuple[FeatureConstraint, ...] = ()
    if (src_end - src_start) == 1:
        own = source_form.segments[src_start].stress
        if own is not None:
            self_stress = (FeatureConstraint("stress", own),)
        if src_start > 0:
            pre_s = source_form.segments[src_start - 1].stress
            if pre_s is not None:
                preceding_stress = (FeatureConstraint("stress", pre_s),)
        if src_end < n:
            fol_s = source_form.segments[src_end].stress
            if fol_s is not None:
                following_stress = (FeatureConstraint("stress", fol_s),)
    return Context(
        position=position,
        preceding=preceding,
        following=following,
        preceding_at_distance=tuple(pre_at),
        following_at_distance=tuple(fol_at),
        somewhere_preceding=somewhere_pre,
        somewhere_following=somewhere_fol,
        same_syllable=same_syl,
        next_syllable=next_syl,
        previous_syllable=prev_syl,
        self_stress=self_stress,
        preceding_stress=preceding_stress,
        following_stress=following_stress,
    )


def _compute_long_range_data(
    form: Form,
    source_features: list[tuple[FeatureConstraint, ...]],
    feature_system: str,
) -> tuple[
    list[tuple[FeatureConstraint, ...]],
    list[tuple[FeatureConstraint, ...]],
    list[int],
    list[int],
    list[tuple[FeatureConstraint, ...]],
    list[tuple[FeatureConstraint, ...]],
]:
    """Precompute auxiliary arrays for long-range context construction.

    Returns ``(left_cum, right_cum, syl_of, syl_starts, syl_features,
    same_syl_excl)``:

    * ``left_cum[i]`` — sorted unique union of feature constraints over
      segments ``[0, i)``. Index 0 is empty.
    * ``right_cum[i]`` — sorted unique union over segments ``[i, n)``.
      Index ``n`` is empty.
    * ``syl_of[i]`` — syllable index of segment ``i``.
    * ``syl_starts[s]`` — starting segment index of syllable ``s``;
      sentinel ``n`` at the end so syllable ``s`` spans
      ``[syl_starts[s], syl_starts[s+1])``.
    * ``syl_features[s]`` — sorted unique union over segments in
      syllable ``s``.
    * ``same_syl_excl[i]`` — sorted unique union over segments in the
      same syllable as ``i``, excluding ``i`` itself.

    Sorting ensures a deterministic context representation across
    runs (constraints used as table keys must be reproducible).
    """
    from regulae.syllabification import compute_syllable_breaks

    n = len(form.segments)
    sort_key = lambda fc: (fc.feature, fc.value)  # noqa: E731

    left_cum: list[tuple[FeatureConstraint, ...]] = [()]
    seen_l: set[FeatureConstraint] = set()
    for i in range(n):
        for fc in source_features[i]:
            seen_l.add(fc)
        left_cum.append(tuple(sorted(seen_l, key=sort_key)))

    right_cum: list[tuple[FeatureConstraint, ...]] = [()] * (n + 1)
    seen_r: set[FeatureConstraint] = set()
    for i in range(n - 1, -1, -1):
        for fc in source_features[i]:
            seen_r.add(fc)
        right_cum[i] = tuple(sorted(seen_r, key=sort_key))

    if n == 0:
        return left_cum, right_cum, [], [0], [], []

    breaks = compute_syllable_breaks(form, feature_system=feature_system)
    syl_starts: list[int] = [0, *breaks, n]
    num_syl = len(syl_starts) - 1
    syl_of = [0] * n
    s = 0
    for i in range(n):
        while s + 1 < num_syl and i >= syl_starts[s + 1]:
            s += 1
        syl_of[i] = s

    syl_features: list[tuple[FeatureConstraint, ...]] = []
    for s in range(num_syl):
        seen: set[FeatureConstraint] = set()
        for i in range(syl_starts[s], syl_starts[s + 1]):
            for fc in source_features[i]:
                seen.add(fc)
        syl_features.append(tuple(sorted(seen, key=sort_key)))

    same_syl_excl: list[tuple[FeatureConstraint, ...]] = []
    for i in range(n):
        s = syl_of[i]
        seen = set()
        for j in range(syl_starts[s], syl_starts[s + 1]):
            if j != i:
                for fc in source_features[j]:
                    seen.add(fc)
        same_syl_excl.append(tuple(sorted(seen, key=sort_key)))

    return left_cum, right_cum, syl_of, syl_starts, syl_features, same_syl_excl


# Cache merkmal feature lookups keyed by (grapheme, feature_system).
# The inner DP loop calls this hundreds of times per alignment and
# merkmal lookups are not trivially cheap.
_NEIGHBOR_FEATURE_CACHE: dict[tuple[str, str], tuple[FeatureConstraint, ...]] = {}


def _features_for_neighbor(
    form: Form,
    index: int,
    feature_system: str,
) -> tuple[FeatureConstraint, ...]:
    """Extract the context-relevant features of the segment at ``index``.

    Returns an empty tuple if the index is out of bounds (link at form
    edge) or if the segment's grapheme is unknown to the feature system.
    Only features in ``_CONTEXT_FEATURES`` are emitted; the rest would
    balloon the context space without adding discriminative power for
    context splits. Results are cached by ``(grapheme, feature_system)``.
    """
    if index < 0 or index >= len(form.segments):
        return ()
    grapheme = form.segments[index].grapheme
    cache_key = (grapheme, feature_system)
    cached = _NEIGHBOR_FEATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        feats = merkmal.get_features(grapheme, system=feature_system)
    except KeyError:
        feats = None
    if feats is None:
        result: tuple[FeatureConstraint, ...] = ()
    else:
        relevant = sorted(f for f in feats if f in _CONTEXT_FEATURES)
        result = tuple(FeatureConstraint(feature=f, value="+") for f in relevant)
    _NEIGHBOR_FEATURE_CACHE[cache_key] = result
    return result


def alignment_cost(
    alignment: Alignment,
    feature_system: str = "descriptive",
    *,
    model: LearnedModel | None = None,
) -> float:
    """Total cost of an alignment: the sum of its link costs under
    the given feature system and (optional) learned model.

    The feature system should match the one used to produce the
    alignment. Scoring an alignment under a different feature system
    is meaningful only as a robustness check.

    If ``model`` is provided, its feature system takes precedence and
    the layered learned scoring applies. An additional additive term
    is applied via
    :func:`regulae.scoring.apply_cross_dimensional_adjustments` that
    accounts for committed cross-dimensional rules in
    ``model.cross_dimensional_table``. The adjustment is zero when
    the table is empty.
    """
    if model is not None:
        feature_system = model.feature_system
    base = sum(
        score_link(link, feature_system=feature_system, model=model)
        for link in alignment.links
    )
    if model is not None:
        base += apply_cross_dimensional_adjustments(alignment, model)
    return base
