import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from regulae.priors import TypologicalPrior

from regulae.config import BICConfig
from regulae.model import (
    ChunkPhraseTable,
    CognateSet,
    LearnedModel,
    MultiLectCorrespondenceClass,
    MultiLectCrossDimensionalLink,
    MultiLectCrossDimensionalLinkTable,
    MultiLectModel,
)
from regulae.search import align_forms
from regulae.uncertainty import wilson_interval
from regulae.types import (
    Context,
    Form,
)
from regulae._discovery import (
    DELTA_BIC_THRESHOLD,
    MAX_SPLIT_DEPTH,
    MIN_SPLIT_OBSERVATIONS,
    _LONG_RANGE_DELTA_BIC_THRESHOLD,
    _LONG_RANGE_MIN_DOMINANT_FRACTION,
    _LONG_RANGE_MIN_SPLIT_OBS,
    _apply_predicate,
    _candidate_predicates,
    _group_cost,
    _long_range_candidate_predicates,
    _partition,
)


def _obs_weight(obs: tuple) -> float:
    """Return the weight of an observation tuple.

    Backward-compatible with older 2-tuples ``(target, Context)``
    used by internal helper tests; those implicitly carry weight 1.0.
    """
    if len(obs) >= 3:
        return float(obs[2])
    return 1.0


class _UnionFind:
    """Tiny union-find over hashable nodes, used only by the
    multi-lect reconciliation stage. Nodes are added implicitly
    on first use."""

    def __init__(self) -> None:
        self._parent: dict[object, object] = {}

    def find(self, x: object) -> object:
        if x not in self._parent:
            self._parent[x] = x
            return x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # type: ignore[index]
            x = self._parent[x]
        return x

    def union(self, a: object, b: object) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def components(self) -> dict[object, list[object]]:
        out: dict[object, list[object]] = {}
        for node in list(self._parent):
            root = self.find(node)
            out.setdefault(root, []).append(node)
        return out


def _collect_lect_ids(corpus: Sequence[CognateSet]) -> tuple[str, ...]:
    """Return the canonical list of lect IDs in first-seen order.

    Preserving insertion order (rather than sorting alphabetically)
    matters for backward compatibility: a pair corpus built via
    ``cognate_sets_from_pairs(pairs, ("A", "B"))`` should train its
    pair model in the A→B direction. Python dicts preserve insertion
    order, so the first cognate set dictates the canonical lect
    ordering.

    Determinism under corpus reordering is preserved for all
    downstream outputs because classes and reconciled tables are
    keyed by sorted segment tuples, not by lect-id order.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for cs in corpus:
        for lect_id in cs.forms:
            if lect_id not in seen_set:
                seen.append(lect_id)
                seen_set.add(lect_id)
    return tuple(seen)


def _subset_for_pair(
    corpus: Sequence[CognateSet], lect_a: str, lect_b: str
) -> tuple[list[tuple[Form, Form]], list[float]]:
    """Extract the pairwise (form_a, form_b) list for cognate sets
    where both lects are present.

    Propagates ``CognateSet.morpheme_boundaries`` onto each Form's
    ``morpheme_breaks`` field if the Form does not already have
    boundaries set. This means consumers can supply boundaries
    either on the Form (preferred) or on the CognateSet (legacy);
    both routes flow into chunk promotion.
    """
    out: list[tuple[Form, Form]] = []
    weights: list[float] = []
    for cs in corpus:
        if lect_a in cs.forms and lect_b in cs.forms:
            form_a = _form_with_boundaries(cs.forms[lect_a], cs, lect_a)
            form_b = _form_with_boundaries(cs.forms[lect_b], cs, lect_b)
            out.append((form_a, form_b))
            weights.append(cs.confidence)
    return out, weights


def _form_with_boundaries(form: Form, cs: CognateSet, lect_id: str) -> Form:
    """Return ``form`` with ``morpheme_breaks`` populated from
    ``cs.morpheme_boundaries[lect_id]`` if the form does not already
    carry breaks. No-op when neither source has boundary info."""
    if form.morpheme_breaks:
        return form
    if cs.morpheme_boundaries is None:
        return form
    bounds = cs.morpheme_boundaries.get(lect_id)
    if not bounds:
        return form
    return replace(form, morpheme_breaks=tuple(bounds))


def _pair_alignment_edges(
    form_a: Form,
    form_b: Form,
    pair_model: LearnedModel,
    lect_a: str,
    lect_b: str,
    *,
    max_chunk_size: int,
) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    """Return the position-level edges induced by aligning one pair.

    For each Link in the top-level alignment:

    * **Equal-length chunks** (including all 1-to-1 links) pair
      position-by-position. This is the common case and preserves
      well-formed classes even when the chunk-promotion stage
      produced a phrase-table entry — e.g., a ``(pat, fad)``
      chunk contributes ``p↔f, a↔a, t↔d`` as if the alignment
      had been three 1-to-1 links.

    * **Unequal-length chunks** (the ``(sk, ʃ)`` case from Old
      English being the canonical example) get decomposed via a
      sub-alignment: a recursive :func:`align_forms` call with
      ``max_chunk_size=1`` and an emptied chunk table so the
      sub-alignment cannot itself return chunks. This gives the
      best 1-to-1 decomposition of the chunk's internal pairing
      under the **same pair model's learned segment table**, and
      we then emit position edges from the sub-alignment.
    """
    alignment = align_forms(
        form_a, form_b, max_chunk_size=max_chunk_size, model=pair_model
    )
    # An empty-chunk-table copy of the model. Used for the
    # unequal-length sub-alignment below so it can't recurse back
    # into the phrase table. Built lazily only when actually needed.
    no_chunks_model: LearnedModel | None = None

    edges: list[tuple[tuple[str, int], tuple[str, int]]] = []
    a_pos = 0
    b_pos = 0
    for link in alignment.links:
        a_len = len(link.source_chunk)
        b_len = len(link.target_chunk)
        if a_len == b_len:
            for i in range(a_len):
                edges.append(((lect_a, a_pos + i), (lect_b, b_pos + i)))
        elif a_len == 0 or b_len == 0:
            # Pure insertion / deletion: no edges to emit. A lect
            # without a segment at this position just doesn't
            # participate in the corresponding class.
            pass
        else:
            # Unequal-length multi-segment chunk. Run a sub-alignment
            # over the chunk's segments, using the pair's learned
            # model stripped of its chunk table so recursion is
            # impossible. This follows the same pattern the
            # context-discovery stage uses to recover 1-to-1
            # equivalents from promoted chunks.
            if no_chunks_model is None:
                no_chunks_model = replace(
                    pair_model, chunk_table=ChunkPhraseTable()
                )
            sub_src = Form(lect_id=lect_a, segments=link.source_chunk)
            sub_tgt = Form(lect_id=lect_b, segments=link.target_chunk)
            sub = align_forms(
                sub_src, sub_tgt, model=no_chunks_model, max_chunk_size=1
            )
            sub_a = 0
            sub_b = 0
            for sub_link in sub.links:
                sa = len(sub_link.source_chunk)
                sb = len(sub_link.target_chunk)
                # With max_chunk_size=1 the sub-alignment only emits
                # 1-to-1, 0-to-1, and 1-to-0 links. 1-to-1 becomes
                # one edge; gap links contribute no edges at all
                # (the gapped side has no position in the outer
                # form, so nothing to union).
                if sa == 1 and sb == 1:
                    edges.append((
                        (lect_a, a_pos + sub_a),
                        (lect_b, b_pos + sub_b),
                    ))
                sub_a += sa
                sub_b += sb
        a_pos += a_len
        b_pos += b_len
    return edges


def _reconcile_cognate_set(
    cs: CognateSet,
    pairwise_models: Mapping[frozenset[str], LearnedModel],
    *,
    max_chunk_size: int,
) -> list[tuple[dict[str, str], dict[str, int]]]:
    """Reconcile one cognate set into a list of observation rows.

    Each observation row is ``(segments, positions)`` where
    ``segments`` maps ``lect_id -> grapheme`` for lects participating
    in this class, and ``positions`` maps ``lect_id -> pos`` (useful
    later for context discovery).

    Components with more than one position per lect (union-find
    inconsistency: different pairs disagreed about which positions to
    align) are skipped. They surface as gaps in the aggregated class
    counts rather than garbage classes.
    """
    import itertools

    uf = _UnionFind()
    # Seed every (lect, position) so singleton nodes are still tracked
    # — even positions that no pair aligns to should show up as a
    # single-lect observation (which is then filtered out as carrying
    # no multi-lect signal).
    for lect_id, form in cs.forms.items():
        for pos in range(len(form.segments)):
            uf.find((lect_id, pos))

    for lect_a, lect_b in itertools.combinations(sorted(cs.forms), 2):
        key = frozenset({lect_a, lect_b})
        if key not in pairwise_models:
            continue
        edges = _pair_alignment_edges(
            cs.forms[lect_a],
            cs.forms[lect_b],
            pairwise_models[key],
            lect_a,
            lect_b,
            max_chunk_size=max_chunk_size,
        )
        for u, v in edges:
            uf.union(u, v)

    observations: list[tuple[dict[str, str], dict[str, int]]] = []
    for component in uf.components().values():
        by_lect: dict[str, int] = {}
        inconsistent = False
        for lect_id, pos in component:  # type: ignore[misc]
            if lect_id in by_lect:
                inconsistent = True
                break
            by_lect[lect_id] = pos
        if inconsistent:
            continue
        if len(by_lect) < 2:
            continue
        segments = {
            lect_id: cs.forms[lect_id].segments[pos].grapheme
            for lect_id, pos in by_lect.items()
        }
        observations.append((segments, by_lect))
    return observations


def _aggregate_unconditioned_classes(
    observations: Sequence[tuple[str, dict[str, str], float]],
) -> tuple[MultiLectCorrespondenceClass, ...]:
    """Group observations by their segment tuple and produce
    unconditioned :class:`MultiLectCorrespondenceClass` entries.

    ``observations`` is a list of ``(cognate_id, segments_dict, weight)``.
    The key for grouping is the sorted tuple of ``(lect_id, grapheme)``
    pairs, which makes two observations with the same participating
    lects and the same graphemes fall into the same class.

    Classes are returned sorted by count (desc), then by key tuple
    (lex asc) for determinism (open question 4 in the spec).
    """
    buckets: dict[
        tuple[tuple[str, str], ...],
        tuple[float, list[str]],
    ] = {}
    for cog_id, segments, weight in observations:
        if weight <= 0.0:
            continue
        key = tuple(sorted(segments.items()))
        count, support = buckets.get(key, (0, []))
        buckets[key] = (count + weight, support + [cog_id])

    # Per participating-lect set, total observations across all
    # classes with that same set. Used as the Wilson denominator:
    # "within cognate sets where these lects all contribute,
    # what fraction shows this segment tuple?".
    participant_totals: dict[frozenset[str], float] = defaultdict(float)
    for key, (count, _support) in buckets.items():
        participant_totals[frozenset(lect for lect, _ in key)] += count

    classes: list[MultiLectCorrespondenceClass] = []
    for class_id, (key, (count, support)) in enumerate(
        sorted(
            buckets.items(),
            key=lambda kv: (-kv[1][0], kv[0]),
        )
    ):
        participants = frozenset(lect for lect, _ in key)
        n = participant_totals.get(participants, 0.0)
        classes.append(
            MultiLectCorrespondenceClass(
                class_id=class_id,
                segments=dict(key),
                contexts=None,
                count=count,
                supporting_cognates=tuple(support),
                uncertainty=wilson_interval(count, n),
            )
        )
    return tuple(classes)


def _compute_single_position_context(
    form: Form, pos: int, feature_system: str
) -> Context:
    """Return the phonological Context at one position in one form.

    Delegates to ``search._compute_link_context`` for a 1-segment
    window so the long-range fields populated at search time are
    also produced here. This keeps observation contexts
    consistent with link contexts, which is required for table
    lookups via ``Context.is_subset_of`` to find conditioned
    entries.
    """
    from regulae.search import _compute_link_context

    return _compute_link_context(
        form, form, pos, pos + 1, 0, 0, feature_system
    )


def _multi_lect_context_discovery(
    corpus: Sequence[CognateSet],
    observations: list[tuple[str, dict[str, str], dict[str, int], float]],
    feature_system: str,
    start_class_id: int,
    *,
    bic_small_sample_correction: bool = True,
    min_commit_scale: float = 1.0,
    bic_config: BICConfig | None = None,
) -> tuple[MultiLectCorrespondenceClass, ...]:
    """Multi-lect class-level context discovery.

    For each (pivot_lect, pivot_grapheme) that appears across
    multiple distinct sister tuples, group the observations and
    run the same sequential-greedy BIC-driven split loop that the
    per-pair context-discovery stage uses on segment
    correspondences. The "target" for the split is the tuple of
    other-lect graphemes at the same class observation; the
    "context" is the pivot lect's phonological neighbourhood at
    its observation position.

    Committed splits become conditioned
    :class:`MultiLectCorrespondenceClass` entries carrying
    ``contexts = {pivot_lect: split_context}`` — other lects in the
    class have empty ``Context()`` as their conditioning slot.
    """
    if not observations:
        return ()

    corpus_by_id = {cs.cognate_id: cs for cs in corpus}

    # For each observation, build (cognate_id, pivot_lect, pivot_grapheme,
    # sister_tuple_key, pivot_context).
    # sister_tuple_key is a canonical encoding of the sister (lect, grapheme)
    # pairs so that we can reuse the per-pair string-target split machinery.
    pivot_buckets: dict[
        tuple[str, str],
        list[tuple[str, str, Context, float]],
    ] = defaultdict(list)
    # Remember the sister tuple behind each string key so we can build
    # classes back out at the end.
    sister_by_key: dict[str, tuple[tuple[str, str], ...]] = {}
    # Track the cognate IDs supporting each committed (pivot, ctx, sisters)
    # triple.
    supporting: dict[
        tuple[str, str, Context, tuple[tuple[str, str], ...]], list[str]
    ] = defaultdict(list)

    for cog_id, segments, positions, obs_weight in observations:
        if obs_weight <= 0.0:
            continue
        cs = corpus_by_id[cog_id]
        for pivot_lect, pivot_g in segments.items():
            sister = tuple(
                sorted((l, g) for l, g in segments.items() if l != pivot_lect)
            )
            if not sister:
                continue
            sister_key = "|".join(f"{l}:{g}" for l, g in sister)
            sister_by_key[sister_key] = sister
            form = cs.forms[pivot_lect]
            pos = positions[pivot_lect]
            ctx = _compute_single_position_context(form, pos, feature_system)
            pivot_buckets[(pivot_lect, pivot_g)].append((sister_key, ctx, obs_weight))

    # Now run a per-pivot greedy split loop. For each pivot bucket
    # with multiple distinct sister targets and enough observations,
    # try context predicates and commit splits that beat BIC. Each
    # committed tuple also carries the pivot's total observation
    # count so we can later compute the coverage-based confidence
    # (count / pivot_bucket_size) of the resulting class.
    committed: list[
        tuple[str, str, Context, tuple[tuple[str, str], ...], float, float]
    ] = []  # (pivot_lect, pivot_g, ctx, sister_tuple, count, pivot_bucket_size)

    # Harvest every distinct stress value the pivot observations
    # carry, so stress-conditioned splits are candidate predicates
    # per pivot. Empty when no observation has a stress annotation.
    observed_stress_values: frozenset[str] = frozenset({
        fc.value
        for _, ctxs in pivot_buckets.items()
        for (_, ctx, _) in ctxs
        for slot in (ctx.self_stress, ctx.preceding_stress, ctx.following_stress)
        for fc in slot
        if fc.feature == "stress"
    })

    cfg = bic_config if bic_config is not None else BICConfig()
    for (pivot_lect, pivot_g), obs in pivot_buckets.items():
        target_set = {t for t, *_rest in obs}
        if len(target_set) < 2 or sum(_obs_weight(o) for o in obs) < 4.0:
            continue
        _commit_multi_lect_splits_for_pivot(
            pivot_lect=pivot_lect,
            pivot_grapheme=pivot_g,
            observations=obs,
            sister_by_key=sister_by_key,
            committed=committed,
            use_bic_small_sample_correction=bic_small_sample_correction,
            min_commit_scale=min_commit_scale,
            observed_stress_values=observed_stress_values,
            bic_config=cfg,
        )
        # Also try long-range predicates on the same pivot bucket.
        # Stricter thresholds (mirroring the per-pair
        # ``_long_range_discovery``) to keep noise out.
        _commit_multi_lect_long_range_splits_for_pivot(
            pivot_lect=pivot_lect,
            pivot_grapheme=pivot_g,
            observations=obs,
            sister_by_key=sister_by_key,
            committed=committed,
            min_commit_scale=min_commit_scale,
            bic_config=cfg,
        )

    # Build MultiLectCorrespondenceClass entries from committed splits,
    # deduplicating across pivot lects.
    #
    # Multiple pivots often discover the SAME rule from different
    # angles: e.g., pivot HAW:k discovers (Haw:k, Mao:t, ..., foll=back)
    # and pivot SAM:t also discovers the same segment tuple under its
    # own foll=back predicate. These are one rule, not six. Dedup
    # groups by full segment tuple; within a group the contexts from
    # different pivots are merged per-lect, keeping the most specific
    # non-empty context for each lect. Count is the maximum observed
    # across contributing commits (the strongest single pivot's view).
    merged: dict[
        tuple[tuple[str, str], ...],
        dict,
    ] = {}
    for pivot_lect, pivot_g, ctx, sister, count, pivot_bucket_size in committed:
        segments_map: dict[str, str] = {pivot_lect: pivot_g}
        for l, g in sister:
            segments_map[l] = g
        seg_key = tuple(sorted(segments_map.items()))
        # Per-pivot coverage: what fraction of the pivot's total
        # observations landed in this class? High coverage means
        # "nearly every time this pivot's grapheme appeared under
        # this context, this was the sister tuple", i.e. a strong
        # rule. Low coverage means "this class is a weak minority
        # of the pivot's observations".
        coverage = count / pivot_bucket_size if pivot_bucket_size > 0 else 0.0
        entry = merged.get(seg_key)
        if entry is None:
            entry = {
                "segments": segments_map,
                "contexts": {pivot_lect: ctx},
                "count": count,
                "confidence": coverage,
                # ``bucket_size`` is the ``n`` used as the Wilson
                # denominator. It tracks the pivot whose coverage
                # won — the clearest witness.
                "bucket_size": pivot_bucket_size,
                "winning_count": count,
            }
            merged[seg_key] = entry
        else:
            # Merge: update count to max, keep the most specific
            # (highest constraint_count) context per lect, and keep
            # the maximum coverage across contributing pivots. The
            # max is the right reduction for coverage: if any one
            # pivot saw this rule with high confidence, that pivot
            # is the clearest witness and its coverage is the
            # cleanest available signal. Averaging would dilute a
            # strong pivot with weaker ones.
            if count > entry["count"]:
                entry["count"] = count
            if coverage > entry["confidence"]:
                entry["confidence"] = coverage
                entry["bucket_size"] = pivot_bucket_size
                entry["winning_count"] = count
            existing_ctx = entry["contexts"].get(pivot_lect)
            if existing_ctx is None or ctx.constraint_count() > existing_ctx.constraint_count():
                entry["contexts"][pivot_lect] = ctx

    # Fill in empty contexts for any lect in the segment tuple that
    # had no specific constraint contributed.
    classes: list[MultiLectCorrespondenceClass] = []
    for seg_key, entry in merged.items():
        segments_map = dict(entry["segments"])
        contexts_map: dict[str, Context] = dict(entry["contexts"])
        for lect_id in segments_map:
            contexts_map.setdefault(lect_id, Context())
        classes.append(
            MultiLectCorrespondenceClass(
                class_id=0,  # reassigned after sort
                segments=segments_map,
                contexts=contexts_map,
                count=float(entry["count"]),
                confidence=float(entry["confidence"]),
                uncertainty=wilson_interval(
                    float(entry["winning_count"]),
                    float(entry["bucket_size"]),
                ),
            )
        )

    # Sort for determinism: by count desc, then by segment tuple lex asc.
    classes.sort(
        key=lambda k: (
            -k.count,
            sorted(k.segments.items()),
        )
    )
    classes = [replace(k, class_id=start_class_id + i) for i, k in enumerate(classes)]
    return tuple(classes)


def _multi_lect_min_commit_count(n_total: int, scale: float) -> int:
    """Adaptive floor for per-sister-tuple emissions in the
    multi-lect class-discovery loop.

    Scales sub-linearly with the pivot observation count::

        min = max(2, ceil(scale * log2(n + 1)))

    With the default ``scale=0.5``:

    - For n=4:  min = max(2, ceil(0.5*2.32)) = 2
    - For n=8:  min = max(2, ceil(0.5*3.17)) = 2
    - For n=16: min = max(2, ceil(0.5*4.09)) = 3
    - For n=64: min = max(2, ceil(0.5*6.02)) = 4
    - For n=256: min = max(2, ceil(0.5*8.01)) = 5

    The ``scale`` knob lets callers tighten (``>1.0``) or relax
    (``<1.0``) the floor. A scale of 0 reverts to the structural
    minimum of 2 (no adaptation). The default was tuned on the
    51-set GLED Romance run and the 44-set GLED Polynesian run:
    scale=1.0 killed legitimate count-2 commits, scale=0
    preserved count-1 noise, scale=0.5 is the sweet spot.
    """
    if scale <= 0.0:
        return 2
    import math as _math
    return max(2, _math.ceil(scale * _math.log2(n_total + 1)))


def _commit_multi_lect_splits_for_pivot(
    pivot_lect: str,
    pivot_grapheme: str,
    observations: list[tuple[str, Context, float]],
    sister_by_key: dict[str, tuple[tuple[str, str], ...]],
    committed: list[
        tuple[str, str, Context, tuple[tuple[str, str], ...], float, float]
    ],
    *,
    use_bic_small_sample_correction: bool,
    min_commit_scale: float,
    observed_stress_values: frozenset[str] | None = None,
    bic_config: BICConfig | None = None,
) -> None:
    """Sequential greedy context splitting for one pivot (lect, grapheme).

    Reuses :func:`_group_cost`, :func:`_candidate_predicates`,
    :func:`_partition`, :func:`_apply_predicate` — the same
    machinery the per-pair context-discovery stage uses. The
    only difference is the emitter: committed partitions are
    turned into multi-lect class tuples rather than written
    into a segment table.

    Adaptive tuning:

    - ``use_bic_small_sample_correction``: adds an AICc-style
      ``2/max(n-1, 1)`` term to the BIC penalty so small-n pivots
      require proportionally more evidence. Default True.
    - ``min_commit_scale``: scales the minimum per-sister-tuple
      count below which a class is not emitted. See
      :func:`_multi_lect_min_commit_count`.
    """
    if bic_config is None:
        bic_config = BICConfig()
    min_obs = bic_config.min_split_observations
    max_depth = bic_config.max_split_depth
    delta_threshold = bic_config.delta_bic_threshold

    remaining = list(observations)
    n_total = sum(_obs_weight(obs) for obs in observations)
    if n_total < 4:
        return
    ln_n = math.log(n_total)
    penalty = ln_n
    if use_bic_small_sample_correction:
        penalty += 2.0 / max(n_total - 1, 1)
    min_commit = _multi_lect_min_commit_count(n_total, min_commit_scale)
    committed_count = 0
    while (
        committed_count < max_depth * 4
        and sum(_obs_weight(obs) for obs in remaining) >= min_obs
    ):
        baseline_cost = _group_cost(remaining)
        best_predicate: tuple[str, str, str | None] | None = None
        best_partitions: tuple[list, list] | None = None
        best_delta = delta_threshold
        for predicate in _candidate_predicates(Context(), observed_stress_values):
            yes_obs, no_obs = _partition(remaining, predicate)
            if (
                sum(_obs_weight(obs) for obs in yes_obs) < min_obs
                or sum(_obs_weight(obs) for obs in no_obs) < min_obs
            ):
                continue
            split_cost = _group_cost(yes_obs) + _group_cost(no_obs)
            reduction = baseline_cost - split_cost
            delta_bic = -2.0 * reduction + penalty
            if delta_bic < best_delta:
                best_delta = delta_bic
                best_predicate = predicate
                best_partitions = (yes_obs, no_obs)
        if best_predicate is None or best_partitions is None:
            break
        yes_obs, no_obs = best_partitions
        yes_ctx = _apply_predicate(Context(), best_predicate)
        # Emit one class per distinct sister tuple in the yes partition
        # whose count meets the adaptive minimum.
        sister_counts: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        for obs in yes_obs:
            sister_key = obs[0]
            sister_counts[sister_by_key[sister_key]] += _obs_weight(obs)
        for sister, count in sister_counts.items():
            if count < min_commit:
                continue
            committed.append(
                (pivot_lect, pivot_grapheme, yes_ctx, sister, count, n_total)
            )
        remaining = no_obs
        committed_count += 1


def _commit_multi_lect_long_range_splits_for_pivot(
    pivot_lect: str,
    pivot_grapheme: str,
    observations: list[tuple[str, Context, float]],
    sister_by_key: dict[str, tuple[tuple[str, str], ...]],
    committed: list[
        tuple[str, str, Context, tuple[tuple[str, str], ...], float, float]
    ],
    *,
    min_commit_scale: float,
    bic_config: BICConfig | None = None,
) -> None:
    """Long-range parallel of the multi-lect class-discovery
    pivot split loop.

    Same shape as :func:`_commit_multi_lect_splits_for_pivot`
    but enumerates only long-range predicates (distance-bounded,
    existential, syllable-structural). Uses the stricter
    thresholds from per-pair long-range discovery
    (``_LONG_RANGE_DELTA_BIC_THRESHOLD``,
    ``_LONG_RANGE_MIN_SPLIT_OBS``,
    ``_LONG_RANGE_MIN_DOMINANT_FRACTION``) so the multi-lect
    lift inherits the same noise filtering proven on real
    per-pair data.

    Operates on the same observation set the immediate-neighbour
    loop already saw — long-range commits are additive overlays,
    not partitions of the residual.
    """
    if bic_config is None:
        bic_config = BICConfig()
    min_obs = bic_config.long_range_min_split_observations
    max_depth = bic_config.max_split_depth
    delta_threshold = bic_config.long_range_delta_bic_threshold
    dominant_fraction = bic_config.long_range_min_dominant_fraction

    n_total = sum(_obs_weight(obs) for obs in observations)
    if n_total < min_obs:
        return
    ln_n = math.log(n_total)
    # Long-range multi-lect commits use a strict floor independent
    # of the adaptive `_multi_lect_min_commit_count` because the
    # adaptive function scales with ``log2(n+1)`` and gives min=2
    # for small pivot buckets — too lax for long-range, which
    # already has many candidate predicates and is prone to
    # 3-observation noise commits. Use the per-pair long-range
    # floor as the firm minimum.
    min_commit = max(
        _multi_lect_min_commit_count(n_total, min_commit_scale),
        min_obs,
    )
    remaining = list(observations)
    committed_count = 0
    while (
        committed_count < max_depth * 4
        and sum(_obs_weight(obs) for obs in remaining) >= min_obs
    ):
        baseline_cost = _group_cost(remaining)
        best_predicate: tuple[str, str, str | None] | None = None
        best_partitions: tuple[list, list] | None = None
        best_delta = delta_threshold
        for predicate in _long_range_candidate_predicates(Context()):
            yes_obs, no_obs = _partition(remaining, predicate)
            if (
                sum(_obs_weight(obs) for obs in yes_obs) < min_obs
                or sum(_obs_weight(obs) for obs in no_obs) < min_obs
            ):
                continue
            yes_target_counts: dict[str, float] = defaultdict(float)
            for obs in yes_obs:
                yes_target_counts[obs[0]] += _obs_weight(obs)
            yes_mode = max(yes_target_counts.values())
            yes_weight = sum(_obs_weight(obs) for obs in yes_obs)
            if yes_weight <= 0.0 or yes_mode / yes_weight < dominant_fraction:
                continue
            split_cost = _group_cost(yes_obs) + _group_cost(no_obs)
            reduction = baseline_cost - split_cost
            delta_bic = -2.0 * reduction + ln_n
            if delta_bic < best_delta:
                best_delta = delta_bic
                best_predicate = predicate
                best_partitions = (yes_obs, no_obs)
        if best_predicate is None or best_partitions is None:
            break
        yes_obs, no_obs = best_partitions
        yes_ctx = _apply_predicate(Context(), best_predicate)
        sister_counts: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        for obs in yes_obs:
            sister_key = obs[0]
            sister_counts[sister_by_key[sister_key]] += _obs_weight(obs)
        for sister, count in sister_counts.items():
            if count < min_commit:
                continue
            committed.append(
                (pivot_lect, pivot_grapheme, yes_ctx, sister, count, n_total)
            )
        remaining = no_obs
        committed_count += 1


def _train_multi_lect(
    corpus: Sequence[CognateSet],
    *,
    feature_system: str,
    max_chunk_size: int,
    temperature: float,
    concentration: float,
    max_iter: int,
    convergence_eps: float,
    segment_weight: float,
    displacement_weight: float,
    tone_weight: float,
    chunk_min_transparency: float = 0.0,
    typological_prior: "TypologicalPrior | None" = None,
    bic_config: BICConfig | None = None,
) -> MultiLectModel:
    """Multi-lect training.

    For N=1 returns an empty model; for N≥2 trains a per-pair
    model for each pair with shared cognates, reconciles the
    alignments through a union-find over corresponding
    positions, runs the multi-lect class-discovery loop on the
    reconciled observations, and lifts per-pair
    cross-dimensional commits to the multi-lect table.
    """
    import itertools

    from regulae.training import _train_pairwise_legacy

    lect_ids = _collect_lect_ids(corpus)
    if len(lect_ids) == 0:
        return MultiLectModel.empty()
    if len(lect_ids) == 1:
        return MultiLectModel(
            pairwise_models={},
            unconditioned_classes=(),
            conditioned_classes=(),
            cognate_corpus=tuple(corpus),
            lect_ids=lect_ids,
        )

    pairwise_models: dict[frozenset[str], LearnedModel] = {}
    for lect_a, lect_b in itertools.combinations(lect_ids, 2):
        pair_corpus, pair_weights = _subset_for_pair(corpus, lect_a, lect_b)
        if not pair_corpus:
            continue
        pair_model = _train_pairwise_legacy(
            pair_corpus,
            pair_weights=pair_weights,
            feature_system=feature_system,
            max_chunk_size=max_chunk_size,
            temperature=temperature,
            concentration=concentration,
            max_iter=max_iter,
            convergence_eps=convergence_eps,
            segment_weight=segment_weight,
            displacement_weight=displacement_weight,
            tone_weight=tone_weight,
            chunk_min_transparency=chunk_min_transparency,
            typological_prior=typological_prior,
            bic_config=bic_config,
        )
        pairwise_models[frozenset({lect_a, lect_b})] = pair_model

    # Reconcile pairwise alignments into multi-lect
    # correspondence class observations, then aggregate across
    # the corpus.
    all_observations: list[tuple[str, dict[str, str], float]] = []
    per_lect_observations: list[tuple[str, dict[str, str], dict[str, int], float]] = []
    for cs in corpus:
        rows = _reconcile_cognate_set(
            cs, pairwise_models, max_chunk_size=max_chunk_size
        )
        for segments, positions in rows:
            all_observations.append((cs.cognate_id, segments, cs.confidence))
            per_lect_observations.append((cs.cognate_id, segments, positions, cs.confidence))
    unconditioned = _aggregate_unconditioned_classes(all_observations)

    # Multi-lect class-level context discovery.
    cfg = bic_config if bic_config is not None else BICConfig()
    conditioned = _multi_lect_context_discovery(
        corpus=corpus,
        observations=per_lect_observations,
        feature_system=feature_system,
        start_class_id=len(unconditioned),
        bic_small_sample_correction=cfg.multi_lect_bic_small_sample_correction,
        min_commit_scale=cfg.multi_lect_min_commit_scale,
        bic_config=cfg,
    )

    # Lift per-pair cross-dimensional rules to the multi-lect
    # level with explicit src_lect/tgt_lect labels.
    cross_dim_table = _lift_cross_dimensional_rules(
        pairwise_models, lect_ids
    )

    return MultiLectModel(
        pairwise_models=pairwise_models,
        unconditioned_classes=unconditioned,
        conditioned_classes=conditioned,
        cognate_corpus=tuple(corpus),
        lect_ids=lect_ids,
        cross_dimensional_table=cross_dim_table,
    )


def _lift_cross_dimensional_rules(
    pairwise_models: Mapping[frozenset[str], LearnedModel],
    lect_ids: tuple[str, ...],
) -> MultiLectCrossDimensionalLinkTable:
    """Lift per-pair cross-dimensional commits to the multi-lect
    level.

    Iterates over the canonical pair ordering from
    :func:`_collect_lect_ids` so the ``(src_lect, tgt_lect)``
    labels on lifted rules match the direction in which the pair
    was actually trained. Every :class:`CrossDimensionalLink` in
    a per-pair table becomes one
    :class:`MultiLectCrossDimensionalLink` with those labels
    attached — no new discovery runs at the multi-lect level, so
    the rules inherit the same consolidation filters (support
    floors, dual-framing dedup) that the per-pair cross-dimensional
    commit loop already applied.

    Deterministic under the ``lect_ids`` ordering.
    """
    import itertools

    entries: list[MultiLectCrossDimensionalLink] = []
    for lect_a, lect_b in itertools.combinations(lect_ids, 2):
        key = frozenset({lect_a, lect_b})
        pm = pairwise_models.get(key)
        if pm is None:
            continue
        for rule in pm.cross_dimensional_table.entries:
            entries.append(
                MultiLectCrossDimensionalLink(
                    src_lect=lect_a,
                    tgt_lect=lect_b,
                    src_feature=rule.src_feature,
                    src_position=rule.src_position,
                    tgt_dimension=rule.tgt_dimension,
                    tgt_value=rule.tgt_value,
                    tgt_position_offset=rule.tgt_position_offset,
                    count=rule.count,
                    src_count=rule.src_count,
                    confidence=rule.confidence,
                    src_feature_2=rule.src_feature_2,
                    src_position_2=rule.src_position_2,
                    uncertainty=rule.uncertainty,
                )
            )
    return MultiLectCrossDimensionalLinkTable(entries=tuple(entries))
