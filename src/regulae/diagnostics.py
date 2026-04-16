"""Diagnostic helpers over a trained :class:`MultiLectModel`.

These functions consume an already-trained model and the corpus it
was trained on, and emit human-oriented reports rather than
modifying the model. Nothing here feeds back into training,
scoring, or alignment — the framework's behavior is unchanged
whether or not any of these are called.

Current inventory:

* :func:`find_cognate_outliers` — rank cognate sets by
  alignment-cost anomaly so the user can spot likely non-cognates,
  loanwords, transcription errors, or genuinely atypical regular
  forms.

-- On cognate input --

The framework uses an explicit-cognate input model: it does **not** try to
infer cognacy from glosses or wordlists, and does **not** filter
out pairs that look non-cognate during training. Every pair the
user passes in is trained on. This is a deliberate choice: glosses
are unreliable cognacy markers, and automatic cognate-detection
belongs to a different tool, not to the alignment engine.

The consequence is that if you pass the framework a pair list
that contains non-cognates — e.g., the classic ``kaput`` /
``kabeθa`` (Latin *caput* "head" vs Spanish "head" < *capitia*)
— those pairs get trained on and contribute observations to the
segment table. The framework has no way to know they shouldn't
be there.

:func:`find_cognate_outliers` is a **post-hoc diagnostic**: after
training, it ranks the input cognate sets by how well their forms
actually align under the model. Pairs whose alignment cost is far
above the corpus mean are flagged as outliers. The user then
decides what to do about them: drop them and retrain, keep them
with a note, or accept them as genuine regular-but-rare
correspondences.

This is a tool, not a gate. The framework stays honest: we never
silently reject input.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

from regulae.model import CognateSet, LearnedModel, MultiLectModel
from regulae.search import align_forms, alignment_cost


@dataclass(frozen=True)
class CognateOutlierReport:
    """One row of a cognate outlier ranking.

    * ``cognate_id`` — identifier of the input cognate set.
    * ``n_pairs`` — number of lect pairs whose alignment was
      computed for this set (``N*(N-1)/2`` for an N-lect set,
      minus any pairs with no trained model).
    * ``cost_per_segment`` — mean alignment cost per segment
      averaged over all contributing pair alignments for this
      cognate set. Length-normalized so short words aren't
      mechanically cheaper than long ones.
    * ``z_score`` — how many standard deviations above the
      corpus mean the ``cost_per_segment`` sits. Positive
      values are the outliers; strong outliers have z > 2.
    """

    cognate_id: str
    n_pairs: int
    cost_per_segment: float
    z_score: float


def _pair_cost_per_segment(
    form_a, form_b, model: LearnedModel, max_chunk_size: int
) -> float:
    """Compute length-normalized alignment cost for one pair.

    Cost normalization divides the total alignment cost by
    ``(len(form_a) + len(form_b)) / 2``. Short cognates and long
    cognates are compared on the same scale.
    """
    alignment = align_forms(
        form_a, form_b, max_chunk_size=max_chunk_size, model=model
    )
    cost = alignment_cost(alignment, model=model)
    denom = (len(form_a.segments) + len(form_b.segments)) / 2.0
    if denom <= 0.0:
        return 0.0
    return cost / denom


def find_cognate_outliers(
    corpus: Sequence[CognateSet],
    model: MultiLectModel,
    *,
    top_k: int | None = None,
    max_chunk_size: int = 3,
) -> list[CognateOutlierReport]:
    """Rank cognate sets in the corpus by alignment-cost anomaly.

    For each cognate set, every pair of lects present in both the
    set and the model's ``pairwise_models`` is aligned, and the
    length-normalized per-segment alignment cost is averaged
    across those pairs. Cognate sets are then sorted by a z-score
    against the corpus-wide mean and standard deviation of that
    statistic.

    Arguments:

    * ``corpus``: the cognate sets the model was trained on (or
      any cognate sets you'd like to score under this model).
    * ``model``: a :class:`MultiLectModel` whose
      ``pairwise_models`` covers the lects in the corpus.
    * ``top_k``: if given, return only the top ``k`` entries by
      z-score (the most suspect sets). If ``None`` (default),
      return every set that had at least one computable pair.
    * ``max_chunk_size``: forwarded to :func:`align_forms`.

    Returns a list of :class:`CognateOutlierReport`, sorted by
    z-score descending (most anomalous first).

    Interpretation:

    * ``z_score`` near 0 means the set aligns about as cheaply as
      a typical set in this corpus — nothing unusual.
    * ``z_score > 2`` is traditionally "significantly above the
      mean". These are candidates for inspection: likely
      non-cognates, loanwords, transcription errors, or
      genuinely unusual regular sets.
    * Negative ``z_score`` means the set aligns cheaper than
      average. These are the boring, confirmatory sets — the
      bread and butter of the correspondence evidence.

    The function is a **diagnostic**, not a filter. It does not
    modify the model or the corpus. It's meant to be called
    once after training to decide which sets deserve a second
    look from a human or a separate reviewer.
    """
    scored: list[tuple[str, int, float]] = []  # (id, n_pairs, cost_per_seg)
    for cs in corpus:
        lects = sorted(cs.forms)
        total = 0.0
        n_pairs = 0
        for lect_a, lect_b in itertools.combinations(lects, 2):
            key = frozenset({lect_a, lect_b})
            pair_model = model.pairwise_models.get(key)
            if pair_model is None:
                continue
            c = _pair_cost_per_segment(
                cs.forms[lect_a],
                cs.forms[lect_b],
                pair_model,
                max_chunk_size,
            )
            total += c
            n_pairs += 1
        if n_pairs == 0:
            continue
        avg = total / n_pairs
        scored.append((cs.cognate_id, n_pairs, avg))

    if not scored:
        return []

    # Compute mean and standard deviation of the per-segment costs.
    # For small corpora (n < 2) std is undefined; treat it as 0 and
    # return z-scores of 0 (nothing is an outlier if we have no
    # distribution to compare against).
    costs = [c for _, _, c in scored]
    n = len(costs)
    mean = sum(costs) / n
    if n < 2:
        std = 0.0
    else:
        variance = sum((c - mean) ** 2 for c in costs) / (n - 1)
        std = math.sqrt(variance)

    reports: list[CognateOutlierReport] = []
    for cid, npairs, cost in scored:
        z = (cost - mean) / std if std > 0.0 else 0.0
        reports.append(
            CognateOutlierReport(
                cognate_id=cid,
                n_pairs=npairs,
                cost_per_segment=cost,
                z_score=z,
            )
        )

    reports.sort(key=lambda r: (-r.z_score, r.cognate_id))
    if top_k is not None:
        reports = reports[:top_k]
    return reports
