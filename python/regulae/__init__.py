"""Native C-backed Python wrapper for regulae.

The package trains through the C core and parses its JSON into dataclasses. It
does no modelling of its own: the boundary is the JSON the core renders, so the
wrapper stays correct for free and inherits every C feature.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from regulae._model import (
    GAP_GRAPHEME,
    ConditionedCorrespondence,
    CorrespondenceClass,
    CrossDimensionalRule,
    NullCorrespondence,
    MultiLectModel,
    PairwiseModel,
    PredictiveEvidence,
    Segment,
    Uncertainty,
)

try:
    from regulae import _native as _native  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - exercised before the extension is built
    raise ImportError(
        "regulae requires its native C extension. Install the package from a built "
        "wheel or run `python -m pip install -e .` from the repository root."
    ) from exc

__version__ = "0.1.0"

#: Raised for a genuine training or parsing failure in the C core.
RegulaeError = _native.RegulaeError
#: Raised for CLDF/CLTS markup that is not a sound (a ValueError subclass, so
#: code catching ValueError is unaffected).
SourceMarkerError = _native.SourceMarkerError


def _corpus_text(source: str | os.PathLike[str]) -> str:
    """A corpus as text: a path is read, a string is taken as the corpus itself.

    A short single-line string with no tab is treated as a path even without a
    newline; anything that looks like TSV content is used directly.
    """
    if isinstance(source, os.PathLike):
        with open(source, encoding="utf-8") as handle:
            return handle.read()
    if "\n" not in source and "\t" not in source and os.path.exists(source):
        with open(source, encoding="utf-8") as handle:
            return handle.read()
    return source


def train_model(
    source: str | os.PathLike[str],
    *,
    fmt: str = "wide",
    options: Mapping[str, Any] | None = None,
) -> MultiLectModel:
    """Train a model on a corpus and return it.

    ``source`` is a corpus as text or a path to one. ``fmt`` is the loader:
    ``"wide"`` (one row per cognate, default), ``"tsv"``, ``"gled"`` or
    ``"arcaverborum"``. ``options`` is a flat mapping of training options passed
    to the C core, which rejects unknown keys.
    """
    options_json = json.dumps(dict(options)) if options else None
    payload = _native.train(_corpus_text(source), fmt, options_json)
    return MultiLectModel.from_json(json.loads(payload))


def train_ensemble(
    source: str | os.PathLike[str],
    *,
    fmt: str = "wide",
    options: Mapping[str, Any] | None = None,
    members: int = 50,
    seed: int = 20261030,
    temperature: float | str = 1.0,
    partitions: bool = False,
    cognacy_pairs: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Train an ensemble and return the ``regulae_package`` (docs/ensembles.md).

    Each member resamples the observation groups with Bayesian-bootstrap
    weights, retrains the whole pipeline, and draws its alignments at
    ``temperature`` (0 is the cheapest path). The package carries the members
    and a summary keyed by content: inclusion probability and rate interval
    per class, per pair correspondence, and per correspondence conditioned
    under any environment. A single model (``train_model``) is tier C0; this
    is C1, the minimum the framework allows for claim export.

    ``temperature="calibrate"`` first chooses it by held-out log loss over
    the default grid. With ``cognacy_pairs`` (form_a, form_b, weight), the
    source is a forms table (form_id, lect_id, meaning, segments) and each
    member samples its own cognate partition from the weights.
    """
    options_json = json.dumps(dict(options)) if options else None
    calibrate = temperature == "calibrate"
    payload = _native.train_ensemble(
        _corpus_text(source),
        fmt,
        options_json,
        members,
        seed,
        1.0 if calibrate else float(temperature),
        partitions,
        calibrate,
        None if cognacy_pairs is None else _corpus_text(cognacy_pairs),
    )
    return json.loads(payload)


def segment(word: str) -> list[str]:
    """Segment one written word into graphemes, as the C core reads it.

    A caller can check how input will be interpreted before a full run.
    Suprasegmentals such as tone are lifted onto their segment, so they do not
    appear as separate entries.
    """
    payload = json.loads(_native.segment(word))
    return list(payload.get("segments", []))


def version() -> str:
    """The C library version string."""
    return str(_native.version())


def abi_version() -> int:
    """The C ABI version integer."""
    return int(_native.abi_version())


__all__ = [
    "GAP_GRAPHEME",
    "ConditionedCorrespondence",
    "CorrespondenceClass",
    "CrossDimensionalRule",
    "NullCorrespondence",
    "MultiLectModel",
    "PairwiseModel",
    "PredictiveEvidence",
    "RegulaeError",
    "Segment",
    "SourceMarkerError",
    "Uncertainty",
    "__version__",
    "abi_version",
    "segment",
    "train_ensemble",
    "train_model",
    "version",
]
