# regulae (Python)

A thin Python wrapper over the regulae C core. It trains through the C library
and parses the model it renders as JSON into dataclasses; it does no modelling
of its own.

The boundary is the JSON the core renders, which means the wrapper inherits
every C feature for free and an ABI change never reaches it.

## Install

The extension compiles the C core — regulae, the vendored cJSON, and the sibling
`merkmal` C checkout — straight in, so a built wheel needs neither an installed
library nor the merkmal Python package. From the repository root:

```
pip install -e .
```

`merkmal`'s C sources are found at `../merkmal` by default; set
`REGULAE_MERKMAL_SOURCE_DIR` to point elsewhere. `libutf8proc` is used for
Unicode normalization when `pkg-config` finds it, with an IPA-focused fallback
otherwise; set `REGULAE_REQUIRE_UTF8PROC=1` in a distribution build to fail
rather than ship the fallback.

## Use

```python
import regulae

model = regulae.train_model(open("cognates.tsv").read(), fmt="tsv")

for klass in model.unconditioned_classes:
    print(klass.graphemes, "on", len(klass.supporting_cognates), "sets")

for rule in model.cross_dimensional:
    kind = "lect-internal" if rule.dimension_from_environment else "cross-lect"
    print(kind, rule.environment_lect, "->", rule.dimension, rule.value)
```

`train_model` takes a corpus as text or a path, a loader (`fmt`: `"wide"`,
`"tsv"`, `"gled"`, `"arcaverborum"`), and an `options` mapping passed to the
core. The returned `MultiLectModel` keeps the full JSON on `.raw` for any field
the dataclasses do not surface.

A `regulae` command is installed for `regulae CORPUS --format tsv`; the C project
ships its own fuller `regulae` binary.
