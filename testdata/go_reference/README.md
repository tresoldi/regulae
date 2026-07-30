# Go Reference Fixtures

This directory contains M0 fixtures for the C rewrite. The files were
generated from the current Go implementation after temporarily restoring
the old sibling merkmal Go module from `../merkmal` history, as described
in `docs/c_reference_freeze.md`.

Use these files as regression evidence while porting. Prefer structured
model parity when C accessors exist; until then, these experiment outputs
are the human-readable golden surface.
