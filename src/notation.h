#ifndef REGULAE_NOTATION_H
#define REGULAE_NOTATION_H

#include "regulae.h"

/* An environment written in rule notation, beside the exact form and never
 * instead of it.
 *
 * The published slot syntax -- `latin: prev-syl[syllable_shape:open]` -- is
 * exact and unreadable to the people the reports are for. A comparativist
 * reads `s ~ r / V _ V`. This renders the second one. The first stays the
 * contract: anything ambiguous here can be resolved by looking one line up.
 *
 * The notation is defined in docs/NOTATION.md, which is the specification and
 * not a summary of this file. There is no notation to adopt wholesale --
 * the core (`_`, `/`, `#`, `∅`, `{}`, `[±F]`, `σ`, `…`) is agreed across the
 * literature and nothing past it is, and six of the eighteen slots have no
 * notation anywhere. Where none existed this defines one and publishes the
 * key.
 *
 * The one departure that is a claim rather than a convention: standard
 * notation is `A > B / X _ Y`, directional, with the environment implicitly in
 * the ancestor's word. regulae has no ancestor. Its environment belongs to a
 * *named lect* -- the computational orientation -- so the notation is
 * `A ~ B / lect: X _ Y` and the label is part of the finding.
 */

/* One lect's side of a correspondence: what it shows, and where.
 *
 * `graphemes` may be a single-element set, and one of its elements may be the
 * gap. `suprasegmentals` and `context` may be null. `lect` may not: a frame
 * with nothing to attribute it to is the one thing this notation may not
 * print. */
typedef struct rg_notation_side {
    const char *lect;
    const char *const *graphemes;
    size_t grapheme_count;
    const rg_suprasegmentals *suprasegmentals;
    const rg_context_spec *context;
} rg_notation_side;

/* One lect's environment as a frame -- "V _ V", "σ⁻[open] _", "# _".
 *
 * Returns an empty string when the environment states nothing, so a caller can
 * test the result rather than counting constraints itself. Returns 0 only on
 * allocation failure. Caller frees. */
char *rg_notation_frame_internal(const rg_context_spec *context);

/* The whole line -- "r ~ s  /  old_latin: V _ V".
 *
 * Lect labels are dropped when two or more sides state the same frame, since
 * repeating it says something the data does not: that the two were found
 * separately. A single conditioning side always keeps its label, because which
 * lect it is is exactly what the reader cannot otherwise know.
 *
 * Caller frees. 0 on allocation failure. */
char *rg_notation_line_internal(const rg_notation_side *sides, size_t count);

/* A cross-dimensional rule -- "daughter [tone=2]  /  proto: _ [+voiced]".
 *
 * Not a correspondence and so not written with `~`: one lect carries a
 * suprasegmental value where another lect's environment holds. Caller frees. */
char *rg_notation_cross_dimensional_internal(
    const char *conditioned_lect,
    const char *dimension,
    const char *value,
    const char *environment_lect,
    const rg_context_spec *environment
);

/* Marks, in `seen`, every environment slot the renderer names, and returns how
 * many slots there are. Zero if `capacity` is too small to answer.
 *
 * The renderer enumerates the slots by hand -- it has to, since each one is
 * written in a different place in the frame -- so the guard against a new slot
 * being added and silently dropped is in two halves: this expansion, which
 * fails to compile when RG_ENV_SLOTS grows a name the renderer has no
 * enumerator for, and a test that requires every slot to render to something.
 * Neither half is sufficient alone. */
size_t rg_notation_slot_census_internal(int *seen, size_t capacity);

#endif
