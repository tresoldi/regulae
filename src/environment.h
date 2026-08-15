#ifndef REGULAE_ENVIRONMENT_H
#define REGULAE_ENVIRONMENT_H

#include "regulae.h"

/* The environment -- `rg_context_spec` -- and everything derived from its slot
 * list.
 *
 * An environment is an `rg_context_spec` wherever it appears: conditioned
 * correspondences and cross-dimensional rules both state one, so both can name
 * more than one predicate. It has eighteen slots, and until this header they
 * were enumerated by hand in fourteen places -- counting, subset matching, the
 * total-order comparator, copy, clear, JSON, the human format, the CLI's key,
 * three predicate dispatchers, two link-context builders and two tests. None of
 * them referenced any other, and they had already drifted: the CLI's key left
 * out `self` and `morpheme_index`, so two environments that differ only there
 * rendered identically, and `multilect_classes.c` carried a second copy of the
 * slot-name strings.
 *
 * So the list lives here once, and every operation expands it. Adding a
 * predicate is a line in this file, and the compiler finds the sites that have
 * to say something about it.
 *
 * The list is a macro rather than a table of offsets on purpose: each expansion
 * is ordinary typed code, so a slot used at the wrong arity or with the wrong
 * constraint type fails to compile rather than reading a wrong number of bytes.
 */

/* Nothing, for a consumer that wants only one kind of slot. */
#define RG_ENV_SLOT_SKIP(name, label)

/* The constraint slots, in the order that *is* the total order over
 * environments -- `rg_context_spec_compare_internal` walks them in exactly this
 * sequence, published tables are sorted by it, and lookups binary-search that
 * sort. Reordering this list reorders every published table and moves every
 * class id downstream. It is not a stylistic choice.
 *
 * F is applied to the feature-constraint slots and D to the distance-constraint
 * ones; pass the same macro twice to treat them alike. The label is the short
 * form the human report prints; the field name serialises itself.
 */
#define RG_ENV_SLOTS(F, D)                              \
    F(preceding,             "pre")                     \
    F(following,             "fol")                     \
    D(preceding_at_distance, "pre@")                    \
    D(following_at_distance, "fol@")                    \
    F(somewhere_preceding,   "somewhere-pre")           \
    F(somewhere_following,   "somewhere-fol")           \
    F(same_syllable,         "same-syl")                \
    F(next_syllable,         "next-syl")                \
    F(previous_syllable,     "prev-syl")                \
    F(self,                  "self")                    \
    F(self_stress,           "self-stress")             \
    F(preceding_stress,      "pre-stress")              \
    F(following_stress,      "fol-stress")

/* The feature slots alone, and the distance slots alone, both in the order
 * above. The reports emit every feature slot and then every distance slot,
 * which is what these two produce in sequence; the comparator interleaves them,
 * which is what RG_ENV_SLOTS produces. Both orders come off the one list, so
 * neither can drift from it. */
#define RG_ENV_FEATURE_SLOTS(X)  RG_ENV_SLOTS(X, RG_ENV_SLOT_SKIP)
#define RG_ENV_DISTANCE_SLOTS(X) RG_ENV_SLOTS(RG_ENV_SLOT_SKIP, X)

/* The slots holding a single string rather than a list, in report order.
 *
 * The comparator does *not* use this list: its order over the three is
 * position, morphological, morpheme_index, and it says so where it writes them
 * out. Two orders over three fields is not worth a second macro, and the total
 * order is worth seeing spelled out at the place it is decided. */
#define RG_ENV_STRING_SLOTS(X)  \
    X(position)                 \
    X(morpheme_index)           \
    X(morphological)

/* A total order over everything an environment can express.
 *
 * Ordering on a summary -- position and constraint count -- leaves rows that
 * differ only in which constraint they carry comparing equal, and qsort is free
 * to return those in either order. It did: the same corpus published the same
 * rows in a different order under the native and the WebAssembly build, which
 * moved every class id downstream. */
int rg_context_spec_compare_internal(const rg_context_spec *a, const rg_context_spec *b);

/* The slots a candidate may name in each of the two families the discovery
 * stages search over. Both the pairwise and the multi-lect stage build
 * candidate lists from them, and until they were shared each kept its own copy
 * of the same strings in the same order.
 *
 * These are subsets of the slot list above, not the whole of it: the immediate
 * neighbours, position and morphology are enumerated where a stage decides what
 * it can afford to search, which differs between the two. */
extern const char *const rg_env_long_range_slots[];
extern const size_t rg_env_long_range_slot_count;
extern const char *const rg_env_stress_slots[];
extern const size_t rg_env_stress_slot_count;

/* Every stress value the environment's three stress slots name, in slot order,
 * handed to `add` one at a time. Both discovery stages build an inventory of
 * the stress values a corpus actually shows, so that the candidates they search
 * hold only those; they kept the walk twice, byte for byte, differing only in
 * which container it appended to. What they keep differing about is the
 * container, which is theirs. */
rg_status rg_env_collect_stress_values(
    const rg_context_spec *context,
    rg_status (*add)(void *user, const char *value),
    void *user
);

#endif
