#include "notation.h"

#include "environment.h"
#include "strbuf.h"

#include <stdlib.h>
#include <string.h>

/* The symbols. Written as escapes because the rest of the library writes its
 * non-ASCII that way (RG_GAP_GRAPHEME, the delta on a displacement), with the
 * character in the comment so the table can still be read. */
#define NOT_MINUS       "\xe2\x88\x92"  /* U+2212 minus, not the hyphen in "close-mid" */
#define NOT_ELLIPSIS    "\xe2\x80\xa6"  /* U+2026 … any number of segments, including none */
#define NOT_MIDDOT      "\xc2\xb7"      /* U+00B7 · one unspecified segment */
#define NOT_SIGMA       "\xcf\x83"      /* U+03C3 σ */
#define NOT_SUP_MINUS   "\xe2\x81\xbb"  /* U+207B ⁻ the previous syllable */
#define NOT_SUP_PLUS    "\xe2\x81\xba"  /* U+207A ⁺ the next syllable */
#define NOT_SUP_EQUALS  "\xe2\x81\xbc"  /* U+207C ⁼ the same syllable */
#define NOT_SUB_MINUS   "\xe2\x82\x8b"  /* U+208B ₋ counted from the end */
#define NOT_SUB_ONE     "\xe2\x82\x81"  /* U+2081 ₁ */
#define NOT_SUB_TWO     "\xe2\x82\x82"  /* U+2082 ₂ */
#define NOT_SUB_THREE   "\xe2\x82\x83"  /* U+2083 ₃ */
#define NOT_ANGLE_OPEN  "\xe2\x9f\xa8"  /* U+27E8 ⟨ a structural predicate, not a feature */
#define NOT_ANGLE_CLOSE "\xe2\x9f\xa9"  /* U+27E9 ⟩ */
#define NOT_ACUTE       "\xc2\xb4"      /* U+00B4 ´ primary stress */
#define NOT_GRAVE       "\xcb\x8b"      /* U+02CB ˋ secondary stress */

/* The cover symbols, each abbreviating exactly one positive merkmal feature.
 *
 * The set is Index Diachronica's, restricted to the symbols that survive the
 * restriction: L, W, Q and M name no single merkmal feature, U would collide
 * with the syllable sigma, H is one symbol for merkmal's separate `glottal`
 * and `guttural`, and D and T are conjunctions -- writing `[+voiced,+stop]` as
 * D would make one symbol mean what the matrix notation already says, and mean
 * it only sometimes.
 *
 * Two of the survivors diverge from Index Diachronica and docs/NOTATION.md
 * says so outright: E and B abbreviate merkmal's `front` and `back`, which are
 * not restricted to vowels, and S is `stop` where much handbook use has S for
 * a sibilant. A reader who misses either misreads the output, which is why
 * they are in the key rather than in a footnote.
 *
 * These are feature *names*, so they hold for the descriptive vocabulary
 * regulae defaults to. Under another feature system the names do not match and
 * every constraint renders as a matrix: correct, and less pretty. */
static const struct {
    const char *feature;
    const char *symbol;
} notation_cover_symbols[] = {
    {"vowel",       "V"},
    {"consonant",   "C"},
    {"nasal",       "N"},
    {"fricative",   "F"},
    {"stop",        "S"},
    {"affricate",   "A"},
    {"obstruent",   "O"},
    {"sonorant",    "R"},
    {"approximant", "J"},
    {"continuant",  "Z"},
    {"labial",      "P"},
    {"velar",       "K"},
    {"front",       "E"},
    {"back",        "B"}
};

/* The slots this file knows how to write.
 *
 * Hand-written, and that is the point: the census below fills it from
 * RG_ENV_SLOTS and RG_ENV_STRING_SLOTS, so a slot added to either list has no
 * enumerator here and the file stops compiling until someone decides how it is
 * notated. Generating this enum from the same macro would make the check pass
 * for a slot nobody had thought about, which is the failure it exists to
 * prevent -- the environment's eighteen slots were once enumerated by hand in
 * fourteen places, and they had already drifted. */
enum notation_slot {
    NOTATION_SLOT_preceding,
    NOTATION_SLOT_following,
    NOTATION_SLOT_preceding_at_distance,
    NOTATION_SLOT_following_at_distance,
    NOTATION_SLOT_somewhere_preceding,
    NOTATION_SLOT_somewhere_following,
    NOTATION_SLOT_same_syllable,
    NOTATION_SLOT_next_syllable,
    NOTATION_SLOT_previous_syllable,
    NOTATION_SLOT_self,
    NOTATION_SLOT_self_stress,
    NOTATION_SLOT_preceding_stress,
    NOTATION_SLOT_following_stress,
    NOTATION_STRING_SLOT_position,
    NOTATION_STRING_SLOT_morpheme_index,
    NOTATION_STRING_SLOT_morphological,
    NOTATION_STRING_SLOT_syllable_role,
    NOTATION_STRING_SLOT_syllable_position,
    NOTATION_SLOT_TOTAL
};

size_t rg_notation_slot_census_internal(int *seen, size_t capacity) {
    size_t i;
    if (seen == 0 || capacity < (size_t)NOTATION_SLOT_TOTAL) {
        return 0;
    }
    for (i = 0; i < capacity; i++) {
        seen[i] = 0;
    }
#define SLOT(name, label, key) seen[NOTATION_SLOT_##name] = 1;
    RG_ENV_SLOTS(SLOT, SLOT)
#undef SLOT
#define STRING_SLOT(name, key) seen[NOTATION_STRING_SLOT_##name] = 1;
    RG_ENV_STRING_SLOTS(STRING_SLOT)
#undef STRING_SLOT
    return (size_t)NOTATION_SLOT_TOTAL;
}

static int notation_present(const char *value) {
    return value != 0 && value[0] != '\0';
}

/* Tokens are separated by exactly one space, decided here rather than at each
 * of the twenty places that emit one. A frame builder starts empty, so the
 * leading token gets no space and none of the callers has to know it is
 * leading. */
static void notation_gap(string_builder *builder) {
    if (builder->length > 0) {
        builder_append(builder, " ");
    }
}

static const char *notation_cover_symbol(const rg_feature_constraint *constraint) {
    size_t i;
    if (constraint->feature == 0 || constraint->value == 0 ||
        strcmp(constraint->value, "+") != 0) {
        return 0;
    }
    for (i = 0; i < sizeof(notation_cover_symbols) / sizeof(notation_cover_symbols[0]); i++) {
        if (strcmp(notation_cover_symbols[i].feature, constraint->feature) == 0) {
            return notation_cover_symbols[i].symbol;
        }
    }
    return 0;
}

/* One constraint, in matrix form.
 *
 * `in_syllable` drops the `syllable_` prefix, so `syllable_shape:open` inside a
 * sigma bracket is `σ⁻[open]` rather than `σ⁻[syllable_shape=open]`. The three
 * syllable dimensions have disjoint value vocabularies -- open/closed,
 * short/long, light/heavy -- so nothing becomes ambiguous by it, and outside a
 * sigma bracket the dimension is written out. */
static void notation_item(
    string_builder *builder,
    const rg_feature_constraint *constraint,
    int in_syllable
) {
    if (constraint->feature == 0 || constraint->value == 0) {
        return;
    }
    /* Stress is a dimension whose two ordinary values have had a notation for a
     * century. `stress=primary` in a matrix would be the one place this
     * notation was less readable than the exact form it is meant to explain. */
    if (strcmp(constraint->feature, "stress") == 0) {
        if (strcmp(constraint->value, "primary") == 0) {
            builder_append(builder, NOT_ACUTE);
            return;
        }
        if (strcmp(constraint->value, "secondary") == 0) {
            builder_append(builder, NOT_GRAVE);
            return;
        }
        builder_appendf(builder, "stress=%s", constraint->value);
        return;
    }
    if (strcmp(constraint->value, "+") == 0) {
        builder_appendf(builder, "+%s", constraint->feature);
        return;
    }
    if (strcmp(constraint->value, "-") == 0) {
        builder_appendf(builder, NOT_MINUS "%s", constraint->feature);
        return;
    }
    if (in_syllable && strncmp(constraint->feature, "syllable_", 9) == 0) {
        builder_append(builder, constraint->value);
        return;
    }
    builder_appendf(builder, "%s=%s", constraint->feature, constraint->value);
}

/* A neighbouring segment: the features required of it and, where the stress
 * slots apply, the stress it bears. Both describe one segment, so they go in
 * one bracket rather than reading as two neighbours.
 *
 * A lone positive feature collapses to its cover symbol, which is the whole
 * reason the symbols exist: `V _ V` and not `[+vowel] _ [+vowel]`. Inside a
 * bracket the symbol is not used -- `[V,+long]` would be one abbreviation
 * beside one expansion of the same kind of thing. */
static void notation_neighbour(
    string_builder *builder,
    const rg_feature_constraint *features,
    size_t feature_count,
    const rg_feature_constraint *stress,
    size_t stress_count
) {
    size_t i;
    size_t written = 0;
    if (feature_count + stress_count == 0) {
        return;
    }
    if (feature_count == 1 && stress_count == 0) {
        const char *symbol = notation_cover_symbol(&features[0]);
        if (symbol != 0) {
            builder_append(builder, symbol);
            return;
        }
    }
    builder_append(builder, "[");
    for (i = 0; i < feature_count; i++) {
        if (written++ > 0) {
            builder_append(builder, ",");
        }
        notation_item(builder, &features[i], 0);
    }
    for (i = 0; i < stress_count; i++) {
        if (written++ > 0) {
            builder_append(builder, ",");
        }
        notation_item(builder, &stress[i], 0);
    }
    builder_append(builder, "]");
}

static void notation_syllable(
    string_builder *builder,
    const char *mark,
    const rg_feature_constraint *features,
    size_t count
) {
    size_t i;
    if (count == 0) {
        return;
    }
    notation_gap(builder);
    builder_append(builder, NOT_SIGMA);
    builder_append(builder, mark);
    builder_append(builder, "[");
    for (i = 0; i < count; i++) {
        if (i > 0) {
            builder_append(builder, ",");
        }
        notation_item(builder, &features[i], 1);
    }
    builder_append(builder, "]");
}

/* Which syllable of the word the target sits in, counted from whichever end
 * names it: σ₁ is the first, σ₋₁ the last, σ₋₂ the penultimate. Any value the
 * syllabifier grows later prints as itself rather than as a wrong number. */
static void notation_syllable_position(string_builder *builder, const char *value) {
    if (strcmp(value, "initial") == 0) {
        builder_append(builder, NOT_SIGMA NOT_SUB_ONE);
    } else if (strcmp(value, "final") == 0) {
        builder_append(builder, NOT_SIGMA NOT_SUB_MINUS NOT_SUB_ONE);
    } else if (strcmp(value, "penultimate") == 0) {
        builder_append(builder, NOT_SIGMA NOT_SUB_MINUS NOT_SUB_TWO);
    } else if (strcmp(value, "antepenultimate") == 0) {
        builder_append(builder, NOT_SIGMA NOT_SUB_MINUS NOT_SUB_THREE);
    } else {
        builder_appendf(builder, NOT_ANGLE_OPEN NOT_SIGMA " %s" NOT_ANGLE_CLOSE, value);
    }
}

/* The target, and everything said about the target itself.
 *
 * Written as `[_ +long]` -- brackets around the underscore -- so that nothing
 * about the target is distinguished from a constraint on the following segment
 * by a space alone. `_ [+long]` and `_[+long]` are different claims and would
 * be one typo apart. */
static void notation_target(string_builder *builder, const rg_context_spec *context) {
    size_t i;
    size_t written = 0;
    int any = context->self_count + context->self_stress_count > 0 ||
              notation_present(context->syllable_role) ||
              notation_present(context->syllable_position) ||
              notation_present(context->morpheme_index);
    notation_gap(builder);
    if (!any) {
        builder_append(builder, "_");
        return;
    }
    builder_append(builder, "[_ ");
    for (i = 0; i < context->self_count; i++) {
        if (written++ > 0) {
            builder_append(builder, ", ");
        }
        notation_item(builder, &context->self[i], 0);
    }
    for (i = 0; i < context->self_stress_count; i++) {
        if (written++ > 0) {
            builder_append(builder, ", ");
        }
        notation_item(builder, &context->self_stress[i], 0);
    }
    if (notation_present(context->syllable_role)) {
        if (written++ > 0) {
            builder_append(builder, ", ");
        }
        builder_appendf(builder, NOT_ANGLE_OPEN "%s" NOT_ANGLE_CLOSE, context->syllable_role);
    }
    if (notation_present(context->syllable_position)) {
        if (written++ > 0) {
            builder_append(builder, ", ");
        }
        notation_syllable_position(builder, context->syllable_position);
    }
    if (notation_present(context->morpheme_index)) {
        if (written++ > 0) {
            builder_append(builder, ", ");
        }
        /* The index saturates at the table's last slot, so the highest one
         * means "this or later" and has to be readable as that. */
        if (strcmp(context->morpheme_index, "7") == 0) {
            builder_append(builder, NOT_ANGLE_OPEN "morpheme 7+" NOT_ANGLE_CLOSE);
        } else {
            builder_appendf(builder, NOT_ANGLE_OPEN "morpheme %s" NOT_ANGLE_CLOSE,
                            context->morpheme_index);
        }
    }
    builder_append(builder, "]");
}

/* An edge, on whichever side of the target it is being claimed for.
 *
 * One system serves the word edge and the morpheme edge alike: the symbol
 * adjacent to the target means the target is *at* that edge, and the symbol
 * with an ellipsis between it and the target means the target is somewhere
 * strictly inside. So `# _` is word-initial, `# … _ … #` is word-internal,
 * `+ _ +` is a whole morpheme, `+ … _ … +` is morpheme-internal. */
static void notation_edge(string_builder *builder, const char *symbol, int adjacent, int leading) {
    notation_gap(builder);
    if (leading) {
        builder_append(builder, symbol);
        if (!adjacent) {
            builder_append(builder, " " NOT_ELLIPSIS);
        }
    } else {
        if (!adjacent) {
            builder_append(builder, NOT_ELLIPSIS " ");
        }
        builder_append(builder, symbol);
    }
}

static void notation_left_edges(string_builder *builder, const rg_context_spec *context) {
    if (notation_present(context->position)) {
        if (strcmp(context->position, "initial") == 0) {
            notation_edge(builder, "#", 1, 1);
        } else if (strcmp(context->position, "medial") == 0) {
            notation_edge(builder, "#", 0, 1);
        }
    }
    if (notation_present(context->morphological)) {
        if (strcmp(context->morphological, "initial") == 0 ||
            strcmp(context->morphological, "only") == 0) {
            notation_edge(builder, "+", 1, 1);
        } else if (strcmp(context->morphological, "internal") == 0) {
            notation_edge(builder, "+", 0, 1);
        }
    }
}

static void notation_right_edges(string_builder *builder, const rg_context_spec *context) {
    if (notation_present(context->morphological)) {
        if (strcmp(context->morphological, "final") == 0 ||
            strcmp(context->morphological, "only") == 0) {
            notation_edge(builder, "+", 1, 0);
        } else if (strcmp(context->morphological, "internal") == 0) {
            notation_edge(builder, "+", 0, 0);
        }
    }
    if (notation_present(context->position)) {
        if (strcmp(context->position, "final") == 0) {
            notation_edge(builder, "#", 1, 0);
        } else if (strcmp(context->position, "medial") == 0) {
            notation_edge(builder, "#", 0, 0);
        }
    }
}

/* The counted-distance slot, as one skeleton rather than one claim per
 * constraint.
 *
 * Each position from the farthest constraint in to the target is written out:
 * the constraint that sits there if there is one, an unspecified segment if
 * there is not. Two before alone is `[+coronal] · _`; two and three together
 * are `[+open] [+coronal] _`, which is the sequence the two constraints
 * actually describe. Emitting each constraint with its own run of dots instead
 * would put the nearer one further out than the farther one and read as four
 * segments where the rule names two.
 *
 * Spelling the skipped positions rather than subscripting the distance keeps
 * them countable, which is what a reader checking the claim against a wordlist
 * does. */
static void notation_distance(
    string_builder *builder,
    const rg_distance_constraint *constraints,
    size_t count,
    int leading
) {
    int furthest = 0;
    int step;
    size_t i;
    for (i = 0; i < count; i++) {
        if (constraints[i].offset > furthest) {
            furthest = constraints[i].offset;
        }
    }
    for (step = 0; step < furthest; step++) {
        /* Outermost first on the left, innermost first on the right. */
        int at = leading ? furthest - step : step + 1;
        const rg_feature_constraint *here = 0;
        for (i = 0; i < count; i++) {
            if (constraints[i].offset == at) {
                here = &constraints[i].constraint;
            }
        }
        notation_gap(builder);
        if (here != 0) {
            notation_neighbour(builder, here, 1, 0, 0);
        } else {
            builder_append(builder, NOT_MIDDOT);
        }
    }
}

/* "Somewhere on this side", one group however many constraints it holds.
 *
 * Each constraint is an independent existential: `somewhere_preceding` with
 * two of them asks for a segment before carrying one and a segment before
 * carrying the other, which may be the same segment or may not. Written as
 * `[+f] … [+g] … _` that would read as a sequence, and as `[+f,+g] … _` it
 * would read as one segment carrying both. Neither is what the model states,
 * so the conjunction is explicit and nothing about order or identity is
 * claimed. Every rule discovered in the corpora carries exactly one; the `&`
 * is here so that the day one carries two, the line says what was found. */
static void notation_somewhere(
    string_builder *builder,
    const rg_feature_constraint *constraints,
    size_t count,
    int leading
) {
    size_t i;
    if (count == 0) {
        return;
    }
    notation_gap(builder);
    if (!leading) {
        builder_append(builder, NOT_ELLIPSIS " ");
    }
    for (i = 0; i < count; i++) {
        if (i > 0) {
            builder_append(builder, " & ");
        }
        notation_neighbour(builder, &constraints[i], 1, 0, 0);
    }
    if (leading) {
        builder_append(builder, " " NOT_ELLIPSIS);
    }
}

char *rg_notation_frame_internal(const rg_context_spec *context) {
    string_builder builder;
    builder_init(&builder);
    if (context == 0 || rg_context_spec_constraint_count(context) == 0) {
        return builder_finish(&builder);
    }

    /* Left to right, outermost first: word edge, morpheme edge, the previous
     * syllable, anything anywhere before, anything a counted distance before,
     * the immediate neighbour, then the target. */
    notation_left_edges(&builder, context);
    notation_syllable(&builder, NOT_SUP_MINUS, context->previous_syllable,
                      context->previous_syllable_count);
    notation_somewhere(&builder, context->somewhere_preceding,
                       context->somewhere_preceding_count, 1);
    notation_distance(&builder, context->preceding_at_distance,
                      context->preceding_at_distance_count, 1);
    if (context->preceding_count + context->preceding_stress_count > 0) {
        notation_gap(&builder);
        notation_neighbour(&builder, context->preceding, context->preceding_count,
                           context->preceding_stress, context->preceding_stress_count);
    }

    notation_target(&builder, context);

    if (context->following_count + context->following_stress_count > 0) {
        notation_gap(&builder);
        notation_neighbour(&builder, context->following, context->following_count,
                           context->following_stress, context->following_stress_count);
    }
    notation_distance(&builder, context->following_at_distance,
                      context->following_at_distance_count, 0);
    notation_somewhere(&builder, context->somewhere_following,
                       context->somewhere_following_count, 0);
    notation_syllable(&builder, NOT_SUP_EQUALS, context->same_syllable,
                      context->same_syllable_count);
    notation_syllable(&builder, NOT_SUP_PLUS, context->next_syllable,
                      context->next_syllable_count);
    notation_right_edges(&builder, context);

    return builder_finish(&builder);
}

static void notation_suprasegmentals(string_builder *builder, const rg_suprasegmentals *s) {
    size_t written = 0;
    if (s == 0) {
        return;
    }
    if (notation_present(s->tone)) {
        builder_appendf(builder, "%stone=%s", written++ == 0 ? "[" : ",", s->tone);
    }
    if (notation_present(s->length)) {
        builder_appendf(builder, "%slength=%s", written++ == 0 ? "[" : ",", s->length);
    }
    if (notation_present(s->stress)) {
        builder_appendf(builder, "%sstress=%s", written++ == 0 ? "[" : ",", s->stress);
    }
    if (written > 0) {
        builder_append(builder, "]");
    }
}

/* One side's outcome. Braces mark a set and are dropped for a single grapheme,
 * as Index Diachronica writes `s → r` but `{m,n} → ŋ`: on the many rows whose
 * outcome is one segment they are noise. */
static void notation_side(string_builder *builder, const rg_notation_side *side) {
    size_t i;
    if (side->grapheme_count == 0) {
        builder_append(builder, RG_GAP_GRAPHEME);
    } else if (side->grapheme_count == 1) {
        builder_append(builder, side->graphemes[0]);
    } else {
        builder_append(builder, "{");
        for (i = 0; i < side->grapheme_count; i++) {
            if (i > 0) {
                builder_append(builder, ",");
            }
            builder_append(builder, side->graphemes[i]);
        }
        builder_append(builder, "}");
    }
    notation_suprasegmentals(builder, side->suprasegmentals);
}

/* The frames, attributed.
 *
 * A label is dropped only when *every* side states a frame and all of them are
 * the same: the environment holds in each lect and repeating it would read as
 * several separate findings. Every other case keeps its labels, and the
 * demanding half of that rule is the one about sides that state nothing.
 * `ancestor` and `conservative` both condition on a following close vowel
 * where `innovator` does not, and an unlabelled `p ~ p ~ f / _ [+close]` says
 * the frame holds in the innovator too, which is the opposite of what was
 * found. A single conditioning side likewise keeps its label: which lect's
 * word the frame is read in is what the reader cannot otherwise know, and it
 * is a claim about which lect kept the conditioning contrast rather than a
 * formatting detail. */
static void notation_frames(
    string_builder *builder,
    const rg_notation_side *sides,
    size_t count,
    char **frames
) {
    size_t i;
    size_t stated = 0;
    size_t first = 0;
    int all_alike = 1;
    for (i = 0; i < count; i++) {
        if (frames[i] != 0 && frames[i][0] != '\0') {
            if (stated == 0) {
                first = i;
            } else if (strcmp(frames[i], frames[first]) != 0) {
                all_alike = 0;
            }
            stated++;
        }
    }
    if (stated == 0) {
        return;
    }
    builder_append(builder, "  /  ");
    if (stated == count && stated > 1 && all_alike) {
        builder_append(builder, frames[first]);
        return;
    }
    stated = 0;
    for (i = 0; i < count; i++) {
        if (frames[i] == 0 || frames[i][0] == '\0') {
            continue;
        }
        if (stated++ > 0) {
            builder_append(builder, " ;  ");
        }
        builder_appendf(builder, "%s: %s", sides[i].lect == 0 ? "?" : sides[i].lect, frames[i]);
    }
}

char *rg_notation_line_internal(const rg_notation_side *sides, size_t count) {
    string_builder builder;
    char **frames;
    size_t i;
    if (sides == 0 || count == 0) {
        return 0;
    }
    frames = (char **)calloc(count, sizeof(*frames));
    if (frames == 0) {
        return 0;
    }
    builder_init(&builder);
    for (i = 0; i < count; i++) {
        if (i > 0) {
            builder_append(&builder, "  ~  ");
        }
        notation_side(&builder, &sides[i]);
        frames[i] = rg_notation_frame_internal(sides[i].context);
    }
    notation_frames(&builder, sides, count, frames);
    for (i = 0; i < count; i++) {
        free(frames[i]);
    }
    free(frames);
    return builder_finish(&builder);
}

char *rg_notation_cross_dimensional_internal(
    const char *conditioned_lect,
    const char *dimension,
    const char *value,
    const char *environment_lect,
    const rg_context_spec *environment
) {
    string_builder builder;
    char *frame;
    builder_init(&builder);
    /* Not written with `~`: there is no second side. One lect carries a value
     * where another lect's environment holds, which is a different shape of
     * claim from a correspondence and must not borrow its symbol. */
    builder_appendf(&builder, "%s [%s=%s]",
                    conditioned_lect == 0 ? "?" : conditioned_lect,
                    dimension == 0 ? "?" : dimension,
                    value == 0 ? "?" : value);
    frame = rg_notation_frame_internal(environment);
    if (frame != 0 && frame[0] != '\0') {
        /* A lect-internal rule reads its environment in the same lect that
         * carries the value, and that lect is already named at the front of the
         * line. Repeating it there would make a self rule look cross-lect, and
         * an unlabelled frame after a named lect can only be that lect's. */
        if (environment_lect != 0 && conditioned_lect != 0 &&
            strcmp(environment_lect, conditioned_lect) == 0) {
            builder_appendf(&builder, "  /  %s", frame);
        } else {
            builder_appendf(&builder, "  /  %s: %s",
                            environment_lect == 0 ? "?" : environment_lect, frame);
        }
    }
    free(frame);
    return builder_finish(&builder);
}
