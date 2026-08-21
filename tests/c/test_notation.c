#include "notation.h"
#include "environment.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The notation renderer, slot by slot, against docs/NOTATION.md.
 *
 * Two things are being guarded here and they are different. One is that each
 * slot renders to the symbol the specification says it does, which is what the
 * per-slot cases below check. The other is that no slot renders to *nothing*:
 * the renderer enumerates the eighteen slots by hand -- it has to, since each
 * one is written in a different place in the frame -- so a slot could be added
 * to RG_ENV_SLOTS and silently dropped from every notation line in the
 * library. The census in notation.c catches that at compile time and
 * `every_slot_says_something` catches it here, and neither is sufficient
 * alone: the census only proves an enumerator exists, and a slot the renderer
 * enumerates but never writes would still pass it.
 */

static void expect(const char *what, const char *got, const char *want) {
    if (got == 0 || strcmp(got, want) != 0) {
        fprintf(stderr, "%s: got \"%s\", want \"%s\"\n", what, got == 0 ? "(null)" : got, want);
        assert(0);
    }
}

static char *frame_of(const rg_context_spec *context) {
    char *out = rg_notation_frame_internal(context);
    assert(out != 0);
    return out;
}

static void check_frame(const char *what, const rg_context_spec *context, const char *want) {
    char *got = frame_of(context);
    expect(what, got, want);
    free(got);
}

static void test_neighbours(void) {
    rg_context_spec c;
    rg_feature_constraint vowel[] = {{"vowel", "+"}};
    rg_feature_constraint front[] = {{"front", "+"}};
    rg_feature_constraint pair[] = {{"front", "+"}, {"open", "+"}};
    rg_distance_constraint two[] = {{2, {"coronal", "+"}}};
    rg_distance_constraint three[] = {{3, {"open", "+"}}};

    rg_context_spec_init_empty(&c);
    c.preceding = vowel;
    c.preceding_count = 1;
    check_frame("preceding", &c, "V _");

    rg_context_spec_init_empty(&c);
    c.following = front;
    c.following_count = 1;
    check_frame("following", &c, "_ E");

    /* Both neighbours, and the shape the whole notation exists for. */
    rg_context_spec_init_empty(&c);
    c.preceding = vowel;
    c.preceding_count = 1;
    c.following = vowel;
    c.following_count = 1;
    check_frame("intervocalic", &c, "V _ V");

    /* Two constraints on one segment are one bracket, not two neighbours. */
    rg_context_spec_init_empty(&c);
    c.following = pair;
    c.following_count = 2;
    check_frame("conjunction", &c, "_ [+front,+open]");

    /* One unspecified segment per step, so the distance stays countable. */
    rg_context_spec_init_empty(&c);
    c.preceding_at_distance = two;
    c.preceding_at_distance_count = 1;
    check_frame("two before", &c, "[+coronal] \xc2\xb7 _");

    rg_context_spec_init_empty(&c);
    c.following_at_distance = three;
    c.following_at_distance_count = 1;
    check_frame("three after", &c, "_ \xc2\xb7 \xc2\xb7 [+open]");

    rg_context_spec_init_empty(&c);
    c.somewhere_preceding = front;
    c.somewhere_preceding_count = 1;
    check_frame("somewhere before", &c, "E \xe2\x80\xa6 _");

    rg_context_spec_init_empty(&c);
    c.somewhere_following = front;
    c.somewhere_following_count = 1;
    check_frame("somewhere after", &c, "_ \xe2\x80\xa6 E");

    /* Two constraints at two distances are one skeleton, in the order they sit
     * in the word, and the position neither of them names is still a position:
     * offset 1 is the immediate neighbour, which this slot does not constrain,
     * so it stays a dot. A run of dots per constraint would instead put the
     * nearer constraint further out than the farther one and read as five
     * segments where the rule names three. */
    {
        rg_distance_constraint both[] = {{2, {"coronal", "+"}}, {3, {"open", "+"}}};
        rg_context_spec_init_empty(&c);
        c.preceding_at_distance = both;
        c.preceding_at_distance_count = 2;
        check_frame("two distances, one skeleton", &c, "[+open] [+coronal] \xc2\xb7 _");
        rg_context_spec_init_empty(&c);
        c.following_at_distance = both;
        c.following_at_distance_count = 2;
        check_frame("two distances after", &c, "_ \xc2\xb7 [+coronal] [+open]");
    }

    /* Two existentials on one side are a conjunction and nothing more. Written
     * `E … [+open] … _` it would claim an order; written `[+front,+open] … _`
     * it would claim one segment carries both. The model states neither. */
    {
        rg_feature_constraint existentials[] = {{"front", "+"}, {"open", "+"}};
        rg_context_spec_init_empty(&c);
        c.somewhere_preceding = existentials;
        c.somewhere_preceding_count = 2;
        check_frame("two existentials before", &c, "E & [+open] \xe2\x80\xa6 _");
        rg_context_spec_init_empty(&c);
        c.somewhere_following = existentials;
        c.somewhere_following_count = 2;
        check_frame("two existentials after", &c, "_ \xe2\x80\xa6 E & [+open]");
    }
}

static void test_cover_symbols(void) {
    rg_context_spec c;
    rg_feature_constraint vowel[] = {{"vowel", "+"}};
    rg_feature_constraint not_vowel[] = {{"vowel", "-"}};
    rg_feature_constraint sibilant[] = {{"sibilant", "+"}};
    rg_feature_constraint stop_and_long[] = {{"stop", "+"}, {"long", "+"}};

    rg_context_spec_init_empty(&c);
    c.preceding = vowel;
    c.preceding_count = 1;
    check_frame("cover symbol", &c, "V _");

    /* A cover symbol abbreviates a feature's *presence*. There is no symbol
     * for its absence, and reusing V for one would make the two readings one
     * character apart. */
    rg_context_spec_init_empty(&c);
    c.preceding = not_vowel;
    c.preceding_count = 1;
    check_frame("negation is never a symbol", &c, "[\xe2\x88\x92vowel] _");

    /* merkmal has features the restricted set has no symbol for; they render
     * as matrices rather than borrowing a symbol that means something else. */
    rg_context_spec_init_empty(&c);
    c.preceding = sibilant;
    c.preceding_count = 1;
    check_frame("no symbol for sibilant", &c, "[+sibilant] _");

    /* Inside a bracket the symbol is not used: one abbreviation beside one
     * expansion of the same kind of thing reads as two different claims. */
    rg_context_spec_init_empty(&c);
    c.preceding = stop_and_long;
    c.preceding_count = 2;
    check_frame("no symbols inside a bracket", &c, "[+stop,+long] _");
}

static void test_syllables(void) {
    rg_context_spec c;
    rg_feature_constraint shape[] = {{"syllable_shape", "open"}};
    rg_feature_constraint stop[] = {{"stop", "+"}};
    rg_feature_constraint nucleus[] = {{"syllable_nucleus", "short"}};

    rg_context_spec_init_empty(&c);
    c.previous_syllable = shape;
    c.previous_syllable_count = 1;
    check_frame("previous syllable", &c, "\xcf\x83\xe2\x81\xbb[open] _");

    rg_context_spec_init_empty(&c);
    c.next_syllable = nucleus;
    c.next_syllable_count = 1;
    check_frame("next syllable", &c, "_ \xcf\x83\xe2\x81\xba[short]");

    rg_context_spec_init_empty(&c);
    c.same_syllable = stop;
    c.same_syllable_count = 1;
    check_frame("same syllable", &c, "_ \xcf\x83\xe2\x81\xbc[+stop]");

    /* The dimension prefix goes only inside a sigma bracket, where the three
     * syllable dimensions have disjoint vocabularies. Elsewhere it is spelled
     * out, since nothing else guarantees the value is unambiguous. */
    rg_context_spec_init_empty(&c);
    c.following = shape;
    c.following_count = 1;
    check_frame("dimension outside a sigma", &c, "_ [syllable_shape=open]");
}

static void test_target(void) {
    rg_context_spec c;
    rg_feature_constraint self[] = {{"long", "+"}};
    rg_feature_constraint stress[] = {{"stress", "primary"}};

    rg_context_spec_init_empty(&c);
    c.self = self;
    c.self_count = 1;
    check_frame("the target itself", &c, "[_ +long]");

    rg_context_spec_init_empty(&c);
    c.self_stress = stress;
    c.self_stress_count = 1;
    check_frame("the target's stress", &c, "[_ \xc2\xb4]");

    rg_context_spec_init_empty(&c);
    c.syllable_role = "coda";
    check_frame("syllable role", &c, "[_ \xe2\x9f\xa8" "coda\xe2\x9f\xa9]");

    rg_context_spec_init_empty(&c);
    c.syllable_position = "penultimate";
    check_frame("syllable position", &c,
                "[_ \xcf\x83\xe2\x82\x8b\xe2\x82\x82]");

    rg_context_spec_init_empty(&c);
    c.syllable_position = "initial";
    check_frame("first syllable", &c, "[_ \xcf\x83\xe2\x82\x81]");

    rg_context_spec_init_empty(&c);
    c.morpheme_index = "2";
    check_frame("morpheme index", &c, "[_ \xe2\x9f\xa8morpheme 2\xe2\x9f\xa9]");

    /* The counter saturates, so the last slot means "this or later" and has to
     * read as that rather than as an exact seventh morpheme. */
    rg_context_spec_init_empty(&c);
    c.morpheme_index = "7";
    check_frame("saturated morpheme index", &c, "[_ \xe2\x9f\xa8morpheme 7+\xe2\x9f\xa9]");

    /* Several things about the target are one bracket. */
    rg_context_spec_init_empty(&c);
    c.self = self;
    c.self_count = 1;
    c.syllable_role = "coda";
    check_frame("target, twice over", &c, "[_ +long, \xe2\x9f\xa8" "coda\xe2\x9f\xa9]");
}

static void test_stress(void) {
    rg_context_spec c;
    rg_feature_constraint primary[] = {{"stress", "primary"}};
    rg_feature_constraint secondary[] = {{"stress", "secondary"}};
    rg_feature_constraint vowel[] = {{"vowel", "+"}};

    rg_context_spec_init_empty(&c);
    c.preceding_stress = primary;
    c.preceding_stress_count = 1;
    check_frame("stress before", &c, "[\xc2\xb4] _");

    rg_context_spec_init_empty(&c);
    c.following_stress = secondary;
    c.following_stress_count = 1;
    check_frame("stress after", &c, "_ [\xcb\x8b]");

    /* The stress slot and the feature slot describe the same segment, so they
     * are one bracket. Two would read as two neighbours. */
    rg_context_spec_init_empty(&c);
    c.following = vowel;
    c.following_count = 1;
    c.following_stress = primary;
    c.following_stress_count = 1;
    check_frame("a stressed vowel is one neighbour", &c, "_ [+vowel,\xc2\xb4]");
}

static void test_edges(void) {
    rg_context_spec c;

    rg_context_spec_init_empty(&c);
    c.position = "initial";
    check_frame("word-initial", &c, "# _");

    rg_context_spec_init_empty(&c);
    c.position = "final";
    check_frame("word-final", &c, "_ #");

    rg_context_spec_init_empty(&c);
    c.position = "medial";
    check_frame("word-internal", &c, "# \xe2\x80\xa6 _ \xe2\x80\xa6 #");

    rg_context_spec_init_empty(&c);
    c.morphological = "initial";
    check_frame("morpheme-initial", &c, "+ _");

    rg_context_spec_init_empty(&c);
    c.morphological = "final";
    check_frame("morpheme-final", &c, "_ +");

    rg_context_spec_init_empty(&c);
    c.morphological = "only";
    check_frame("a whole morpheme", &c, "+ _ +");

    rg_context_spec_init_empty(&c);
    c.morphological = "internal";
    check_frame("morpheme-internal", &c, "+ \xe2\x80\xa6 _ \xe2\x80\xa6 +");

    /* The word edge is outside the morpheme edge. */
    rg_context_spec_init_empty(&c);
    c.position = "initial";
    c.morphological = "initial";
    check_frame("both edges", &c, "# + _");
}

static void test_empty(void) {
    rg_context_spec c;
    char *got;
    rg_context_spec_init_empty(&c);
    /* An unconditioned environment is the empty string, not a bare underscore:
     * a caller tests the result rather than counting constraints itself, and
     * "_" would be a frame that says nothing while looking like a claim. */
    got = frame_of(&c);
    expect("unconditioned", got, "");
    free(got);
    got = rg_notation_frame_internal(0);
    expect("no environment at all", got, "");
    free(got);
}

static void test_lines(void) {
    rg_context_spec intervocalic;
    rg_context_spec previous;
    rg_context_spec empty;
    rg_feature_constraint vowel[] = {{"vowel", "+"}};
    rg_feature_constraint shape[] = {{"syllable_shape", "open"}};
    const char *r[] = {"r"};
    const char *s[] = {"s"};
    const char *fricatives[] = {"f", "x", "\xce\xb8"};
    const char *stops[] = {"k", "p", "t"};
    const char *gap[] = {RG_GAP_GRAPHEME};
    rg_notation_side sides[3];
    char *line;

    rg_context_spec_init_empty(&intervocalic);
    intervocalic.preceding = vowel;
    intervocalic.preceding_count = 1;
    intervocalic.following = vowel;
    intervocalic.following_count = 1;
    rg_context_spec_init_empty(&previous);
    previous.previous_syllable = shape;
    previous.previous_syllable_count = 1;
    rg_context_spec_init_empty(&empty);

    memset(sides, 0, sizeof(sides));
    sides[0].lect = "latin";
    sides[0].graphemes = r;
    sides[0].grapheme_count = 1;
    sides[0].context = &empty;
    sides[1].lect = "old_latin";
    sides[1].graphemes = s;
    sides[1].grapheme_count = 1;
    sides[1].context = &intervocalic;
    line = rg_notation_line_internal(sides, 2);
    /* One side conditions, and it keeps its label: which lect's word the frame
     * is read in is the finding, not a formatting detail. */
    expect("one-sided", line, "r  ~  s  /  old_latin: V _ V");
    free(line);

    sides[0].context = &previous;
    line = rg_notation_line_internal(sides, 2);
    expect("two-sided", line,
           "r  ~  s  /  latin: \xcf\x83\xe2\x81\xbb[open] _ ;  old_latin: V _ V");
    free(line);

    /* Both sides state the same frame: one finding, written once. */
    sides[0].context = &intervocalic;
    line = rg_notation_line_internal(sides, 2);
    expect("shared frame", line, "r  ~  s  /  V _ V");
    free(line);

    /* But not when a side states nothing. An unlabelled frame over three sides
     * claims it holds in all three, and here it does not hold in the third. */
    sides[2].lect = "innovator";
    sides[2].graphemes = s;
    sides[2].grapheme_count = 1;
    sides[2].context = &empty;
    line = rg_notation_line_internal(sides, 3);
    expect("a silent side keeps every label", line,
           "r  ~  s  ~  s  /  latin: V _ V ;  old_latin: V _ V");
    free(line);

    /* Braces mark a set and are dropped for a single grapheme; the gap is the
     * grapheme it already is elsewhere in the library. */
    memset(sides, 0, sizeof(sides));
    sides[0].lect = "daughter";
    sides[0].graphemes = fricatives;
    sides[0].grapheme_count = 3;
    sides[0].context = &intervocalic;
    sides[1].lect = "proto";
    sides[1].graphemes = stops;
    sides[1].grapheme_count = 3;
    sides[1].context = &empty;
    line = rg_notation_line_internal(sides, 2);
    expect("sets", line, "{f,x,\xce\xb8}  ~  {k,p,t}  /  daughter: V _ V");
    free(line);

    sides[1].graphemes = gap;
    sides[1].grapheme_count = 1;
    line = rg_notation_line_internal(sides, 2);
    expect("a correspondence to nothing", line,
           "{f,x,\xce\xb8}  ~  \xe2\x88\x85  /  daughter: V _ V");
    free(line);
}

static void test_cross_dimensional(void) {
    rg_context_spec voiced;
    rg_feature_constraint constraint[] = {{"voiced", "+"}};
    char *line;

    rg_context_spec_init_empty(&voiced);
    voiced.preceding = constraint;
    voiced.preceding_count = 1;

    /* Written without `~`: there is no second side. One lect carries a value
     * where another lect's environment holds, and borrowing the correspondence
     * symbol for that would make it read as a correspondence. */
    line = rg_notation_cross_dimensional_internal("daughter", "tone", "11", "proto", &voiced);
    expect("cross-lect", line, "daughter [tone=11]  /  proto: [+voiced] _");
    free(line);

    /* Lect-internal: the lect is already named at the front, so an unlabelled
     * frame can only be its own, and repeating it would make a self rule look
     * cross-lect. */
    line = rg_notation_cross_dimensional_internal("daughter", "tone", "11", "daughter", &voiced);
    expect("lect-internal", line, "daughter [tone=11]  /  [+voiced] _");
    free(line);
}

/* Every slot in RG_ENV_SLOTS and RG_ENV_STRING_SLOTS renders to something.
 *
 * The census half of this lives in notation.c and is a compile-time check that
 * the renderer has an enumerator for each slot. This is the other half: a slot
 * the renderer enumerates but never writes out would pass the census and drop
 * silently from every notation line in the library. Values are chosen to be
 * plausible for the slot rather than realistic; what is being asserted is that
 * something comes out, not what. */
static void test_every_slot_says_something(void) {
    int seen[64];
    size_t total = rg_notation_slot_census_internal(seen, sizeof(seen) / sizeof(seen[0]));
    rg_feature_constraint feature[] = {{"vowel", "+"}};
    rg_feature_constraint stress[] = {{"stress", "primary"}};
    rg_distance_constraint distance[] = {{2, {"vowel", "+"}}};
    size_t i;
    size_t checked = 0;

    assert(total > 0);
    for (i = 0; i < total; i++) {
        assert(seen[i] == 1);
    }

#define FEATURE_SLOT(name, label, key)                                          \
    {                                                                           \
        rg_context_spec c;                                                      \
        char *got;                                                              \
        rg_context_spec_init_empty(&c);                                         \
        c.name = strcmp(#name, "self_stress") == 0 ||                           \
                 strcmp(#name, "preceding_stress") == 0 ||                      \
                 strcmp(#name, "following_stress") == 0 ? stress : feature;     \
        c.name##_count = 1;                                                     \
        got = frame_of(&c);                                                     \
        if (got[0] == '\0') {                                                   \
            fprintf(stderr, "slot %s renders to nothing\n", #name);             \
            assert(0);                                                          \
        }                                                                       \
        free(got);                                                              \
        checked++;                                                              \
    }
#define DISTANCE_SLOT(name, label, key)                                         \
    {                                                                           \
        rg_context_spec c;                                                      \
        char *got;                                                              \
        rg_context_spec_init_empty(&c);                                         \
        c.name = distance;                                                      \
        c.name##_count = 1;                                                     \
        got = frame_of(&c);                                                     \
        if (got[0] == '\0') {                                                   \
            fprintf(stderr, "slot %s renders to nothing\n", #name);             \
            assert(0);                                                          \
        }                                                                       \
        free(got);                                                              \
        checked++;                                                              \
    }
    RG_ENV_SLOTS(FEATURE_SLOT, DISTANCE_SLOT)
#undef FEATURE_SLOT
#undef DISTANCE_SLOT

    /* The string slots take a value from their own vocabulary; "initial" is in
     * all of position's, morphological's and syllable_position's, and the two
     * that would not accept it are given something they would. */
#define STRING_SLOT(name, key)                                                  \
    {                                                                           \
        rg_context_spec c;                                                      \
        char *got;                                                              \
        rg_context_spec_init_empty(&c);                                         \
        c.name = strcmp(#name, "syllable_role") == 0 ? "onset"                  \
               : strcmp(#name, "morpheme_index") == 0 ? "1" : "initial";        \
        got = frame_of(&c);                                                     \
        if (got[0] == '\0') {                                                   \
            fprintf(stderr, "slot %s renders to nothing\n", #name);             \
            assert(0);                                                          \
        }                                                                       \
        free(got);                                                              \
        checked++;                                                              \
    }
    RG_ENV_STRING_SLOTS(STRING_SLOT)
#undef STRING_SLOT

    assert(checked == total);
}

/* The specification and the renderer cannot fall out of step.
 *
 * Every symbol this file can emit has to appear in docs/NOTATION.md. A symbol
 * the code produces and the key does not explain is worse than no notation:
 * the reader has something that looks like a convention and is not one. */
static void test_the_key_explains_every_symbol(void) {
    static const char *const symbols[] = {
        "~", "/", ";", "_", "\xe2\x80\xa6", "\xc2\xb7", "\xe2\x88\x85",
        "\xcf\x83\xe2\x81\xbb", "\xcf\x83\xe2\x81\xba", "\xcf\x83\xe2\x81\xbc",
        "\xcf\x83\xe2\x82\x81", "\xcf\x83\xe2\x82\x8b\xe2\x82\x81",
        "\xcf\x83\xe2\x82\x8b\xe2\x82\x82", "\xcf\x83\xe2\x82\x8b\xe2\x82\x83",
        /* The two syllable roles whose names begin with a hex digit are written
         * from their closing bracket, not their opening one: `\xa8` followed by
         * `c` or `a` is one escape and not two, and splitting the literal to
         * say otherwise reads as a missing comma in a list of them. */
        "\xe2\x9f\xa8onset\xe2\x9f\xa9", "\xe2\x9f\xa8nucleus\xe2\x9f\xa9",
        "coda\xe2\x9f\xa9", "ambisyllabic\xe2\x9f\xa9",
        "\xe2\x9f\xa8morpheme", "\xc2\xb4", "\xcb\x8b", "\xe2\x88\x92",
        "# _", "_ #", "+ _", "_ +", "+ _ +",
        "V", "C", "N", "F", "S", "A", "O", "R", "J", "Z", "P", "K", "E", "B"
    };
    char path[1024];
    char *text;
    long size;
    size_t i;
    FILE *file;

    snprintf(path, sizeof(path), "%s/docs/NOTATION.md", REGULAE_SOURCE_DIR);
    file = fopen(path, "rb");
    assert(file != 0);
    assert(fseek(file, 0, SEEK_END) == 0);
    size = ftell(file);
    assert(size > 0);
    assert(fseek(file, 0, SEEK_SET) == 0);
    text = (char *)malloc((size_t)size + 1);
    assert(text != 0);
    assert(fread(text, 1, (size_t)size, file) == (size_t)size);
    text[size] = '\0';
    fclose(file);

    for (i = 0; i < sizeof(symbols) / sizeof(symbols[0]); i++) {
        if (strstr(text, symbols[i]) == 0) {
            fprintf(stderr, "docs/NOTATION.md does not explain \"%s\"\n", symbols[i]);
            assert(0);
        }
    }
    free(text);
}

int main(void) {
    test_neighbours();
    test_cover_symbols();
    test_syllables();
    test_target();
    test_stress();
    test_edges();
    test_empty();
    test_lines();
    test_cross_dimensional();
    test_every_slot_says_something();
    test_the_key_explains_every_symbol();
    return 0;
}
