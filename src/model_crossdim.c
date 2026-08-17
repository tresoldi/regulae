#include "model_internal.h"
#include "split_score.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Whether the position an environment names exists at all. "The preceding
 * segment is not voiced" is a claim about a preceding segment, and word-initial
 * position is not a voiceless onset: counting the edge as the negative side
 * merges a positional environment into a featural one, and a rule stated over
 * that union cannot be read as either. Positions off the end of the form are
 * excluded from both sides of the contrast rather than assigned to one. */
static int context_position_exists(const rg_form *form, int index) {
    return form != 0 && index >= 0 && (size_t)index < form->segment_count;
}

static int segment_has_context_feature(const rg_context *ctx, const rg_form *form, int index, const char *feature) {
    const rg_feature_set *features = 0;
    rg_status status;
    int found = 0;
    if (ctx == 0 || form == 0 || feature == 0 || index < 0 || (size_t)index >= form->segment_count) {
        return 0;
    }
    if (form->segments[index].grapheme == 0) {
        return 0;
    }
    status = rg_context_features_internal(ctx, form->segments[index].grapheme, &features);
    if (status != RG_OK) {
        return 0;
    }
    {
        size_t i;
        for (i = 0; i < rg_feature_set_size(features); i++) {
            const char *item = rg_feature_set_get(features, i);
            if (item != 0 && strcmp(item, feature) == 0) {
                found = 1;
                break;
            }
        }
    }
    return found;
}

typedef struct tone_mass {
    char *tone;
    double count;
} tone_mass;

static void tone_masses_clear(tone_mass *items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        free(items[i].tone);
    }
    free(items);
}

static rg_status add_tone_mass(tone_mass **items, size_t *count, size_t *cap, const char *tone, double weight) {
    size_t i;
    tone_mass *next;
    if (tone == 0 || tone[0] == '\0') {
        return RG_OK;
    }
    for (i = 0; i < *count; i++) {
        if (strcmp((*items)[i].tone, tone) == 0) {
            (*items)[i].count += weight;
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 4 : *cap * 2;
        next = (tone_mass *)realloc(*items, next_cap * sizeof(**items));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[*count].tone = rg_strdup_internal(tone);
    if ((*items)[*count].tone == 0) {
        return RG_ERR_OOM;
    }
    (*items)[*count].count = weight;
    (*count)++;
    return RG_OK;
}

static rg_status append_cross_dimensional_row(
    rg_cross_dimensional_row **rows,
    size_t *count,
    size_t *cap,
    const rg_context_spec *environment,
    int context_is_target,
    int dimension_from_environment,
    const char *dimension,
    const char *value,
    int position_offset,
    double rule_count,
    double source_count,
    double contrast_count,
    double contrast_source_count,
    double delta_bic,
    int decision_index,
    double search_margin
) {
    rg_cross_dimensional_row *next;
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 8 : *cap * 2;
        next = (rg_cross_dimensional_row *)realloc(*rows, next_cap * sizeof(**rows));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *rows = next;
        *cap = next_cap;
    }
    memset(&(*rows)[*count], 0, sizeof((*rows)[*count]));
    if (rg_context_spec_copy_internal(environment, &(*rows)[*count].environment) != RG_OK) {
        return RG_ERR_OOM;
    }
    (*rows)[*count].context_is_target = context_is_target;
    (*rows)[*count].dimension_from_environment = dimension_from_environment;
    (*rows)[*count].dimension = rg_strdup_internal(dimension);
    (*rows)[*count].value = rg_strdup_internal(value);
    (*rows)[*count].position_offset = position_offset;
    (*rows)[*count].count = rule_count;
    (*rows)[*count].source_count = source_count;
    (*rows)[*count].confidence = source_count > 0.0 ? rule_count / source_count : 0.0;
    (*rows)[*count].contrast_count = contrast_count;
    (*rows)[*count].contrast_source_count = contrast_source_count;
    (*rows)[*count].contrast_confidence =
        contrast_source_count > 0.0 ? contrast_count / contrast_source_count : 0.0;
    (*rows)[*count].evidence.delta_bic = delta_bic;
    (*rows)[*count].evidence.decision_index = decision_index;
    (*rows)[*count].evidence.search_margin = search_margin;
    (*rows)[*count].uncertainty = rg_wilson_default_internal(rule_count, source_count);
    if ((*rows)[*count].dimension == 0 ||
        (*rows)[*count].value == 0) {
        cross_dimensional_row_clear(&(*rows)[*count]);
        return RG_ERR_OOM;
    }
    (*count)++;
    return RG_OK;
}

static void tone_masses_sort(tone_mass *items, size_t count) {
    size_t i;
    for (i = 1; i < count; i++) {
        tone_mass key = items[i];
        size_t j = i;
        while (j > 0 && strcmp(items[j - 1].tone, key.tone) > 0) {
            items[j] = items[j - 1];
            j--;
        }
        items[j] = key;
    }
}

/* Scores whether this value's rate differs between the two sides, on the 2x2 table
 * of (value, not-value) by (inside, outside). The environment passing its own
 * test says the distribution moved; it does not say which value moved, and on
 * a dimension with several values most of them did not. Without this, an
 * environment that genuinely conditions one tone also publishes every other
 * tone that drifted upward inside it. Negative means the difference is worth
 * its parameter. */
static rg_status value_split_delta_score(
    double here_mass,
    double here_total,
    double there_mass,
    double there_total,
    const rg_split_score_config *base_config,
    double *out
) {
    double total = here_total + there_total;
    double pooled = (here_mass + there_mass) / total;
    double pooled_mass[2];
    double here[2];
    double there[2];
    rg_split_score_config config = *base_config;
    rg_split_score_result scored;
    if (total <= 0.0 || here_total <= 0.0 || there_total <= 0.0) {
        *out = 0.0;
        return RG_OK;
    }
    if (pooled <= 0.0 || pooled >= 1.0) {
        *out = 0.0;
        return RG_OK;
    }
    pooled_mass[0] = here_mass + there_mass;
    pooled_mass[1] = total - pooled_mass[0];
    here[0] = here_mass;
    here[1] = here_total - here_mass;
    there[0] = there_mass;
    there[1] = there_total - there_mass;
    config.bic_log_sample_size = log(total);
    config.bic_extra_penalty = 0.0;
    config.candidate_count = 1;
    config.search_gamma = 0.0;
    if (rg_categorical_split_score_internal(pooled_mass, here, there, 2, &config, &scored) != RG_OK) {
        return RG_ERR_UNSUPPORTED_OPTION;
    }
    *out = scored.delta;
    return RG_OK;
}

static double tone_mass_of(const tone_mass *items, size_t count, const char *tone) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(items[i].tone, tone) == 0) {
            return items[i].count;
        }
    }
    return 0.0;
}

static double tone_total(const tone_mass *items, size_t count) {
    double total = 0.0;
    size_t i;
    for (i = 0; i < count; i++) {
        total += items[i].count;
    }
    return total;
}

/* One aligned link whose target segment carries tone: the evidence
 * cross-dimensional discovery reasons over. `environments` is a bitmask over
 * the candidate environment table, so testing membership costs a shift rather
 * than a feature lookup, and `live` goes to zero once a committed rule
 * accounts for this observation. */
/* One predicate about the source form, relative to the segment whose target
 * dimension is being explained. Suprasegmentals appear as ordinary features
 * named "tone", "length" and "stress", so a predicate about the source's own
 * tone is the same kind of thing as one about a neighbour's voicing. */
typedef struct xdim_predicate {
    const char *feature;
    const char *value;
    int offset;
    /* A split is two-sided, and both sides are findings: "voiced onsets give
     * tone 4" is half of a tonogenesis and "voiceless onsets give tone 1" is
     * the other half. The negative side used to be published by reporting the
     * complement of whichever environment was committed, which cannot express
     * the complement of a conjunction -- "not (voiced and tone 2)" is not
     * "not voiced and not tone 2". So each predicate has its own negation and
     * both are searched. */
    int negated;
} xdim_predicate;

/* An environment: one predicate, or two conjoined. */
typedef struct xdim_environment {
    size_t first;
    size_t second;
    int conjoined;
} xdim_environment;

typedef struct xdim_observation {
    const char *value;
    double weight;
    unsigned char *holds;
    unsigned char *defined;
    int live;
} xdim_observation;

typedef struct xdim_scored {
    double delta_score;
    double search_charge;
    tone_mass *inside;
    size_t inside_count;
    tone_mass *outside;
    size_t outside_count;
    double inside_total;
    double outside_total;
    int usable;
} xdim_scored;

static void xdim_scored_clear(xdim_scored *scored) {
    tone_masses_clear(scored->inside, scored->inside_count);
    tone_masses_clear(scored->outside, scored->outside_count);
    scored->inside = 0;
    scored->outside = 0;
    scored->inside_count = 0;
    scored->outside_count = 0;
}

static int xdim_holds(const xdim_observation *observation, const xdim_environment *environment) {
    if (!observation->holds[environment->first]) {
        return 0;
    }
    return environment->conjoined ? observation->holds[environment->second] : 1;
}

static int xdim_defined(const xdim_observation *observation, const xdim_environment *environment);

/* Whether the second predicate of a conjunction excludes anything the first
 * one admits. A conjunct that excludes nothing has not narrowed the
 * environment: it is true of everything left, and conjoining it states a
 * condition that is not a condition. */
static int xdim_conjunction_narrows(
    const xdim_observation *observations,
    size_t observation_count,
    const xdim_environment *environment
) {
    size_t i;
    if (!environment->conjoined) {
        return 0;
    }
    for (i = 0; i < observation_count; i++) {
        const xdim_observation *observation = &observations[i];
        if (!observation->live) {
            continue;
        }
        if (!observation->defined[environment->first] || !observation->defined[environment->second]) {
            continue;
        }
        if (observation->holds[environment->first] && !observation->holds[environment->second]) {
            return 1;
        }
        if (observation->holds[environment->second] && !observation->holds[environment->first]) {
            return 1;
        }
    }
    return 0;
}

/* How two environments divide the observations: 1 for the same orientation,
 * -1 when they exchange inside and outside, and 0 for different splits. Different predicates
 * often carve one corpus identically -- `tone:2` and `tone:2 and not close-mid`
 * where nothing left is close-mid -- and committing each in turn republishes
 * one finding as several. */
static int xdim_partition_relation(
    const xdim_observation *observations,
    size_t observation_count,
    const xdim_environment *a,
    const xdim_environment *b
) {
    size_t i;
    int same = 1;
    int opposite = 1;
    for (i = 0; i < observation_count; i++) {
        const xdim_observation *observation = &observations[i];
        int a_defined;
        int b_defined;
        if (!observation->live) {
            continue;
        }
        a_defined = observation->defined[a->first] && (!a->conjoined || observation->defined[a->second]);
        b_defined = observation->defined[b->first] && (!b->conjoined || observation->defined[b->second]);
        if (a_defined != b_defined) {
            return 0;
        }
        if (!a_defined) {
            continue;
        }
        same = same && xdim_holds(observation, a) == xdim_holds(observation, b);
        opposite = opposite && xdim_holds(observation, a) != xdim_holds(observation, b);
    }
    return same ? 1 : (opposite ? -1 : 0);
}

static rg_status xdim_distinct_partition_count(
    const xdim_observation *observations,
    size_t observation_count,
    const xdim_environment *candidates,
    size_t candidate_count,
    size_t *out
) {
    uint64_t *hashes;
    size_t distinct = 0;
    size_t c;
    hashes = (uint64_t *)calloc(candidate_count == 0 ? 1 : candidate_count, sizeof(*hashes));
    if (hashes == 0) {
        return RG_ERR_OOM;
    }
    for (c = 0; c < candidate_count; c++) {
        uint64_t hash = UINT64_C(1469598103934665603);
        uint64_t complement_hash = UINT64_C(1469598103934665603);
        size_t i;
        int duplicate = 0;
        for (i = 0; i < observation_count; i++) {
            unsigned int state = 0;
            if (observations[i].live) {
                state = xdim_defined(&observations[i], &candidates[c])
                    ? (unsigned int)(2 + xdim_holds(&observations[i], &candidates[c]))
                    : 1u;
            }
            hash ^= (uint64_t)state;
            hash *= UINT64_C(1099511628211);
            if (state == 2u) {
                state = 3u;
            } else if (state == 3u) {
                state = 2u;
            }
            complement_hash ^= (uint64_t)state;
            complement_hash *= UINT64_C(1099511628211);
        }
        hash = hash < complement_hash ? hash : complement_hash;
        for (i = 0; i < c; i++) {
            if (hashes[i] == hash &&
                xdim_partition_relation(observations, observation_count,
                                        &candidates[c], &candidates[i]) != 0) {
                duplicate = 1;
                break;
            }
        }
        hashes[c] = hash;
        if (!duplicate) {
            distinct++;
        }
    }
    free(hashes);
    *out = distinct;
    return RG_OK;
}

static int xdim_defined(const xdim_observation *observation, const xdim_environment *environment) {
    if (!observation->defined[environment->first]) {
        return 0;
    }
    return environment->conjoined ? observation->defined[environment->second] : 1;
}

/* Whether every observation `a` selects is also selected by `b`. A narrower
 * environment inside one that already determined its outcome says nothing new:
 * `voiced and stop` inside `voiced and source tone 2` names nine of the same
 * twelve and reports the same value. Refining an environment that did *not*
 * determine its outcome is the opposite -- it is the whole point -- so this
 * screen applies only to the determined ones. */
static int xdim_inside_subset(
    const xdim_observation *observations,
    size_t observation_count,
    const xdim_environment *a,
    const xdim_environment *b
) {
    size_t i;
    size_t shared = 0;
    for (i = 0; i < observation_count; i++) {
        const xdim_observation *observation = &observations[i];
        if (!observation->live || !xdim_defined(observation, a) || !xdim_holds(observation, a)) {
            continue;
        }
        if (!xdim_defined(observation, b) || !xdim_holds(observation, b)) {
            return 0;
        }
        shared++;
    }
    return shared > 0;
}

static rg_status xdim_score_environment(
    const xdim_observation *observations,
    size_t observation_count,
    const xdim_environment *environment,
    double min_count,
    size_t candidate_count,
    const rg_split_score_config *base_config,
    xdim_scored *out
) {
    tone_mass *pooled = 0;
    size_t pooled_count = 0;
    size_t pooled_cap = 0;
    size_t inside_cap = 0;
    size_t outside_cap = 0;
    size_t i;
    rg_status status = RG_OK;

    memset(out, 0, sizeof(*out));
    for (i = 0; i < observation_count && status == RG_OK; i++) {
        const xdim_observation *observation = &observations[i];
        if (!observation->live || !xdim_defined(observation, environment)) {
            continue;
        }
        if (xdim_holds(observation, environment)) {
            status = add_tone_mass(&out->inside, &out->inside_count, &inside_cap,
                                   observation->value, observation->weight);
        } else {
            status = add_tone_mass(&out->outside, &out->outside_count, &outside_cap,
                                   observation->value, observation->weight);
        }
        if (status == RG_OK) {
            status = add_tone_mass(&pooled, &pooled_count, &pooled_cap,
                                   observation->value, observation->weight);
        }
    }
    if (status != RG_OK) {
        tone_masses_clear(pooled, pooled_count);
        xdim_scored_clear(out);
        return status;
    }
    out->inside_total = tone_total(out->inside, out->inside_count);
    out->outside_total = tone_total(out->outside, out->outside_count);
    /* Both sides of the contrast must be attested, or there is no split to
     * test: a predicate holding of every segment partitions nothing. */
    if (out->inside_total >= min_count && out->outside_total >= min_count && pooled_count > 1) {
        double pooled_total = out->inside_total + out->outside_total;
        double *pooled_mass;
        double *inside_mass;
        double *outside_mass;
        rg_split_score_config config = *base_config;
        rg_split_score_result scored;
        tone_masses_sort(pooled, pooled_count);
        pooled_mass = (double *)calloc(pooled_count * 3, sizeof(*pooled_mass));
        if (pooled_mass == 0) {
            tone_masses_clear(pooled, pooled_count);
            xdim_scored_clear(out);
            return RG_ERR_OOM;
        }
        inside_mass = pooled_mass + pooled_count;
        outside_mass = inside_mass + pooled_count;
        for (i = 0; i < pooled_count; i++) {
            pooled_mass[i] = pooled[i].count;
            inside_mass[i] = tone_mass_of(out->inside, out->inside_count, pooled[i].tone);
            outside_mass[i] = tone_mass_of(out->outside, out->outside_count, pooled[i].tone);
        }
        config.bic_log_sample_size = log(pooled_total);
        config.bic_extra_penalty = 0.0;
        config.candidate_count = candidate_count;
        status = rg_categorical_split_score_internal(pooled_mass, inside_mass, outside_mass,
                                                     pooled_count, &config, &scored);
        free(pooled_mass);
        if (status != RG_OK) {
            tone_masses_clear(pooled, pooled_count);
            xdim_scored_clear(out);
            return status;
        }
        out->delta_score = scored.delta;
        out->search_charge = scored.search_charge;
        out->usable = 1;
    }
    tone_masses_clear(pooled, pooled_count);
    return RG_OK;
}

/* The environment as a context, which is how every other conditioned rule in
 * this library states one. Offset -1 is the preceding segment, 0 the segment
 * itself, +1 the following one. */
static rg_status xdim_environment_context(
    const xdim_predicate *predicates,
    const xdim_environment *environment,
    int flip,
    rg_context_spec *out
) {
    rg_split_candidate candidate;
    rg_context_spec first;
    rg_status status;
    size_t which;

    rg_context_spec_init_empty(out);
    for (which = 0; which < (environment->conjoined ? 2u : 1u); which++) {
        const xdim_predicate *predicate =
            &predicates[which == 0 ? environment->first : environment->second];
        candidate.slot = predicate->offset < 0 ? "preceding"
            : (predicate->offset > 0 ? "following" : "self");
        candidate.feature = predicate->feature;
        /* The complement of a conjunction is not a conjunction of complements,
         * so a negated environment is published as the predicates it fails,
         * with the value marked. A reader sees which predicates it is the
         * complement of. */
        candidate.value = (predicate->negated != flip) ? "-" : predicate->value;
        if (which == 0) {
            status = rg_context_from_candidate_internal(&candidate, out);
            if (status != RG_OK) {
                return status;
            }
        } else {
            first = *out;
            status = rg_context_extend_internal(&first, &candidate, out);
            rg_context_spec_clear_internal(&first);
            if (status != RG_OK) {
                return status;
            }
        }
    }
    return RG_OK;
}

/* How many singles a conjunction is searched over. Every usable single is
 * scored; the pairs are formed among the best of them, because P predicates
 * give P(P-1)/2 pairs and P is in the hundreds once the vocabulary is derived
 * from the corpus. The cap is reported when it bites. */
#define RG_XDIM_MAX_PAIR_BASE 16

/* Which suprasegmental dimensions a rule can be about, as source predicate and
 * as target alike. The scorer has handled length and stress as target
 * dimensions since the port; only this stage never proposed them. */
static const char *const xdim_dimension_names[] = { "tone", "length", "stress" };

static const char *segment_dimension_value(const rg_segment *segment, const char *dimension) {
    if (strcmp(dimension, "tone") == 0) {
        return segment->tone;
    }
    if (strcmp(dimension, "length") == 0) {
        return segment->length;
    }
    return segment->stress;
}

static rg_status xdim_add_predicate(
    xdim_predicate **items,
    size_t *count,
    size_t *cap,
    const char *feature,
    const char *value,
    int offset,
    int negated
) {
    size_t i;
    for (i = 0; i < *count; i++) {
        if ((*items)[i].offset == offset && (*items)[i].negated == negated &&
            strcmp((*items)[i].feature, feature) == 0 &&
            strcmp((*items)[i].value, value) == 0) {
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 64 : *cap * 2;
        xdim_predicate *next = (xdim_predicate *)realloc(*items, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[*count].feature = feature;
    (*items)[*count].value = value;
    (*items)[*count].offset = offset;
    (*items)[*count].negated = negated;
    (*count)++;
    return RG_OK;
}

/* Whether the source form satisfies a predicate at a position. */
static int xdim_predicate_holds(
    const rg_context *ctx,
    const rg_form *form,
    int index,
    const xdim_predicate *predicate
) {
    size_t d;
    for (d = 0; d < sizeof(xdim_dimension_names) / sizeof(xdim_dimension_names[0]); d++) {
        if (strcmp(predicate->feature, xdim_dimension_names[d]) == 0) {
            const char *value;
            if (index < 0 || (size_t)index >= form->segment_count) {
                return 0;
            }
            value = segment_dimension_value(&form->segments[index], xdim_dimension_names[d]);
            return value != 0 && strcmp(value, predicate->value) == 0;
        }
    }
    return segment_has_context_feature(ctx, form, index, predicate->feature);
}

static int xdim_predicate_satisfied(
    const rg_context *ctx,
    const rg_form *form,
    int index,
    const xdim_predicate *predicate
) {
    int holds = xdim_predicate_holds(ctx, form, index, predicate);
    return predicate->negated ? !holds : holds;
}

/* One orientation of cross-dimensional discovery: does something about one
 * form condition a suprasegmental value on the other?
 *
 * `context_is_target` picks which form states the environment; the conditioned
 * dimension is read from the other one. The caller runs both.
 *
 * The claim is a conditioned split, so it is tested the way this pipeline
 * tests one. An environment earns a rule only when
 *
 *   1. the complementary environment exists and is attested. A predicate that
 *      holds of every segment partitions nothing, and "the preceding segment
 *      is a consonant" on a corpus of CV syllables is not an environment; it
 *      is a description of the corpus.
 *   2. modelling the dimension separately inside and outside beats modelling
 *      it once by more than the extra parameters *and the search* cost. This
 *      is the criterion, and the code shape, of context discovery.
 *
 * and a value inside that environment is reported only when the environment
 * raises it above its rate in the contrast. That last condition is what makes
 * the output read as historical linguistics rather than as a frequency table:
 * a rule says the environment *did something*, and on a two-valued dimension
 * two rules naming one environment can no longer contradict each other. The
 * stage previously committed on P(value | environment) >= 0.5 with no contrast
 * at all, which reported the ambient distribution as though it were a rule.
 *
 * An environment may conjoin two predicates, and has to. The Middle Chinese
 * register split conditions the tone on the preceding onset's voicing *and* on
 * that segment's own tone: a tone 2 becomes tone 4 after a voiced onset and
 * tone 2 after a voiceless one, while a tone 1 is unaffected either way.
 * Neither predicate alone predicts anything -- voicing alone reported that rule
 * at confidence 0.50 -- and until 2026-08-15 neither the row nor the search
 * could say both at once. The environment form's own tone was not in the
 * predicate vocabulary either.
 *
 * Environments are committed one at a time, best first, against what the
 * already-committed rules have not accounted for. Without that, every
 * correlated framing of one fact commits separately -- on a corpus of CV
 * syllables `consonant`, `sonorant`, `nasal` and `voiced` at the same offset
 * are four descriptions of the same coda, and a reader has no way to tell that
 * they are one finding. Explaining the residue is what this stage is for; it
 * runs after the segmental and tonal baselines for the same reason. */
static rg_status discover_cross_dimensional_orientation(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    const rg_feature_vocabulary *vocabulary,
    int context_is_target,
    int same_lect,
    rg_cross_dimensional_row **rows,
    size_t *row_count,
    size_t *row_cap
) {
    rg_status status = RG_OK;
    double min_count = 3.0;
    double min_confidence = 0.0;
    double delta_threshold = 0.0;
    int max_iterations = 5;
    int max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    rg_split_score_config score_config;
    size_t dimension_i;

    if (options != 0) {
        if (options->bic.cross_dim_min_rule_count > 0) {
            min_count = (double)options->bic.cross_dim_min_rule_count;
        }
        if (options->bic.cross_dim_min_rule_confidence > 0.0) {
            min_confidence = options->bic.cross_dim_min_rule_confidence;
        }
        delta_threshold = options->bic.cross_dim_delta_bic_threshold;
        if (options->bic.cross_dim_max_iterations > 0) {
            max_iterations = options->bic.cross_dim_max_iterations;
        }
        if (options->max_chunk_size > 0) {
            max_chunk_size = options->max_chunk_size;
        }
    }
    score_config.scorer = options == 0
        ? RG_SPLIT_SCORER_CORRECTED_BIC : options->bic.split_scorer;
    score_config.dirichlet_concentration = options == 0
        ? 1.0 : options->bic.split_prior_concentration;
    score_config.bic_log_sample_size = 0.0;
    score_config.bic_extra_penalty = 0.0;
    score_config.candidate_count = 0;
    score_config.search_gamma = options == 0
        ? RG_SEARCH_PENALTY_GAMMA : options->bic.search_penalty_gamma;

    /* One pass per target dimension. The scorer has handled length and stress
     * as targets since the port; this stage only ever proposed tone. */
    for (dimension_i = 0;
         dimension_i < sizeof(xdim_dimension_names) / sizeof(xdim_dimension_names[0]) && status == RG_OK;
         dimension_i++) {
        const char *target_dimension = xdim_dimension_names[dimension_i];
        xdim_predicate *predicates = 0;
        size_t predicate_count = 0;
        size_t predicate_cap = 0;
        xdim_environment *candidates = 0;
        size_t candidate_count = 0;
        size_t candidate_cap = 0;
        xdim_observation *observations = 0;
        size_t observation_count = 0;
        size_t observation_cap = 0;
        int *committed = 0;
        int *committed_predicate = 0;
        int *determined = 0;
        size_t pair_i;
        size_t i;
        int offset;
        int iteration;

        /* The predicate vocabulary: whatever the corpus contrasts segmentally,
         * plus every suprasegmental value it actually carries on the source
         * side. Both at each of the three positions a rule can name. */
        for (i = 0; i < vocabulary->count && status == RG_OK; i++) {
            for (offset = -1; offset <= 1 && status == RG_OK; offset++) {
                status = xdim_add_predicate(&predicates, &predicate_count, &predicate_cap,
                                            vocabulary->entries[i].feature,
                                            vocabulary->entries[i].value, offset, 0);
                if (status == RG_OK) {
                    status = xdim_add_predicate(&predicates, &predicate_count, &predicate_cap,
                                                vocabulary->entries[i].feature,
                                                vocabulary->entries[i].value, offset, 1);
                }
            }
        }
        for (pair_i = 0; pair_i < pair_count && status == RG_OK; pair_i++) {
            const rg_form *environment_form =
                context_is_target ? &pairs[pair_i].target : &pairs[pair_i].source;
            size_t seg_i;
            for (seg_i = 0; seg_i < environment_form->segment_count && status == RG_OK; seg_i++) {
                size_t d;
                for (d = 0; d < sizeof(xdim_dimension_names) / sizeof(xdim_dimension_names[0]) &&
                            status == RG_OK; d++) {
                    const char *value = segment_dimension_value(&environment_form->segments[seg_i],
                                                                xdim_dimension_names[d]);
                    if (value == 0 || value[0] == '\0') {
                        continue;
                    }
                    for (offset = -1; offset <= 1 && status == RG_OK; offset++) {
                        status = xdim_add_predicate(&predicates, &predicate_count, &predicate_cap,
                                                    xdim_dimension_names[d], value, offset, 0);
                        if (status == RG_OK) {
                            status = xdim_add_predicate(&predicates, &predicate_count, &predicate_cap,
                                                        xdim_dimension_names[d], value, offset, 1);
                        }
                    }
                }
            }
        }
        if (status != RG_OK || predicate_count == 0) {
            free(predicates);
            continue;
        }
        /* One alignment pass, with each link's predicate membership recorded
         * as it is walked. */
        for (pair_i = 0; pair_i < pair_count && status == RG_OK; pair_i++) {
            rg_alignment *alignment = 0;
            /* The alignment is always searched in the direction the model was
             * trained in. Only which side supplies the environment and which
             * supplies the conditioned dimension changes here. */
            const rg_form *environment_form =
                context_is_target ? &pairs[pair_i].target : &pairs[pair_i].source;
            const rg_form *conditioned_form =
                context_is_target ? &pairs[pair_i].source : &pairs[pair_i].target;
            size_t link_i;
            size_t src_pos = 0;
            size_t tgt_pos = 0;
            double weight = pairs[pair_i].weight == 0.0 ? 1.0 : pairs[pair_i].weight;
            if (weight <= 0.0) {
                continue;
            }
            status = rg_align_forms_with_model(ctx, model, options, &pairs[pair_i].source,
                                               &pairs[pair_i].target, max_chunk_size, &alignment);
            if (status != RG_OK) {
                break;
            }
            for (link_i = 0; link_i < rg_alignment_link_count(alignment) && status == RG_OK; link_i++) {
                const rg_link *link = rg_alignment_link_at(alignment, link_i);
                if (link->source_count == 1 && link->target_count == 1) {
                    size_t environment_pos = context_is_target ? tgt_pos : src_pos;
                    /* Lect-internal: the tone is read from the same form the
                     * environment is stated on, at the environment position --
                     * the onset conditions the tone on its own vowel. */
                    size_t conditioned_pos = same_lect ? environment_pos
                                                       : (context_is_target ? src_pos : tgt_pos);
                    const rg_form *value_form = same_lect ? environment_form : conditioned_form;
                    const char *value = segment_dimension_value(
                        &value_form->segments[conditioned_pos],
                        target_dimension);
                    if (value != 0 && value[0] != '\0') {
                        xdim_observation *slot;
                        if (observation_count == observation_cap) {
                            size_t next_cap = observation_cap == 0 ? 64 : observation_cap * 2;
                            xdim_observation *next = (xdim_observation *)realloc(
                                observations, next_cap * sizeof(*next));
                            if (next == 0) {
                                status = RG_ERR_OOM;
                                break;
                            }
                            observations = next;
                            observation_cap = next_cap;
                        }
                        slot = &observations[observation_count];
                        memset(slot, 0, sizeof(*slot));
                        slot->value = value;
                        slot->weight = weight;
                        slot->live = 1;
                        slot->holds = (unsigned char *)calloc(predicate_count, sizeof(*slot->holds));
                        slot->defined = (unsigned char *)calloc(predicate_count, sizeof(*slot->defined));
                        if (slot->holds == 0 || slot->defined == 0) {
                            free(slot->holds);
                            free(slot->defined);
                            status = RG_ERR_OOM;
                            break;
                        }
                        for (i = 0; i < predicate_count; i++) {
                            int index = (int)environment_pos + predicates[i].offset;
                            if (!context_position_exists(environment_form, index)) {
                                continue;
                            }
                            /* Lect-internal, a predicate on the outcome's own
                             * dimension at its own position predicts itself: a
                             * tone always has its own tone. Left undefined so it
                             * partitions nothing. Offsets away from zero remain
                             * -- a preceding tone conditioning this one is tone
                             * sandhi, a real lect-internal rule. */
                            if (same_lect && predicates[i].offset == 0 &&
                                strcmp(predicates[i].feature, target_dimension) == 0) {
                                continue;
                            }
                            slot->defined[i] = 1;
                            if (xdim_predicate_satisfied(ctx, environment_form, index, &predicates[i])) {
                                slot->holds[i] = 1;
                            }
                        }
                        observation_count++;
                    }
                }
                src_pos += link->source_count;
                tgt_pos += link->target_count;
            }
            rg_alignment_free(alignment);
        }

        /* Every single predicate, then conjunctions among the ones that carry
         * a usable split on their own. Pairs are P(P-1)/2 and P is in the
         * hundreds once the vocabulary comes from the corpus, so the base is
         * capped at the best RG_XDIM_MAX_PAIR_BASE singles by delta-BIC. */
        for (i = 0; i < predicate_count && status == RG_OK; i++) {
            if (candidate_count == candidate_cap) {
                size_t next_cap = candidate_cap == 0 ? 128 : candidate_cap * 2;
                xdim_environment *next = (xdim_environment *)realloc(candidates, next_cap * sizeof(*next));
                if (next == 0) {
                    status = RG_ERR_OOM;
                    break;
                }
                candidates = next;
                candidate_cap = next_cap;
            }
            candidates[candidate_count].first = i;
            candidates[candidate_count].second = 0;
            candidates[candidate_count].conjoined = 0;
            candidate_count++;
        }
        if (status == RG_OK) {
            size_t base[RG_XDIM_MAX_PAIR_BASE];
            double base_score[RG_XDIM_MAX_PAIR_BASE];
            size_t base_count = 0;
            size_t a;
            for (i = 0; i < predicate_count && status == RG_OK; i++) {
                xdim_scored scored;
                size_t slot_i;
                status = xdim_score_environment(observations, observation_count,
                                                &candidates[i], min_count, 1,
                                                &score_config, &scored);
                if (status != RG_OK) {
                    break;
                }
                if (scored.usable) {
                    slot_i = base_count;
                    while (slot_i > 0 && base_score[slot_i - 1] > scored.delta_score) {
                        if (slot_i < RG_XDIM_MAX_PAIR_BASE) {
                            base[slot_i] = base[slot_i - 1];
                            base_score[slot_i] = base_score[slot_i - 1];
                        }
                        slot_i--;
                    }
                    if (slot_i < RG_XDIM_MAX_PAIR_BASE) {
                        base[slot_i] = i;
                        base_score[slot_i] = scored.delta_score;
                        if (base_count < RG_XDIM_MAX_PAIR_BASE) {
                            base_count++;
                        }
                    }
                }
                xdim_scored_clear(&scored);
            }
            for (a = 0; a + 1 < base_count && status == RG_OK; a++) {
                size_t b;
                for (b = a + 1; b < base_count && status == RG_OK; b++) {
                    if (predicates[base[a]].offset == predicates[base[b]].offset &&
                        strcmp(predicates[base[a]].feature, predicates[base[b]].feature) == 0) {
                        /* Two values of one feature at one position cannot both
                         * hold; the conjunction is empty by construction. */
                        continue;
                    }
                    if (candidate_count == candidate_cap) {
                        size_t next_cap = candidate_cap == 0 ? 128 : candidate_cap * 2;
                        xdim_environment *next = (xdim_environment *)realloc(candidates, next_cap * sizeof(*next));
                        if (next == 0) {
                            status = RG_ERR_OOM;
                            break;
                        }
                        candidates = next;
                        candidate_cap = next_cap;
                    }
                    candidates[candidate_count].first = base[a];
                    candidates[candidate_count].second = base[b];
                    candidates[candidate_count].conjoined = 1;
                    candidate_count++;
                }
            }
        }

        if (status == RG_OK) {
            committed = (int *)calloc(candidate_count == 0 ? 1 : candidate_count, sizeof(*committed));
            committed_predicate = (int *)calloc(predicate_count, sizeof(*committed_predicate));
            determined = (int *)calloc(candidate_count == 0 ? 1 : candidate_count, sizeof(*determined));
            if (committed == 0 || committed_predicate == 0 || determined == 0) {
                status = RG_ERR_OOM;
            }
        }

        for (iteration = 0; iteration < max_iterations && status == RG_OK; iteration++) {
            xdim_scored best;
            size_t best_env = 0;
            int found = 0;
            int decision_index = model->decision_count;
            double best_margin = 0.0;
            size_t env_i;
            size_t partition_count = 0;

            memset(&best, 0, sizeof(best));
            status = xdim_distinct_partition_count(observations, observation_count,
                                                   candidates, candidate_count,
                                                   &partition_count);
            for (env_i = 0; env_i < candidate_count && status == RG_OK; env_i++) {
                xdim_scored scored;
                size_t seen_i;
                int duplicate = 0;
                if (committed[env_i]) {
                    continue;
                }
                for (seen_i = 0; seen_i < candidate_count && !duplicate; seen_i++) {
                    if (!committed[seen_i]) {
                        continue;
                    }
                    duplicate = xdim_partition_relation(observations, observation_count,
                                                        &candidates[env_i],
                                                        &candidates[seen_i]) == 1 ||
                        (determined[seen_i] &&
                         xdim_inside_subset(observations, observation_count,
                                            &candidates[env_i], &candidates[seen_i]));
                }
                if (duplicate) {
                    continue;
                }
                status = xdim_score_environment(observations, observation_count, &candidates[env_i],
                                                min_count, partition_count,
                                                &score_config, &scored);
                if (status != RG_OK) {
                    break;
                }
                /* On a tie, a conjunction wins only when one of its predicates
                 * has already been committed. Such a conjunct re-states the
                 * residue the rule was found on, and without it the rule reads
                 * as a claim about the corpus: "a voiced onset gives tone 4"
                 * holds of everything left after source tone is accounted for,
                 * and of half the corpus. A conjunct that names nothing
                 * already committed and does not change the split is noise,
                 * and loses to the single. */
                int restates_residue = candidates[env_i].conjoined &&
                    (committed_predicate[candidates[env_i].first] ||
                     committed_predicate[candidates[env_i].second]) &&
                    xdim_conjunction_narrows(observations, observation_count, &candidates[env_i]);
                if (scored.usable && scored.delta_score < delta_threshold &&
                    (!found ||
                     scored.delta_score < best.delta_score - RG_TIE_EPSILON ||
                     (restates_residue && !candidates[best_env].conjoined &&
                      scored.delta_score < best.delta_score + RG_TIE_EPSILON))) {
                    if (found) {
                        xdim_scored_clear(&best);
                    }
                    best = scored;
                    best_env = env_i;
                    /* How heavy a search charge this rule's evidence carries,
                     * in the same units the context splitter reports, so it
                     * can be read against the corpus's shuffled ceiling. */
                    best_margin = partition_count > 1
                        ? (delta_threshold - (scored.delta_score - scored.search_charge)) /
                          (2.0 * log((double)partition_count))
                        : 0.0;
                    found = 1;
                } else {
                    xdim_scored_clear(&scored);
                }
            }
            if (status != RG_OK || !found) {
                xdim_scored_clear(&best);
                break;
            }
            {
                /* A split is a two-sided statement, and both sides are
                 * findings: "voiced onsets give tone 4" is half of the
                 * tonogenesis and "voiceless onsets give tone 1" is the other
                 * half. */
                int side;
                int emitted = 0;
                /* A single predicate publishes both sides: "voiced onsets give
                 * tone 4" is half of a tonogenesis and "voiceless onsets give
                 * tone 1" is the other half, and the complement of one
                 * predicate is one predicate.
                 *
                 * A conjunction publishes only the side it holds on. The
                 * complement of "voiced and source tone 2" is not "voiceless
                 * and not source tone 2", and rg_context_spec cannot say "not
                 * (A and B)". The other half is reachable as its own
                 * conjunction, because negation is a predicate. */
                int sides = candidates[best_env].conjoined ? 1 : 2;
                int determined_here = 0;
                for (side = 0; side < sides && status == RG_OK; side++) {
                    const tone_mass *here = side == 0 ? best.inside : best.outside;
                    size_t here_count = side == 0 ? best.inside_count : best.outside_count;
                    const tone_mass *there = side == 0 ? best.outside : best.inside;
                    size_t there_count = side == 0 ? best.outside_count : best.inside_count;
                    double here_total = side == 0 ? best.inside_total : best.outside_total;
                    double there_total = side == 0 ? best.outside_total : best.inside_total;
                    size_t value_i;
                    rg_context_spec environment;
                    status = xdim_environment_context(predicates, &candidates[best_env], side != 0,
                                                      &environment);
                    if (status != RG_OK) {
                        break;
                    }
                    /* Which values this side raises. Decided before anything
                     * is published, because how many there are decides whether
                     * the environment has determined the outcome. */
                    size_t raised = 0;
                    for (value_i = 0; value_i < here_count; value_i++) {
                        double here_mass = here[value_i].count;
                        double there_mass = tone_mass_of(there, there_count, here[value_i].tone);
                        if (here_mass < min_count || here_mass / here_total < min_confidence) {
                            continue;
                        }
                        if (here_mass / here_total <= there_mass / there_total) {
                            continue;
                        }
                        double value_score;
                        status = value_split_delta_score(here_mass, here_total,
                                                         there_mass, there_total,
                                                         &score_config, &value_score);
                        if (status != RG_OK || value_score >= delta_threshold) {
                            continue;
                        }
                        raised++;
                    }
                    for (value_i = 0; value_i < here_count && status == RG_OK; value_i++) {
                        double here_mass = here[value_i].count;
                        double there_mass = tone_mass_of(there, there_count, here[value_i].tone);
                        double confidence = here_mass / here_total;
                        double contrast = there_mass / there_total;
                        if (here_mass < min_count || confidence < min_confidence) {
                            continue;
                        }
                        if (confidence <= contrast) {
                            continue;
                        }
                        double value_score;
                        status = value_split_delta_score(here_mass, here_total,
                                                         there_mass, there_total,
                                                         &score_config, &value_score);
                        if (status != RG_OK || value_score >= delta_threshold) {
                            continue;
                        }
                        status = append_cross_dimensional_row(
                            rows, row_count, row_cap,
                            &environment, context_is_target, same_lect,
                            target_dimension, here[value_i].tone, 0,
                            here_mass, here_total, there_mass, there_total,
                            best.delta_score, decision_index, best_margin);
                        if (status == RG_OK) {
                            size_t obs_i;
                            /* Retire what this rule accounts for. An
                             * environment that raises *several* values has
                             * said something true and has not determined the
                             * outcome, so its members stay live for a narrower
                             * environment to refine. Retiring them anyway is
                             * what stopped the Middle Chinese register split
                             * being found: source tone 2 raises both tone 2
                             * and tone 4, and consuming both left the
                             * conjunction with nothing to explain. */
                            /* A conjunction keeps its members live. Retiring
                             * them leaves the complement homogeneous, and a
                             * homogeneous group is not a split, so the other
                             * half of the rule -- a voiceless onset gives tone
                             * 2 -- would become unstatable. Re-deriving the
                             * same finding is prevented by the partition check
                             * above rather than by consuming the evidence. */
                            if (raised == 1 && !candidates[best_env].conjoined) {
                                for (obs_i = 0; obs_i < observation_count; obs_i++) {
                                    int inside;
                                    if (!xdim_defined(&observations[obs_i], &candidates[best_env])) {
                                        continue;
                                    }
                                    inside = xdim_holds(&observations[obs_i], &candidates[best_env]);
                                    if (observations[obs_i].live && inside == (side == 0) &&
                                        strcmp(observations[obs_i].value, here[value_i].tone) == 0) {
                                        observations[obs_i].live = 0;
                                    }
                                }
                            }
                            emitted = 1;
                            /* Only a conjunction can shut the door on
                             * refinement. A single predicate that determined
                             * one side has retired that side's members, so
                             * nothing is left there to refine, and its *other*
                             * side is exactly what a narrower environment
                             * should be allowed to explain -- which is how the
                             * register split is found. */
                            if (raised == 1 && candidates[best_env].conjoined) {
                                determined_here = 1;
                            }
                        }
                    }
                    rg_context_spec_clear_internal(&environment);
                }
                model->decision_count++;
                committed[best_env] = 1;
                if (determined_here) {
                    determined[best_env] = 1;
                }
                committed_predicate[candidates[best_env].first] = 1;
                if (candidates[best_env].conjoined) {
                    committed_predicate[candidates[best_env].second] = 1;
                }
                xdim_scored_clear(&best);
                if (!emitted) {
                    break;
                }
            }
        }

        for (i = 0; i < observation_count; i++) {
            free(observations[i].holds);
            free(observations[i].defined);
        }
        free(observations);
        free(candidates);
        free(committed);
        free(committed_predicate);
        free(determined);
        free(predicates);
    }

    return status;
}

/* Both orientations of the pair, because they ask different questions.
 *
 * "Does something about the source form condition a suprasegmental value on
 * the target?" has an answer only where the source still shows the
 * conditioning contrast and the target carries the dimension. A lect that
 * merged the voicing contrast has nothing to state an environment over, and a
 * lect with no tone has nothing to condition, so a corpus can carry a rule in
 * one orientation, in both, or in neither.
 *
 * Searching one orientation only made a published finding depend on which lect
 * name sorted first, which is metadata. Neither orientation is a direction of
 * change: which lect the environment sits in is a fact about what each lect
 * preserved. */
rg_status discover_cross_dimensional_rows(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    const rg_feature_vocabulary *vocabulary
) {
    rg_cross_dimensional_row *rows = 0;
    size_t row_count = 0;
    size_t row_cap = 0;
    rg_status status;
    int context_is_target;
    int same_lect;

    if (ctx == 0 || model == 0 || (pair_count > 0 && pairs == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    /* Four orientations, each with its own observation pool so the cross-lect
     * output is unchanged by the lect-internal passes. same_lect=0 asks whether
     * one lect's material predicts the other's tone; same_lect=1 asks whether a
     * lect's own onset predicts its own tone, which is tonogenesis and what the
     * stage could not state before. context_is_target then names the form both
     * are read from. */
    for (same_lect = 0; same_lect < 2; same_lect++) {
        for (context_is_target = 0; context_is_target < 2; context_is_target++) {
            status = discover_cross_dimensional_orientation(
                ctx, pairs, pair_count, options, model, vocabulary, context_is_target,
                same_lect, &rows, &row_count, &row_cap);
            if (status != RG_OK) {
                size_t i;
                for (i = 0; i < row_count; i++) {
                    cross_dimensional_row_clear(&rows[i]);
                }
                free(rows);
                return status;
            }
        }
    }
    if (row_count > 1) {
        qsort(rows, row_count, sizeof(*rows), cross_dimensional_row_cmp);
    }
    {
        size_t i;
        for (i = 0; i < model->cross_dimensional_count; i++) {
            cross_dimensional_row_clear(&model->cross_dimensional_rows[i]);
        }
    }
    free(model->cross_dimensional_rows);
    model->cross_dimensional_rows = rows;
    model->cross_dimensional_count = row_count;
    return RG_OK;
}
