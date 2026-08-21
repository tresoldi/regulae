#ifndef REGULAE_H
#define REGULAE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(REGULAE_SHARED)
#  if defined(REGULAE_BUILDING_LIBRARY)
#    define RG_API __declspec(dllexport)
#  else
#    define RG_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define RG_API __attribute__((visibility("default")))
#else
#  define RG_API
#endif

#define RG_VERSION_MAJOR 0
#define RG_VERSION_MINOR 1
#define RG_VERSION_PATCH 0
#define RG_VERSION_STRING "0.1.0"
#define RG_ABI_VERSION 42
#define RG_DEFAULT_MAX_CHUNK_SIZE 3
/* merkmal's own default. It reads the same graphemes and returns the same
 * feature labels as "descriptive", but scores through its own dimensions, and
 * on BDPA gold alignments it is the one merkmal cannot distinguish from SCA.
 * rg_context_use_system takes any system the registry holds. */
#define RG_DEFAULT_FEATURE_SYSTEM "distinctive"
/* The grapheme a multi-lect class carries for a lect that deleted the segment
 * the others keep -- "∅", U+2205. A class row names the loss as `lect:∅`, so a
 * loss is distinct from a lect that never had the word rather than the deleting
 * lect being simply absent from the class.
 * It is only ever a class-table grapheme, never fed back to
 * the feature system, and it appears only in the unconditioned table -- a gap
 * conditions nothing, so it is kept out of the conditioning search. */
#define RG_GAP_GRAPHEME "\xe2\x88\x85"

typedef struct rg_context rg_context;
typedef struct rg_feature_set rg_feature_set;
typedef struct rg_alignment rg_alignment;
typedef struct rg_pairwise_model rg_pairwise_model;
typedef struct rg_multi_model rg_multi_model;
typedef struct rg_corpus rg_corpus;
typedef struct rg_translation_table rg_translation_table;

typedef enum rg_status {
    RG_OK = 0,
    RG_ERR_INVALID_ARGUMENT,
    RG_ERR_IO,
    RG_ERR_PARSE,
    RG_ERR_MERKMAL,
    RG_ERR_UNKNOWN_GRAPHEME,
    RG_ERR_UNSUPPORTED_OPTION,
    RG_ERR_OOM,
    RG_ERR_CANCELLED,
    /* The token is CLDF/CLTS markup rather than a transcription of a sound:
     * `<?>`, `<<...>>`, and the boundary marks `+`, `_` and `#`. This is a
     * documented gap in the source data, not a sound the feature system fails
     * to cover, and the two call for different responses from the user. */
    RG_ERR_SOURCE_MARKER
} rg_status;

/* Reports training progress. stage names the pipeline step just finished;
 * completed and total count steps, so completed/total is a usable fraction.
 * Return true to abort the run, which surfaces as RG_ERR_CANCELLED.
 * Called from the training thread, between stages, never mid-stage. */
typedef bool (*rg_progress_fn)(
    const char *stage,
    size_t completed,
    size_t total,
    void *user_data
);

typedef enum rg_split_scorer {
    RG_SPLIT_SCORER_CORRECTED_BIC = 0,
    /* Exact for integer categorical counts. Training returns
     * RG_ERR_UNSUPPORTED_OPTION if a scored split carries fractional mass. */
    RG_SPLIT_SCORER_MULTINOMIAL_NML = 1,
    RG_SPLIT_SCORER_DIRICHLET_MARGINAL = 2
} rg_split_scorer;

RG_API const char *rg_split_scorer_string(rg_split_scorer scorer);

/* How a multi-lect class split is charged for its added outcome parameters.
 *
 * `SISTER_TUPLE` scores one categorical over the sister tuples on the other
 * side of the correspondence -- the full list of `(lect, grapheme)` pairs --
 * and charges corrected BIC `(K-1)` where `K` counts those distinct tuples. A
 * tuple is an outcome crossed with which languages a set happened to cover and
 * any one-off reflex in any one sister, so `K` grows with the sample rather
 * than the structure and a conditioned correspondence recovered at two lects is
 * lost at three, four and five. It is retained only for comparison and is not
 * the default.
 *
 * `PER_SISTER_LECT` (the default) scores the split as a sum over
 * sister lects instead: for each sister lect present on both sides of the
 * split, the outcome is that lect's grapheme and the charge is `(K_q-1)*ln n_q`
 * with `K_q` bounded by the lect's inventory, not by arity. Coverage falls out
 * -- an observation that lacks a lect does not enter that lect's sub-count, and
 * a lect sitting entirely on one side contributes no contrast and is not
 * charged -- so a real conditioned correspondence is recovered at every arity.
 * This is what the pairwise stage already does, and why it works at every
 * arity. Only the multi-lect class search reads this; the
 * pairwise stages score real target graphemes and are unaffected. */
typedef enum rg_class_outcome_mode {
    RG_CLASS_OUTCOME_SISTER_TUPLE = 0,
    RG_CLASS_OUTCOME_PER_SISTER_LECT = 1
} rg_class_outcome_mode;

RG_API const char *rg_class_outcome_mode_string(rg_class_outcome_mode mode);

typedef struct rg_bic_config {
    /* The criterion used to compare one pooled categorical distribution with
     * the two distributions induced by an environment. The struct keeps its
     * historical name because its remaining fields are source-compatible
     * BIC-era discovery gates; `split_scorer` names what is actually scored. */
    rg_split_scorer split_scorer;
    /* Total mass of the symmetric Dirichlet prior used only by
     * DIRICHLET_MARGINAL. It is divided equally over the pooled observed
     * outcome alphabet and is deliberately separate from the feature-shaped
     * alignment-EM concentration. */
    double split_prior_concentration;
    double delta_bic_threshold;
    int min_split_observations;
    int max_split_depth;
    int min_chunk_observations;
    double long_range_delta_bic_threshold;
    int long_range_min_split_observations;
    double long_range_min_dominant_fraction;
    int cross_dim_max_iterations;
    int cross_dim_min_rule_count;
    /* A floor on P(target_value | environment). Defaults to 0.0, which is not
     * an oversight: a fixed fraction is not a measure of conditioning. With
     * two possible values 0.5 is chance; with ten it is overwhelming evidence,
     * and a threshold that rejects a value occurring at 0.4 against a base
     * rate of 0.05 throws away exactly the conditioned splits this stage
     * exists to find. Whether an environment conditions anything is decided by
     * cross_dim_delta_bic_threshold against the complementary environment.
     * Set this only to suppress weak rules in a report. */
    double cross_dim_min_rule_confidence;
    double cross_dim_delta_bic_threshold;
    /* Legacy BIC-only `2/(n-1)` addition. Off by default because it
     * has no derivation in the selected criterion; retained only to reproduce
     * earlier experimental models. Ignored by the other scorers. */
    bool multi_lect_bic_small_sample_correction;
    double multi_lect_min_commit_scale;
    /* How much of the search a split is charged for, on top of its K-1 added
     * outcome parameters. The penalty gains
     * `search_penalty_gamma * 2 * ln(distinct_partitions)`: the environment
     * that survives is the best of many, while two predicate names that make
     * the same unordered two-way division are one search opportunity.
     *
     * The default is 1.0, selected under a recorded restraint, null, power and
     * predictive protocol; it remains an empirical setting, not a probability
     * cutoff. Setting `permutation_count` and
     * `tune_search_penalty` replaces it with a value measured from the corpus
     * itself. */
    double search_penalty_gamma;
    /* How a multi-lect class split is charged for its added outcome
     * parameters -- see rg_class_outcome_mode. */
    rg_class_outcome_mode class_outcome_mode;
} rg_bic_config;

typedef enum rg_observation_unit {
    RG_OBSERVATION_UNIT_AUTO = 0,
    RG_OBSERVATION_UNIT_COGNATE_SET = 1,
    RG_OBSERVATION_UNIT_ETYMON_GROUP = 2,
    RG_OBSERVATION_UNIT_SOURCE_GROUP = 3,
    RG_OBSERVATION_UNIT_ALIGNED_SPAN = 4,
    RG_OBSERVATION_UNIT_ALIGNED_POSITION = 5,
    /* A connected component under caller-supplied etymon groups, source
     * groups and cognate ids. Predictive validation uses this unit so no two
     * rows that may describe one history can cross a train/test boundary. */
    RG_OBSERVATION_UNIT_DEPENDENCY_COMPONENT = 6
} rg_observation_unit;

RG_API const char *rg_observation_unit_string(rg_observation_unit unit);

/* The feature system is not here. It belongs to the context -- see
 * rg_context_use_system -- because it decides what a grapheme means before any
 * training option is consulted. It is deliberately not a training option. */
typedef struct rg_train_options {
    int max_chunk_size;
    double temperature;
    double concentration;
    int max_iter;
    double convergence_eps;
    double segment_weight;
    double displacement_weight;
    double tone_weight;
    double chunk_min_transparency;
    rg_bic_config bic;
    int bootstrap_n;
    int bootstrap_seed;
    /* Unit resampled by the bootstrap. AUTO uses etymon groups when at least
     * one is supplied and otherwise treats cognate sets as independent.
     * SOURCE_GROUP is opt-in: a publication is provenance, and automatically
     * collapsing a whole publication to one draw would usually be too coarse. */
    rg_observation_unit bootstrap_unit;
    /* Shuffled-baseline runs for the fit summary. Each one is a full training
     * run on a corpus whose pairings have been permuted, so this multiplies
     * training time; 0 (the default) skips it. */
    int permutation_count;
    int permutation_seed;
    /* Set the search charge from the corpus's own shuffled baseline rather
     * than from bic.search_penalty_gamma: the shuffles are trained with no
     * charge at all, and the level their spurious rules reach becomes the bar
     * the real run has to clear. Requires permutation_count > 0.
     *
     * This buys precision with recall, and the trade is real: a corpus whose
     * genuine conditioning is weak relative to its own noise will lose rules
     * that the fixed 0.5 keeps. It is off by default for that reason. */
    bool tune_search_penalty;
    /* Group-held-out predictive validation. Zero (the default) does not run
     * it. A positive value requests that many folds; folds are dependency
     * components, never aligned positions or expanded doublet rows. Every
     * fold trains and selects its environments afresh, then freezes that
     * decision list before aligning and scoring its held-out groups. */
    int predictive_folds;
    int predictive_seed;
    /* A completed validation needs at least this many independent components
     * in both the training and held-out partitions across the run. Smaller
     * corpora are published as descriptive-only, not forced through an
     * unstable estimate. */
    int predictive_min_groups;
    /* A reflex prediction whose largest categorical probability is below this
     * value abstains. Log loss and top-k coverage still include it; selective
     * top-1 coverage does not. */
    double predictive_abstention_threshold;
    int predictive_top_k;
    /* Run an IBM Model 1 pass before the DP alignment EM and use the learned
     * probabilities as the Dirichlet prior. 0 runs the step; negative
     * (the default) skips it. */
    int ibm1_prior;
    rg_progress_fn progress;
    void *progress_user_data;
} rg_train_options;

typedef struct rg_segment {
    const char *grapheme;
    const char *tone;
    const char *length;
    const char *stress;
} rg_segment;

typedef struct rg_form {
    const char *lect_id;
    const rg_segment *segments;
    size_t segment_count;
    const int *syllable_breaks;
    size_t syllable_break_count;
    const int *morpheme_breaks;
    size_t morpheme_break_count;
} rg_form;

typedef struct rg_feature_constraint {
    const char *feature;
    const char *value;
} rg_feature_constraint;

typedef struct rg_distance_constraint {
    int offset;
    rg_feature_constraint constraint;
} rg_distance_constraint;

typedef struct rg_context_spec {
    const char *position;
    const rg_feature_constraint *preceding;
    size_t preceding_count;
    const rg_feature_constraint *following;
    size_t following_count;
    /* Where the segment sits in its own morpheme -- "initial", "final",
     * "internal", or "only" -- derived from the morpheme boundaries the caller
     * supplied on the form. Absent when the form carries none. */
    const char *morphological;
    /* Which morpheme, counted from the start of the word: "0", "1", ... A rule
     * stated in it does not travel between a suffixing language and a
     * prefixing one, where the same index is a different thing. */
    const char *morpheme_index;
    const char *syllable_role;
    const char *syllable_position;
    const rg_distance_constraint *preceding_at_distance;
    size_t preceding_at_distance_count;
    const rg_distance_constraint *following_at_distance;
    size_t following_at_distance_count;
    const rg_feature_constraint *somewhere_preceding;
    size_t somewhere_preceding_count;
    const rg_feature_constraint *somewhere_following;
    size_t somewhere_following_count;
    const rg_feature_constraint *same_syllable;
    size_t same_syllable_count;
    const rg_feature_constraint *next_syllable;
    size_t next_syllable_count;
    const rg_feature_constraint *previous_syllable;
    size_t previous_syllable_count;
    /* Constraints on the segment the environment is *about*, rather than on
     * its neighbours. A conditioned correspondence rarely needs this -- the
     * segment is already the key -- but a cross-dimensional rule does: the
     * Middle Chinese register split conditions a target tone on the preceding
     * onset's voicing *and* on the source segment's own tone, and the second
     * of those is a statement about the segment itself. Suprasegmentals appear
     * here as ordinary features named "tone", "length" and "stress". */
    const rg_feature_constraint *self;
    size_t self_count;
    const rg_feature_constraint *self_stress;
    size_t self_stress_count;
    const rg_feature_constraint *preceding_stress;
    size_t preceding_stress_count;
    const rg_feature_constraint *following_stress;
    size_t following_stress_count;
} rg_context_spec;

typedef struct rg_feature_displacement {
    const char *feature;
    const char *from_value;
    const char *to_value;
} rg_feature_displacement;

/* How an interval was produced. A consumer cannot otherwise tell a closed-form
 * interval from a resampled one, and the two answer different questions: the
 * Wilson interval asks how much the count alone constrains the rate, the
 * bootstrap asks how much the rate moves when the corpus is resampled. */
typedef enum rg_uncertainty_method {
    /* No observations backed the estimate, so the interval is the whole
     * range. Not an error: an unobserved rate is unconstrained, not zero. */
    RG_UNCERTAINTY_NONE = 0,
    RG_UNCERTAINTY_WILSON = 1,
    RG_UNCERTAINTY_BOOTSTRAP = 2
} rg_uncertainty_method;

/* A two-sided interval on a rate in [0, 1] — the conditional probability the
 * count represents — plus the provenance needed to read it. n is the
 * denominator behind the point estimate, not the number of bootstrap samples. */
/* Whether a rule stands above what the search finds in this corpus with the
 * correspondences taken out of it.
 *
 * Every conditioned rule already reports a count, a contrast, a split score, a
 * search margin and an interval, and the corpus reports what its own shuffles
 * reach. Putting those together was left to the reader, which is a synthesis
 * a reader should not have to do on twenty-five rules -- and is exactly the
 * judgement the tool is for. */
typedef enum rg_rule_standing {
    /* No shuffled baseline was run, so there is nothing to stand above.
     * Not a verdict: pass permutation_count to get one. */
    RG_RULE_STANDING_UNMEASURED = 0,
    /* The rule's evidence carries a heavier search charge than the level the
     * same search reaches on the shuffled corpus. */
    RG_RULE_STANDING_ABOVE_NOISE = 1,
    /* It does not. The rule may still be true; what it is not is
     * distinguishable from an artefact of having looked. */
    RG_RULE_STANDING_WITHIN_NOISE = 2
} rg_rule_standing;

RG_API const char *rg_rule_standing_string(rg_rule_standing standing);

/* Predictive evidence is deliberately not a second name for in-sample split
 * evidence. A rule can describe the supplied corpus faithfully and still be
 * unconfirmed on unseen etymon/source groups. */
typedef enum rg_predictive_status {
    RG_PREDICTIVE_UNMEASURED = 0,
    RG_PREDICTIVE_DESCRIPTIVE_ONLY = 1,
    RG_PREDICTIVE_CONFIRMED = 2,
    RG_PREDICTIVE_NOT_CONFIRMED = 3
} rg_predictive_status;

RG_API const char *rg_predictive_status_string(rg_predictive_status status);

/* Proper categorical scores over held-out reflexes. The inventory is learned
 * from the training partition only; an unseen test reflex is scored through an
 * explicit unknown category and counted below, never inserted into the
 * candidate vocabulary. All rates are confidence-weighted. */
typedef struct rg_predictive_score {
    size_t observation_count;
    size_t unseen_reflex_count;
    double observation_weight;
    double log_loss;
    double top1_coverage;
    double top_k_coverage;
    double brier_score;
    double calibration_error;
    double abstention_rate;
    double accepted_top1_coverage;
} rg_predictive_score;

typedef struct rg_predictive_evidence {
    rg_predictive_status status;
    rg_observation_unit observation_unit;
    size_t folds;
    rg_predictive_score conditioned;
    rg_predictive_score unconditioned;
    /* Positive means conditioning reduced held-out log loss. */
    double log_loss_gain;
} rg_predictive_evidence;

typedef enum rg_null_model {
    RG_NULL_MODEL_NONE = 0,
    RG_NULL_MODEL_PAIRING_SHUFFLE = 1,
    RG_NULL_MODEL_WITHIN_BUCKET_SHUFFLE = 2,
    RG_NULL_MODEL_PARAMETRIC_UNCONDITIONED = 3,
    RG_NULL_MODEL_REAL_NEGATIVE_PANEL = 4
} rg_null_model;

RG_API const char *rg_null_model_string(rg_null_model model);

/* What a search decided, and how the decision stands.
 *
 * Every rule a discovery stage commits carries the same four facts, and they
 * were copy-pasted field for field into four published row types. Naming them
 * once means the standing verdict is computed in one place for every table,
 * and that an appender takes one argument rather than four positional doubles
 * and ints -- the two sibling appenders took the same trailing pair in
 * opposite orders, both a double and an int, so a swap compiled.
 *
 * Not on the aggregated tables. A segment, displacement, tonal or chunk row is
 * counted, not decided: there is no comparison behind it and no place in a
 * decision list for it to hold. Those rows carry an interval and nothing else.
 *
 * `uncertainty` is deliberately not here either. How well a rate is pinned is a
 * different question from whether the environment is real, it is asked of every
 * row including the aggregated ones, and rg_uncertainty_estimate already names
 * it. */
typedef struct rg_rule_evidence {
    /* Criterion that selected the split. */
    rg_split_scorer scorer;
    /* What the named criterion scored, in twice-negative-log-probability or
     * code-length units. Negative means the split paid for its complexity. */
    double delta_score;
    /* Compatibility alias for delta_score. New consumers must read
     * `scorer` and `delta_score`; under NML or a marginal likelihood this is
     * not a BIC value despite the historical field name. */
    double delta_bic;
    /* Where this rule sits in the decision list.
     *
     * Discovery is greedy and each rule is committed against what the earlier
     * ones left unexplained, so the rules are ordered and the order carries
     * meaning: a later rule refines, or applies within, what an earlier one
     * did not settle. The Middle Chinese register split reads as three
     * decisions in sequence -- source tone accounts for one class, then the
     * onset's voicing splits what it left -- and as an unordered set it reads
     * as three unrelated facts, one of them at confidence 0.50.
     *
     * Published tables are sorted by key so lookups can binary-search them,
     * which destroys that order; this preserves it. Rules committed by one
     * decision share an index. -1 where the row was not committed by a search:
     * an unconditioned class is aggregated, not decided. */
    int decision_index;
    /* How heavy a search charge this rule's evidence could carry and still
     * commit. Comparable across corpora, and comparable against the same
     * number measured on the corpus shuffled -- see rg_corpus_fit. */
    double search_margin;
    /* Set once the shuffled baseline has been measured; see rg_rule_standing. */
    rg_rule_standing standing;
    /* The comparison that supports `standing`; NONE when unmeasured. */
    rg_null_model standing_null;
    rg_predictive_evidence predictive;
} rg_rule_evidence;

typedef struct rg_uncertainty_estimate {
    double estimate;
    double lower;
    double upper;
    double n;
    double alpha;
    rg_uncertainty_method method;
    /* Sampling level and independent-unit count used for the interval. `n`
     * remains the weighted rate denominator; it need not be an effective
     * sample size. AUTO means a standalone interval caller did not name one. */
    rg_observation_unit observation_unit;
    double effective_n;
    /* Set when the row's environment was chosen by the same data the interval
     * is computed from. The interval then says how well the rate is pinned
     * *given* that environment, and not whether the environment is real -- for
     * which see `search_margin` against `rg_corpus_fit.null_search_margin`. */
    bool post_selection;
} rg_uncertainty_estimate;

typedef struct rg_link {
    const rg_segment *source;
    size_t source_count;
    const rg_segment *target;
    size_t target_count;
    rg_context_spec context;
    const rg_feature_displacement *feature_displacement;
    size_t feature_displacement_count;
    double confidence;
} rg_link;

typedef struct rg_form_pair {
    rg_form source;
    rg_form target;
    double weight;
} rg_form_pair;

typedef struct rg_cognate_form {
    const char *lect_id;
    rg_form form;
} rg_cognate_form;

typedef struct rg_cognate_set {
    const char *cognate_id;
    /* Optional caller-supplied dependence groups. The model never infers
     * either from ids or forms. Borrowed for the duration of training; a
     * loaded rg_corpus owns its copies. */
    const char *etymon_group;
    const char *source_group;
    const rg_cognate_form *forms;
    size_t form_count;
    double confidence;
} rg_cognate_set;

typedef struct rg_segment_count_row {
    const char *source;
    const char *target;
    double count;
    double source_total;
    /* How much mass answers to this target overall. Alignment is scored
     * symmetrically -- the geometric mean of the two conditional directions --
     * and that needs both denominators, so both are published. */
    double target_total;
    rg_uncertainty_estimate uncertainty;
} rg_segment_count_row;

/* One observed feature-displacement vector: the whole set of features gained
 * and lost between a source and target segment, counted as a unit. */
typedef struct rg_displacement_row {
    const rg_feature_displacement *items;
    size_t item_count;
    double count;
    double total;
    rg_uncertainty_estimate uncertainty;
} rg_displacement_row;

typedef struct rg_tonal_count_row {
    const char *source_tone;
    const char *target_tone;
    double count;
    double source_total;
    rg_uncertainty_estimate uncertainty;
} rg_tonal_count_row;

/* A correspondence conditioned on an environment, and which form the
 * environment is read from.
 *
 * A sound change is conditioned by the environment it happened in, which lives
 * in the ancestor -- and regulae refuses to decide which lect that is. So it
 * looks from both: a rule may name the source form's environment or the
 * target's, and `context_is_target` says which. The distinction is not
 * cosmetic. A change is only *visible* from the side that has the split: where
 * Greek reflects Grassmann's Law, Proto-Indo-European tʰ answers to Greek t in
 * some words and tʰ in others, while from the Greek side each segment has one
 * source and there is nothing to condition. Looking from one side only leaves
 * half of every pair's conditioning unreachable, and which half depends on
 * which lect happened to sort first. */
/* A conditioned correspondence, and the comparison that bought it.
 *
 * A conditioning claim is a comparison: this environment against its
 * complement. Publishing the environment's own count without the complement's
 * makes the row unreadable -- "s ~ r between vowels, count 14" says nothing
 * until you know what s does elsewhere. contrast_count is the same
 * correspondence in the observations where the environment does not hold, and
 * contrast_total is that side's denominator. `evidence.delta_score` is what
 * the named scorer assigned the split: negative means it paid for its
 * complexity and search. */
typedef struct rg_conditioned_segment_count_row {
    const char *source;
    const char *target;
    rg_context_spec context;
    bool context_is_target;
    double count;
    double source_total;
    double contrast_count;
    double contrast_total;
    rg_rule_evidence evidence;
    rg_uncertainty_estimate uncertainty;
} rg_conditioned_segment_count_row;

typedef struct rg_chunk_row {
    const rg_segment *source;
    size_t source_count;
    const rg_segment *target;
    size_t target_count;
    double cost;
    double count;
    /* Set when the target is the source's own segments in another order: a
     * reordering, not a set of substitutions. The segment table records the
     * matching pairs -- s answering s, k answering k -- so without this the
     * model would say nothing happened. */
    bool reordering;
    /* How readable this chunk is as one historical process, in [0, 1]. A chunk
     * that is short, balanced, decomposes without gaps and does not merely
     * wrap a smaller promoted chunk scores high; a long lopsided bundle of
     * unrelated changes that happened to pay for itself jointly scores low.
     * Always computed. `rg_train_options.chunk_min_transparency` is what acts
     * on it, and its default of zero drops nothing.
     *
     * A heuristic, and advisory: it is a claim about how easy the chunk is to
     * read, not about whether it is right. */
    double transparency;
    rg_uncertainty_estimate uncertainty;
} rg_chunk_row;

/* A claim that a feature on one lect's form goes with a suprasegmental value
 * on the other's: in `environment`, the other form's `dimension` takes `value`.
 *
 * `context_is_target` says which form the environment is read from, exactly as
 * it does on a conditioned correspondence, and the conditioned dimension is
 * then read from the other one. Both computational orientations of a pair are
 * searched. They ask different questions -- a lect that has merged the voicing
 * contrast has nothing for an environment to say, and a lect with no tone has
 * nothing to condition -- and searching only one made a finding depend on which
 * lect name sorted first: on `tone_chinese_like_clean`, a rule true at
 * confidence 1.00 by construction was committed when the conditioning lect
 * sorted first and not committed when it sorted second.
 *
 * Neither orientation is a direction of change. The environment sits in the
 * lect that still shows the conditioning contrast, which is a fact about what
 * each lect preserved, not about which one is ancestral.
 *
 * A conditioning environment is only conditioning if the complementary
 * environment behaves differently, so every row carries the contrast it was
 * measured against. `count`/`source_count`/`confidence` describe the
 * environment; `contrast_count`/`contrast_source_count`/`contrast_confidence`
 * describe everywhere else. Reading `confidence` alone will mislead: a rule
 * holding at 0.9 where the contrast also holds at 0.9 states the ambient
 * distribution, not a conditioned split. The row is published only when the
 * environment raises the value above its contrast.
 *
 * The environment is an rg_context_spec, the same type a conditioned
 * correspondence uses, so it can name more than one predicate. It has to: the
 * Middle Chinese register split conditions the tone on the preceding onset's
 * voicing *and* on that segment's own tone, and neither alone predicts it above
 * chance. A single-predicate row reported that rule at confidence 0.50 and
 * looked like a weak finding rather than half of one.
 *
 * `evidence.delta_score` is for the environment as a whole, not for this
 * value: it compares modelling the dimension separately inside and outside
 * under the named scorer. It is negative for every published row, and more
 * negative is stronger. */
typedef struct rg_cross_dimensional_row {
    rg_context_spec environment;
    /* The environment is read from the target form, and `dimension` from the
     * source form. Zero is the other way round. */
    int context_is_target;
    /* Whether the conditioned dimension is read from the same form that states
     * the environment. 0 is the cross-lect rule -- one lect's material predicts
     * the other lect's tone. 1 is lect-internal: an onset and the tone it
     * conditions in the same language, which is what tonogenesis leaves behind.
     * With `context_is_target`
     * this names the single form both are read from: target when
     * `context_is_target`, source otherwise. */
    int dimension_from_environment;
    const char *dimension;
    const char *value;
    /* Where the conditioned segment sits relative to the aligned position the
     * environment is stated at. */
    int position_offset;
    double count;
    double source_count;
    double confidence;
    double contrast_count;
    double contrast_source_count;
    double contrast_confidence;
    /* How many other features carve this rule's observations the same way -- a
     * different phonological dimension the corpus cannot tell apart from the
     * committed environment. 0 when the environment is uniquely identifiable;
     * >0 when it is confounded, and the stated environment is one of several the
     * data supports equally well. The count is of distinct alternative features
     * that induce the identical (or exactly complementary) partition of the same
     * observations, so it fires only on a perfect confound, never on a partial
     * correlation.
     *
     * A confounded environment stated at full confidence is a report's most
     * anchoring line: on a corpus where onset voicing and vowel frontness are
     * perfectly confounded, a reader given only the report adopts the one
     * predicate it names and misses the ambiguity, while a reader of the raw
     * table catches it. This flag exists to surface that. It says nothing about
     * whether the split is real -- that is
     * `evidence` and the contrast fields -- only whether its cause is pinned. */
    int environment_alternatives;
    rg_rule_evidence evidence;
    rg_uncertainty_estimate uncertainty;
} rg_cross_dimensional_row;

typedef struct rg_multi_pair_model_row {
    const char *lect_a;
    const char *lect_b;
    const rg_pairwise_model *model;
} rg_multi_pair_model_row;

/* One correspondence under one environment.
 *
 * A segment tuple may appear in **several** conditioned rows, and a consumer
 * that indexes this table by tuple has to expect that. Discovery is greedy and
 * each rule is committed against what the earlier ones left, so a change
 * conditioned by something that is not a natural class comes out as a decision
 * list: RUKI's *s retracts after r, u, k and i, which is four rules with one
 * outcome, and no single feature covers the four.
 *
 * It did not until 2026-08-17. Rows were merged on the tuple alone, so the
 * whole list collapsed into one and the survivor's environment was whichever
 * had the most constraints -- on the four-lect Romance corpus that discarded
 * 32 of the 58 splits the search had committed, and on Latin/Spanish it is the
 * difference between 20 conditioned classes and 25. Rows are merged now when a
 * later split's observations are a subset of an earlier row's, which is what a
 * second *description* of a rule looks like, and kept apart when it brings
 * observations no earlier row has, which is what the next rule in the list
 * looks like.
 *
 * `evidence.decision_index` is the order to read them in. */

/* The suprasegmentals one segment of a class carries, parallel to `graphemes`.
 *
 * Each is "" when the segment carries none. They are part of the class's
 * outcome identity, not of its environment: a class of the same graphemes
 * under a different tone is a different class, so `north:a central:a south:a`
 * with tones `⁵⁵ ⁵⁵ ³³` and the same tuple with `¹¹ ³³ ¹¹` are two rows. For
 * Sinitic, Hmong-Mien, Tai-Kadai, Bantu register and much of Otomanguean the
 * tone correspondence set *is* the correspondence set, and it earns a class row
 * at every arity rather than reaching only the per-pair count tables and
 * directed cross-dimensional rules and being dropped at reconciliation. */
typedef struct rg_suprasegmentals {
    const char *tone;
    const char *length;
    const char *stress;
} rg_suprasegmentals;

typedef struct rg_multi_class_row {
    int class_id;
    const char *const *lect_ids;
    const char *const *graphemes;
    /* Parallel to `graphemes`, one per segment; see rg_suprasegmentals. Carried
     * on an unconditioned class, where the suprasegmentals are part of the
     * reconciled outcome. NULL on a conditioned class: its outcome is the
     * segmental split, and a suprasegmental predicted by an environment is a
     * cross-dimensional rule (rg_cross_dimensional_row), not a class. Also NULL
     * on the empty model. When present, every segment has an entry, its fields
     * "" where the segment carries no tone, length or stress. */
    const rg_suprasegmentals *suprasegmentals;
    const rg_context_spec *contexts;
    size_t segment_count;
    double count;
    double confidence;
    /* The same segment tuple where the environment does not hold. Zero on an
     * unconditioned class, which has no environment and so no complement to
     * compare against -- as are that class's evidence fields, which was not
     * committed by a search.
     *
     * On a genuine conditioning split this is ~0 by construction: the split
     * exists precisely because the pivot takes a DIFFERENT reflex out of the
     * environment, so the same tuple barely recurs there. It is the wrong
     * number to judge the split by; `contrast_class_id` names the right one. */
    double contrast_count;
    /* The class holding the pivot's majority reflex where this row's
     * environment does not hold -- the comparison the split was scored on, and
     * the row a reader needs to judge it. On Verner the conditioned
     * `gothic:d ~ pgmc:θ` (before a vowel) points here at `gothic:d ~ pgmc:d`:
     * the contrast that makes the conditioning real; `contrast_class_id` links
     * the two, which an unrelated row would otherwise leave unconnected.
     *
     * -1 on an unconditioned class, and on a conditioned class whose complement
     * was empty or had no majority reflex. Indexes `class_id`, which is a
     * position in the unconditioned array or, past its end, the conditioned
     * array; the target is usually unconditioned (the pivot's aggregate other
     * reflex) but may be another conditioned class. */
    int contrast_class_id;
    /* The mass of that majority reflex in the complement -- the denominator the
     * split was scored against, local to this pivot's observations. Distinct
     * from the linked class's own `count`, which aggregates that tuple across
     * the whole corpus. Zero when `contrast_class_id` is -1. */
    double contrast_alternative_count;
    /* Rival conditioners at another position the corpus cannot tell this
     * conditioned class's environment from -- the same identifiability flag the
     * cross-dimensional row carries, here for a segment split.
     * 0 on an unconditioned class and on a conditioned one whose environment is
     * uniquely identifiable; >0 when a different neighbour's feature carves the
     * split the same way and the corpus cannot say which conditions it. */
    int environment_alternatives;
    rg_rule_evidence evidence;
    /* The distinct cognate sets this class rests on, each listed once however
     * many aligned positions in it realise the class. Owned by the model and
     * valid until it is freed; sorted by the order the sets were first reached,
     * which is the order they appear in the corpus.
     *
     * Read `supporting_cognate_count` wherever the question is whether a
     * correspondence recurs. `count` cannot answer it: it is aligned positions
     * weighted by cognate confidence, so a single word with a geminate or a
     * repeated segment reaches 2, and such a row is easily misread as two words
     * supporting it. This list gives each supporting set once, so its length is
     * the count of distinct sets -- almost certainly the number a consumer
     * wants. */
    const char *const *supporting_cognates;
    size_t supporting_cognate_count;
    rg_uncertainty_estimate uncertainty;
} rg_multi_class_row;

/* A cross-dimensional rule, and which pair of lects it was found in.
 *
 * The rule itself is exactly the pairwise row -- the multi-lect table is built
 * by lifting every pair's rows into one place, and the lifting used to be forty
 * lines copying fifteen identical fields across, in the same order, into a
 * struct that differed from its source by two strings.
 *
 * `source_lect` and `target_lect` name the pair's computational orientation,
 * which is the order the pair was trained in and nothing more. The lect the
 * environment is stated over is `rule.context_is_target ? target_lect :
 * source_lect`, and the conditioned dimension is on the other one. */
typedef struct rg_multi_cross_dimensional_row {
    const char *source_lect;
    const char *target_lect;
    rg_cross_dimensional_row rule;
} rg_multi_cross_dimensional_row;

/* One lect's part in a proposed event: the graphemes it contributes across the
 * member classes, and the features that name them as a class if any do. */
typedef struct rg_event_member {
    const char *lect_id;
    const char *const *graphemes;
    size_t grapheme_count;
    /* The suprasegmentals this lect carries across the member classes, or NULL
     * where it carries none.
     *
     * One value, not an array parallel to `graphemes`: every member class of an
     * event agrees with every other on the suprasegmentals at each slot, which
     * is what makes them one change rather than several landing together. A
     * tone shift over several vowels -- `a[¹¹] ~ a[³³]` beside `e[¹¹] ~ e[³³]`
     * -- is one event whose grapheme sets are `{a,e}` on both sides, and
     * without this the row reads as `{a,e} ~ {a,e}` and states nothing.
     *
     * A dimension neither lect writes, and a dimension only one of them writes,
     * are both NULL here: an asymmetry between transcriptions is not a change.
     * Owned. */
    const rg_suprasegmentals *suprasegmentals;
    /* Features carried by every grapheme above and by no other grapheme this
     * lect shows in the corpus -- what makes the set a class rather than a
     * list. Empty when no feature separates it, which is the common case
     * rather than a failure: no feature theory tested against attested active
     * classes expressed more than 71% of them (Mielke 2008), so a set that
     * cannot be named may still be a real class. `featurally_definable` on the
     * event says whether every lect managed it. */
    const char *const *class_features;
    size_t class_feature_count;
    /* The environment every member class states at this slot, and nothing
     * else. Empty when they share none, and when there is none to share.
     *
     * Intersected rather than taken from a member, because the members were
     * each searched on their own and one routinely carries a predicate the
     * others did not need: on Verner both members turn on primary stress in
     * Proto-Germanic and one also picked up "before a vowel" in Gothic.
     * Publishing that member's environment would state the incidental conjunct
     * as part of the law.
     *
     * An empty spec means different things on different axes, and `axis` on the
     * event says which: under RG_EVENT_AXIS_OUTCOME the members differ in
     * environment by design and the emptiness is the finding, while under
     * RG_EVENT_AXIS_DISPLACEMENT there was never an environment at all. Owned;
     * `rg_context_spec_constraint_count` is 0 when empty. */
    rg_context_spec context;
} rg_event_member;

/* What holds an event's members together -- and so what an empty
 * `rg_event_member.context` on it means. */
typedef enum rg_event_axis {
    /* One environment, several outcomes: a change across a class of segments.
     * The members' shared environment is on each member. Also the axis of a
     * conditioned change that grouped with nothing, where `class_id_count` is
     * 1 and the "shared" environment is simply that class's own. */
    RG_EVENT_AXIS_ENVIRONMENT = 0,
    /* One outcome, several environments: a change stated as a decision list.
     * The members differ in environment -- that is the axis -- so the shared
     * environment is normally empty and its emptiness is not a finding about
     * conditioning. Read the member classes through `class_ids` for the
     * disjunction. */
    RG_EVENT_AXIS_OUTCOME = 1,
    /* No environment: unconditioned classes held together by what they
     * displace. Nothing was conditioned, so nothing is stated. */
    RG_EVENT_AXIS_DISPLACEMENT = 2
} rg_event_axis;

/* Several conditioned classes that look like one change.
 *
 * A change applying to more than one segment is published as one class per
 * segment: on the natural-class fixture, voicing four continuants between
 * vowels comes out as four rows differing only in their graphemes, carrying
 * the same environment and the same score to three decimal places. They are
 * one event and no field said so, and the cost is not only readability --
 * the same evidence states the change at a search margin over ten as one rule
 * and under four as four rules.
 *
 * This groups them and stops. It does **not** decide that the pooled
 * description is the better one: scoring a pooled hypothesis against the
 * fragmented alternative is a model-selection question this does not answer,
 * and `testdata/soundlaws/natural_class.tsv` with its control is what any
 * answer has to be scored on. Every member class stays published exactly as
 * it was, so a consumer that disagrees ignores this table and loses nothing.
 *
 * Rows group along whichever of three axes holds them together: one
 * environment over several outcomes (a change across a class of segments), one
 * outcome over several environments (a change stated as a decision list), or,
 * where there is no environment, a shared feature displacement. A displacement
 * grouping asks for a displacement its members *share*, not an identical one --
 * Grimm's p~f, t~θ and k~x agree on stop→fricative and disagree on place --
 * held in check by requiring the group's intersected displacement to stay
 * non-empty and its members to read as one regular change from some lect's
 * side: distinct inputs, each the leading answer for its own segment.
 *
 * Retentions are not members. An identity row is evidence about a split and
 * `contrast_class_id` on the member row links to it; an event made of identity
 * rows would group what did not happen.
 *
 * A proposal, and the field name says so. */
typedef struct rg_proposed_event_row {
    const rg_event_member *members;
    size_t member_count;
    /* The classes this event proposes to join, in ascending order. Each is
     * published in its own right; nothing here replaces them.
     *
     * One id is a real answer and the common one: a change that happened to
     * land on a single segment is a change, and a table that showed only
     * groupings was empty on eighteen of the twenty-three corpora it had
     * nothing to say about -- vowel harmony, umlaut, rhotacism and most of the
     * conditioning ladder among them. Only a *conditioned* class earns a row
     * this way, having been committed by a search against a contrast; an
     * unconditioned class is an aggregate no search decided, and admitting
     * those would reprint the correspondence table. Filter on
     * `class_id_count > 1` for groupings alone. */
    const int *class_ids;
    size_t class_id_count;
    /* Aligned positions across every member class, and the distinct cognate
     * sets behind them. The pooled evidence, which is the number a reader
     * wants and the one no single member row carries. */
    double count;
    const char *const *supporting_cognates;
    size_t supporting_cognate_count;
    /* Whether every lect named its grapheme set with a feature. */
    bool featurally_definable;
    /* What holds the members together, and so how to read an empty
     * `rg_event_member.context`. */
    rg_event_axis axis;
    /* Rival conditioners the corpus cannot tell this event's environment from,
     * taken as the largest any member reports. 0 when the environment is
     * pinned, or when there is no environment.
     *
     * The same flag `rg_multi_class_row` carries (§4.3), lifted so that an
     * event stating an environment also states whether that environment is the
     * one doing the work. On `testdata/soundlaws/conditioned_confound.tsv` the
     * change is committed after a sonorant and "before a front vowel" carves
     * the same split, so the event names one conditioner and this says the
     * corpus cannot choose between them. */
    int environment_alternatives;
    /* The weakest member's evidence: the lowest `search_margin` and the
     * highest (least negative) `delta_score` any member class carries, so both
     * read as "every member clears at least this".
     *
     * Not a pooled score. Scoring the pooled description against the
     * fragmented one is the model-selection question this table does not
     * answer, and a number that looked pooled would be read as that answer.
     * The members' scores are close where the grouping is right -- one change
     * split per segment lands them within a fraction of each other -- but they
     * are not identical, and the spread is itself worth seeing: members far
     * apart are a grouping to look at twice.
     *
     * Zero on an event grouped from unconditioned classes, which carry no
     * search evidence to summarise. */
    double search_margin;
    double delta_score;
    /* The feature displacement shared by every member class, measured
     * between the first two member slots. What changed is the rule the
     * grouping implies, and this states it: `+long` means the first slot
     * carries `long` where the second does not; `close-mid: present→absent`
     * with `close: absent→present` means a height step. Empty when fewer
     * than two slots exist or the members disagree. Owned. */
    const rg_feature_displacement *shared_displacement;
    size_t shared_displacement_count;
} rg_proposed_event_row;

/* How one lect writes a sound that another lect in the same corpus writes as a
 * sequence -- see rg_find_transcription_drift. */
typedef enum rg_drift_kind {
    /* A grapheme merkmal itself splits: `tʃ` against `t ʃ`. */
    RG_DRIFT_SEGMENTATION = 0,
    /* A modifier letter the other source spells with an ordinary one: `tʰ`
     * against `t h`, `kʷ` against `k w`. */
    RG_DRIFT_MODIFIER = 1,
    /* Length written by doubling: `aː` against `a a`. */
    RG_DRIFT_LENGTH = 2
} rg_drift_kind;

typedef struct rg_transcription_drift_row {
    /* The lect that writes the sound whole, and the one that writes it apart. */
    const char *lect;
    const char *other_lect;
    const char *grapheme;
    /* What the other lect writes instead, space-separated. */
    const char *written_as;
    rg_drift_kind kind;
    /* Cognate sets where `lect` uses the grapheme and `other_lect` is present,
     * and how many of those have `other_lect` writing `written_as` as adjacent
     * segments.
     *
     * The second number is the evidence. An inventory asymmetry on its own is
     * what two different languages look like -- one of them lost its
     * affricates -- and what it is not is the same cognate sets showing the
     * pieces in the same order. A ratio near 1 is a transcription difference;
     * a low one is a sound change. */
    size_t forms;
    size_t corroborated;
} rg_transcription_drift_row;

typedef struct rg_cognate_outlier_row {
    const char *cognate_id;
    int pair_count;
    double cost_per_segment;
    double z_score;
    /* The caller-supplied confidence of the cognate set, carried through so a
     * reader of the residue can weigh a bad alignment against how sure the
     * judgement behind it was. 1.0 when the input stated none. */
    double confidence;
} rg_cognate_outlier_row;

/* How well the model fits the corpus it was trained on, and what the same
 * training does to data with the correspondences taken out of it.
 *
 * The class counts are not a measure of relatedness. They rise when the signal
 * is removed: a corpus whose pairings have been shuffled has no correspondences
 * left to find, and greedy splitting over a large candidate inventory finds
 * more environments in it, not fewer. `cost_per_segment` is the number that
 * separates the two -- it is strongly negative on real cognates and near zero
 * on shuffled ones -- and the permutation baseline is what makes it readable,
 * since its scale depends on the corpus.
 *
 * The baseline is off by default because it costs one full training run per
 * shuffle. `permutation_count == 0` means it was not run and every `null_`
 * field is zero. */
typedef struct rg_corpus_fit {
    double cost_per_segment;
    size_t scored_set_count;
    rg_split_scorer split_scorer;
    double split_prior_concentration;
    /* The observational hierarchy actually present in the input. A missing
     * group label makes that cognate set its own group; the missing counts
     * state that assumption instead of silently inventing membership. */
    size_t etymon_group_count;
    size_t source_group_count;
    size_t sets_without_etymon_group;
    size_t sets_without_source_group;
    rg_observation_unit bootstrap_unit;
    size_t bootstrap_effective_unit_count;
    size_t unconditioned_class_count;
    size_t conditioned_class_count;
    size_t permutation_count;
    double null_cost_per_segment_mean;
    double null_cost_per_segment_sd;
    /* Standard deviations between the observed fit and the shuffled baseline.
     * Strongly negative means the corpus aligns far better than chance. Zero
     * when no baseline was run, or when the baseline had no spread. */
    double cost_per_segment_z;
    double null_unconditioned_class_mean;
    double null_conditioned_class_mean;
    /* The search margin the shuffled corpus reached, at the quantile named
     * below: the level a rule has to clear to be saying more than the search
     * itself does. Rules at or under it were findable in data with no
     * correspondences left in it. */
    double null_search_margin;
    double null_search_margin_quantile;
    /* Forms with no nucleus of their own -- no vowel and no syllabic consonant
     * -- which were given one so the syllable predicates have something to
     * hold of. Vowelless words are real and their analyses differ, so a
     * syllable-conditioned rule on a corpus with many of these is resting on a
     * guess. */
    size_t inferred_nucleus_form_count;
    size_t syllabified_form_count;
    /* How many conditioned rules stand above the shuffled baseline, of how
     * many that were measured. Zero and zero when no baseline was run.
     *
     * These count the multi-lect rules: the conditioned classes, and the
     * cross-dimensional rules -- which already include every pair's, because
     * the multi-lect table is built by lifting them. */
    size_t rules_above_noise;
    size_t rules_measured;
    /* The same, for the conditioned correspondences each lect pair carries in
     * its own model. Reported separately rather than added to the pair above,
     * and the reason is that they are counted per pair.
     *
     * A rule visible in every pair of a four-lect corpus is six rules here and
     * one above, so merging them would make the ratio depend on how many lects
     * the corpus happens to sample -- on real_romance_4lect the denominator
     * goes from 26 to 253. Kept apart, each ratio compares like with like.
     *
     * Reading only the pair above will still mislead, which is why these exist.
     * On testdata/soundlaws/grassmann the multi-lect verdict is 0 of 4 -- and
     * the rule that is Grassmann's Law, Greek t answering PIE tʰ where an
     * aspirate follows, stands here at a margin of 1.57 against a baseline of
     * 1.32. A corpus whose one real finding is in this table read as a corpus
     * that had found nothing. */
    size_t pairwise_rules_above_noise;
    size_t pairwise_rules_measured;
    /* How many views of this corpus the pair counts above are counting, and
     * how many of them are not independent evidence.
     *
     * Pairwise rows are correlated views of one multi-lect corpus, and the
     * denominator of `pairwise_rules_measured` grows as though they were not:
     * a relation visible in every pair of four lects is six rows there and one
     * relation. `lect_count` and `pair_count` are the divisors that make that
     * readable.
     *
     * `duplicate_lect_count` counts lects whose forms repeat an earlier lect's
     * in every cognate set the two share -- the same variety sampled twice,
     * or one wordlist copied under two names. Each one adds pairs and no
     * evidence, and the pair count alone cannot show it.
     *
     * `missing_form_count` counts the lect-and-set slots with no form. They are
     * the opportunities the pair count implies and the corpus does not have.
     *
     * None of the three is a taxonomic claim. regulae has no family or area
     * labels and does not infer them: a duplicate here is an identical
     * wordlist, not a sister language. */
    size_t lect_count;
    size_t pair_count;
    size_t duplicate_lect_count;
    size_t missing_form_count;
    /* Whether the cognate sets align in one group or two.
     *
     * `cost_per_segment` above is a mean, and a mean says nothing about shape.
     * Two wordlists half of one of which was borrowed from the other produce a
     * perfectly good mean and a corpus that is not one thing: the loans align
     * beautifully and the native vocabulary does not, with no overlap between
     * them. Japanese and Chinese are the textbook case, and the Balkans,
     * mainland Southeast Asia, South Asia and much of Australia have pairs
     * like it.
     *
     * `cost_split_separation` is how far apart the two groups are, in pooled
     * standard deviations, at the best two-way split of the per-set costs.
     * `cost_split_fraction` is the share of sets on the worse-aligning side of
     * that split. Both are zero on a corpus with fewer than four scored sets.
     *
     * A unimodal sample still has a best split, so the number is never zero and
     * has to be read against something, and it has to be read *with* the
     * fraction. Measured on this repository's fixtures:
     *
     *     real pair corpora        2.7 - 3.0   fraction 39-65%
     *     chance (unrelated)       2.4         fraction 52%
     *     contaminated             7.2         fraction 11%
     *     contact (half borrowed)  6.3         fraction 50%
     *     stratum, diffusion      23.0, 31.5   fraction 50%
     *
     * Read together: a high separation with a *small* fraction is a tail of
     * sets that do not belong -- five bad judgements in forty-five. A high
     * separation at about half is a corpus that is two populations.
     *
     * **This is not a borrowing test and must not be quoted as one.** The two
     * highest numbers in that table are `stratum` and `diffusion`, where every
     * set is cognate and nothing was borrowed at all: half the words underwent
     * a change and half did not, so half align one way and half the other.
     * Nothing in the distribution of segments separates a loan stratum from an
     * inherited one -- that judgement needs the semantic fields, the direction
     * of cultural flow and the dates. What a high separation says is that the
     * corpus is not one thing, which is a reason to ask a different question
     * and not an answer to this one. */
    double cost_split_separation;
    double cost_split_fraction;
    /* Selection-nested, dependency-group-held-out evidence. Environment
     * discovery, feature-vocabulary construction and alignment training all
     * happen inside each training partition. `conditioned` is compared with
     * four frozen baselines. Pairwise scores contain both orientations;
     * leave-one-lect-out pools predictions from the other lects where at least
     * two are present. */
    rg_predictive_evidence predictive;
    size_t predictive_group_count;
    size_t predictive_folds_requested;
    size_t predictive_pair_orientations;
    size_t predictive_leave_one_lect_out_cases;
    size_t predictive_unscored_span_count;
    double predictive_abstention_threshold;
    int predictive_top_k;
    rg_predictive_score predictive_identity;
    rg_predictive_score predictive_inventory_frequency;
    rg_predictive_score predictive_feature_distance;
    rg_predictive_score predictive_leave_one_lect_out_conditioned;
    rg_predictive_score predictive_leave_one_lect_out_unconditioned;
} rg_corpus_fit;

RG_API const char *rg_version_string(void);
RG_API int rg_version_major(void);
RG_API int rg_version_minor(void);
RG_API int rg_version_patch(void);
RG_API uint32_t rg_abi_version(void);
RG_API const char *rg_status_string(rg_status status);
RG_API void rg_string_free(char *value);

RG_API void rg_bic_config_init_defaults(rg_bic_config *config);
RG_API void rg_train_options_init_defaults(rg_train_options *options);

/* The significance level every table in a trained model is published at.
 * 0.05 is a 95% interval. */
#define RG_DEFAULT_ALPHA 0.05

/* Wilson score interval on the rate count/total. Accepts fractional counts,
 * because confidence-weighted training produces them. total <= 0 yields the
 * full [0, 1] range with method RG_UNCERTAINTY_NONE. Returns
 * RG_ERR_UNSUPPORTED_OPTION for an alpha outside {0.10, 0.05, 0.01}. */
RG_API rg_status rg_wilson_interval(
    double count,
    double total,
    double alpha,
    rg_uncertainty_estimate *out
);

/* Distribution-free percentile interval over samples, each a rate drawn from
 * one resampled training run. total is the denominator behind the point
 * estimate and is recorded, not used. Samples are clamped to [0, 1]. An empty
 * sample set yields the full [0, 1] range with method RG_UNCERTAINTY_NONE.
 * Returns RG_ERR_UNSUPPORTED_OPTION for an unsupported alpha. */
RG_API rg_status rg_percentile_interval(
    const double *samples,
    size_t sample_count,
    double estimate,
    double total,
    double alpha,
    rg_uncertainty_estimate *out
);

/* Stable lowercase name of an interval method: "none", "wilson", "bootstrap".
 * Borrowed and static. */
RG_API const char *rg_uncertainty_method_string(rg_uncertainty_method method);

/* How a written word is cut into segments.
 *
 * Neither reading is universally right, so the choice belongs to whoever knows
 * the corpus. Orthographic honours the tie bar, which is how a transcription
 * says "this is one segment": "t͡ʃ" is one and untied "tʃ" is two. Longest
 * match instead asks the feature system what it recognises, which reads untied
 * "tʃ" and "kp" as single segments -- and, on the same rule, "kk", "dr", "ng"
 * and "st", which are sequences in most corpora regulae is pointed at. Latin
 * "bukka" comes out as b/u/kk/a under it. */
typedef enum rg_segmentation {
    RG_SEGMENT_ORTHOGRAPHIC = 0,
    RG_SEGMENT_SYSTEM_LONGEST_MATCH
} rg_segmentation;

RG_API rg_status rg_context_new_builtin(rg_context **out);
RG_API rg_status rg_context_set_segmentation(rg_context *ctx, rg_segmentation mode);
RG_API rg_segmentation rg_context_segmentation(const rg_context *ctx);
RG_API void rg_context_free(rg_context *ctx);
RG_API rg_status rg_context_use_system(rg_context *ctx, const char *system_name);
RG_API rg_status rg_context_system_name(const rg_context *ctx, const char **out);
RG_API rg_status rg_context_is_segment(const rg_context *ctx, const char *grapheme, bool *out);
/* Names the grapheme behind the most recent RG_ERR_UNKNOWN_GRAPHEME, and the
 * feature system that rejected it. Both are borrowed and valid until the next
 * failure or until the context is freed; grapheme is null if none has failed.
 * "unknown grapheme" without saying which one is unactionable on a corpus of
 * any size. */
RG_API void rg_context_last_error(
    const rg_context *ctx,
    const char **grapheme,
    const char **feature_system
);

/* Why a grapheme was refused. status is what a feature lookup would return:
 * RG_ERR_UNKNOWN_GRAPHEME for a sound the system does not cover,
 * RG_ERR_SOURCE_MARKER for CLDF/CLTS markup, RG_ERR_PARSE for a token the
 * system recognises and rejects, RG_OK for one that resolves.
 *
 * valid_prefix_bytes is the longest prefix that does resolve, which localises
 * the problem and is usually the repair; it is 0 when nothing resolves and the
 * whole length when nothing is wrong. offending_offset is the byte offset just
 * past that prefix, and offending holds the character there, empty when there
 * is none. There is deliberately no nearest-valid-grapheme suggestion: that
 * would be a guess presented as an answer. */
typedef struct rg_grapheme_diagnosis {
    rg_status status;
    size_t valid_prefix_bytes;
    size_t offending_offset;
    char offending[8];
} rg_grapheme_diagnosis;

/* Diagnoses one grapheme. Returns RG_OK unless the arguments are unusable: a
 * refused grapheme is the normal case and is reported in out->status. */
RG_API rg_status rg_context_diagnose(
    const rg_context *ctx,
    const char *grapheme,
    rg_grapheme_diagnosis *out
);

/* The diagnosis behind the most recent refusal, alongside the grapheme
 * rg_context_last_error names. Returns false when nothing has been refused. */
RG_API bool rg_context_last_diagnosis(
    const rg_context *ctx,
    rg_grapheme_diagnosis *out
);

RG_API rg_status rg_context_segment_distance(
    const rg_context *ctx,
    const char *a,
    const char *b,
    double *out
);
RG_API rg_status rg_context_grapheme_features(
    const rg_context *ctx,
    const char *grapheme,
    rg_feature_set **out
);

RG_API rg_status rg_context_segment_word(
    const rg_context *ctx,
    const char *word,
    rg_segment **out,
    size_t *out_count
);
RG_API void rg_segments_free(rg_segment *segments, size_t count);

RG_API size_t rg_feature_set_size(const rg_feature_set *features);
RG_API const char *rg_feature_set_get(const rg_feature_set *features, size_t index);
RG_API void rg_feature_set_free(rg_feature_set *features);

RG_API void rg_context_spec_init_empty(rg_context_spec *context);
RG_API size_t rg_context_spec_constraint_count(const rg_context_spec *context);
RG_API rg_status rg_context_spec_is_subset(
    const rg_context_spec *subset,
    const rg_context_spec *other,
    bool *out
);

RG_API rg_status rg_score_link(
    const rg_context *ctx,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    double *out
);
RG_API rg_status rg_score_link_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    double *out
);
RG_API rg_status rg_compute_displacement(
    const rg_context *ctx,
    rg_segment source,
    rg_segment target,
    rg_feature_displacement **out,
    size_t *out_count
);
RG_API void rg_feature_displacement_free(rg_feature_displacement *items, size_t count);

RG_API rg_status rg_compute_syllable_breaks(
    const rg_context *ctx,
    const rg_form *form,
    int **out,
    size_t *out_count
);
RG_API void rg_syllable_breaks_free(int *breaks);

RG_API rg_status rg_align_forms(
    const rg_context *ctx,
    const rg_form *source,
    const rg_form *target,
    int max_chunk_size,
    rg_alignment **out
);
RG_API rg_status rg_align_forms_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_form *source,
    const rg_form *target,
    int max_chunk_size,
    rg_alignment **out
);
RG_API void rg_alignment_free(rg_alignment *alignment);
RG_API size_t rg_alignment_link_count(const rg_alignment *alignment);
RG_API const rg_link *rg_alignment_link_at(const rg_alignment *alignment, size_t index);
RG_API rg_status rg_alignment_cost(const rg_context *ctx, const rg_alignment *alignment, double *out);
RG_API rg_status rg_alignment_cost_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_alignment *alignment,
    double *out
);

RG_API rg_status rg_train_pairwise_segment_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
);
RG_API rg_status rg_train_pairwise(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
);
RG_API void rg_pairwise_model_free(rg_pairwise_model *model);
RG_API const rg_segment_count_row *rg_pairwise_model_segment_counts(const rg_pairwise_model *model, size_t *count);
RG_API const rg_displacement_row *rg_pairwise_model_displacements(const rg_pairwise_model *model, size_t *count);
RG_API const rg_tonal_count_row *rg_pairwise_model_tonal_counts(const rg_pairwise_model *model, size_t *count);
/* Correspondences to nothing: the commonest sound change, loss, and the shape
 * the 1-to-1 table above cannot state. A loss is a row `g ~ ∅` (RG_GAP_GRAPHEME
 * on the target), an epenthesis a row `∅ ~ g`; `count / source_total` (or
 * `count / target_total` for an epenthesis, the kept side) is the rate the
 * grapheme is dropped or inserted. These are unconditioned correspondences like
 * any other -- they are kept out of `rg_pairwise_model_segment_counts` only
 * because that table also drives alignment scoring and must stay 1-to-1. Not
 * counted here are non-one-to-one links that are not a pure loss (a 2-to-1
 * fusion): those are chunk-table rows. */
RG_API const rg_segment_count_row *rg_pairwise_model_null_correspondences(const rg_pairwise_model *model, size_t *count);
RG_API const rg_conditioned_segment_count_row *rg_pairwise_model_conditioned_segment_counts(const rg_pairwise_model *model, size_t *count);
RG_API const rg_chunk_row *rg_pairwise_model_chunks(const rg_pairwise_model *model, size_t *count);
RG_API const rg_cross_dimensional_row *rg_pairwise_model_cross_dimensional_rows(const rg_pairwise_model *model, size_t *count);

typedef struct rg_tsv_load_options {
    const char *cognate_id_column;
    const char *lect_id_column;
    const char *segments_column;
    const char *alignment_column;
    const char *confidence_column;
    /* Optional dependence metadata, defaulting to columns named
     * "etymon_group" and "source_group" when present. */
    const char *etymon_group_column;
    const char *source_group_column;
    /* Per-segment tone, whitespace-separated and positionally parallel to the
     * segments cell: "1 0 3" tones the first, second and third segment. "-" or
     * an empty token leaves that segment untoned, which is how a tone-bearing
     * corpus marks its consonants. Defaults to the "tone" column when present.
     *
     * Tone gets its own column rather than riding inline on the grapheme
     * because merkmal's segmenter does not merge Chao digits — "pa1" segments
     * as p, a, 1, and the stray digit is then an unknown grapheme. A separate
     * column also keeps the tone out of feature lookup entirely, which is
     * where it belongs: tone is a suprasegmental, not part of the grapheme. */
    const char *tone_column;
    /* Per-segment stress, laid out like tone: one value per segment, "-" for a
     * segment that carries none. Defaults to the "stress" column when present.
     *
     * A corpus can also write the IPA marks in the word, where the segmenter
     * lifts them onto the syllable nucleus. The column is for corpora that
     * record stress as an annotation rather than in the transcription, and for
     * anything the marks cannot express. */
    const char *stress_column;
    /* Morpheme boundaries as segment indices, default "breaks". */
    const char *morpheme_breaks_column;
    /* Per-segment length values, default "length". A corpus may instead write
     * length into the grapheme as `aː`, which merkmal reads as its own segment
     * carrying the `long` feature; that is the right shape where length is
     * contrastive. This column is for treating it as a dimension -- something
     * that happens to a vowel -- so a lengthening can be found as a rule
     * rather than as a correspondence between two different vowels. */
    const char *length_column;
    /* Syllable boundaries as segment indices, default "syllables". Supplying
     * them overrides the sonority syllabifier, which is language-agnostic by
     * design and will be wrong wherever a language's phonotactics are not. */
    const char *syllable_breaks_column;
} rg_tsv_load_options;

/* Wide-format TSV: one row per cognate, one column per lect, cells holding
 * whole unsegmented words. This is the shape linguistic data is actually
 * written in; the generic long-format loader wants pre-segmented input.
 * A "<lect>_breaks" column supplies morpheme boundary indices for that lect. */
typedef struct rg_wide_load_options {
    const char *cognate_id_column;
    const char *confidence_column;
    const char *etymon_group_column;
    const char *source_group_column;
    const char *const *lect_columns;
    size_t lect_column_count;
} rg_wide_load_options;

typedef struct rg_gled_load_options {
    const char *family;
    const char *const *doculects;
    size_t doculect_count;
    int min_lects;
} rg_gled_load_options;

typedef struct rg_arcaverborum_load_options {
    const char *dataset;
    const char *family;
    const char *const *language_ids;
    size_t language_id_count;
    int min_lects;
    /* Zero selects the delimiter from the file extension when loading a path,
     * and comma when parsing text. */
    char delimiter;
} rg_arcaverborum_load_options;

/* Why a load failed, and where.
 *
 * The caller owns the storage, which is the whole point: a failed load returns
 * no corpus to hang a message on, and a caller-owned diagnosis is thread-safe
 * and never stale between calls -- the alternative, a process-wide buffer, is
 * neither.
 *
 * Every load and parse entry point clears this on entry when one is supplied,
 * and passing null asks for no reporting. line is 1-based, and 0 when the
 * failure is not tied to one. message is empty when the loader recorded no
 * detail beyond the status -- "parse error" with no line is unactionable on a
 * corpus of any size, which is the first thing a new user meets. */
typedef struct rg_load_diagnosis {
    size_t line;
    char message[256];
} rg_load_diagnosis;

RG_API rg_status rg_corpus_load_tsv(
    const char *path,
    const rg_tsv_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_load_wide_tsv(
    const rg_context *ctx,
    const char *path,
    const rg_wide_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_load_gled(
    const char *path,
    const rg_gled_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_load_arcaverborum(
    const char *path,
    const rg_arcaverborum_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_from_pairs(
    const rg_form_pair *pairs,
    size_t pair_count,
    const char *lect_a,
    const char *lect_b,
    const char *cognate_id_prefix,
    rg_corpus **out
);
/* Parse variants take the corpus as text rather than a path, so a caller that
 * already holds the data needs no temporary file. The WebAssembly build links
 * without a filesystem and uses these exclusively. */
RG_API rg_status rg_corpus_parse_tsv(
    const char *text,
    const rg_tsv_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_parse_wide_tsv(
    const rg_context *ctx,
    const char *text,
    const rg_wide_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_parse_gled(
    const char *text,
    const rg_gled_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_parse_arcaverborum(
    const char *text,
    const rg_arcaverborum_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);

RG_API void rg_corpus_free(rg_corpus *corpus);
RG_API size_t rg_corpus_cognate_count(const rg_corpus *corpus);
/* Cognate sets in which some lect contributed more than one reflex, and how
 * many extra sets the corpus was expanded into as a result.
 *
 * A doublet is a fact about a language, not an error in a file: it is 3.2% of
 * cognate-set members across the Lexibank datasets with expert judgements, and
 * near 10% in some Austronesian ones. The corpus carries one set per
 * combination of reflexes, each with its share of the original confidence, so
 * both reflexes are counted and neither is invented. These say how often that
 * happened. */
RG_API size_t rg_corpus_doublet_set_count(const rg_corpus *corpus);
RG_API size_t rg_corpus_doublet_expansion_count(const rg_corpus *corpus);

RG_API const rg_cognate_set *rg_corpus_cognates(const rg_corpus *corpus);
RG_API const rg_cognate_set *rg_corpus_cognate_at(const rg_corpus *corpus, size_t index);

/* Formatting helpers. The strings these produce are diagnostics meant to be
 * read; iterate the model accessors for anything a program depends on. Every
 * returned string is caller-owned and freed with rg_string_free. */
typedef struct rg_format_model_options {
    int top_segments;
    int top_displacements;
    double min_count;
    int top_chunks;
    int top_classes;
} rg_format_model_options;

RG_API void rg_format_model_options_init_defaults(rg_format_model_options *options);
RG_API char *rg_format_segments(const rg_segment *segments, size_t count);
RG_API char *rg_format_link(const rg_link *link);
RG_API char *rg_format_alignment(const rg_alignment *alignment);
RG_API char *rg_format_pairwise_model(
    const rg_pairwise_model *model,
    const rg_format_model_options *options
);
/* The machine-readable summary the CLI prints by default: one tab-separated
 * line per class and per cross-dimensional rule, environments rendered with
 * sorted constraint lists so the same environment always spells the same. A
 * stable contract; caller frees with rg_string_free. */
RG_API char *rg_format_multi_model_summary(const rg_multi_model *model);

/* The per-pair learned tables, tab-separated. Caller frees with
 * rg_string_free. */
RG_API char *rg_format_pairwise_tables(const rg_multi_model *model);

RG_API char *rg_format_multi_model(
    const rg_multi_model *model,
    const rg_format_model_options *options
);
RG_API char *rg_describe_multi_class(
    const rg_multi_model *model,
    const char *lect_id,
    const char *grapheme
);

/* Renders transcription-drift rows as a human-readable string for printing
 * above a model report. Caller owns the string (rg_string_free). Returns an
 * empty string when count is 0. */
RG_API char *rg_format_drift(
    const rg_transcription_drift_row *rows,
    size_t count
);

/* Renders a trained model as JSON. Caller owns the string (rg_string_free).
 * This is a debug/inspection snapshot, labelled as such in the payload; it is
 * not the interchange format, which carries ensembles rather than a single
 * maximum-a-posteriori system. */
RG_API char *rg_model_to_json(
    const rg_context *ctx,
    const rg_multi_model *model,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    bool include_alignments,
    bool include_outliers
);

/* Renders the trained model's promoted multi-segment chunks, per lect pair,
 * as JSON: { pairs: [{ source_lect, target_lect, chunks: [{ source, target,
 * count, reordering, transparency }] }] }. Kept out of rg_model_to_json so
 * the documented model export stays byte-stable; this is the display surface
 * the page's multi-segment pane reads. Caller owns the string. */
RG_API char *rg_multi_model_pair_chunks_json(const rg_multi_model *model);

/* Renders rg_find_transcription_drift's rows as JSON: { drift: [{ lect,
 * other_lect, grapheme, written_as, corroborated, forms }] }. Caller owns
 * the string. */
RG_API char *rg_corpus_drift_json(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count
);

/* Renders an error as the same JSON envelope a successful call uses, so a
 * caller has one shape to handle. Caller owns the string. */
RG_API char *rg_error_to_json(rg_status status, const char *detail);

/* Reads a flat options object. Unknown keys are rejected rather than ignored,
 * with the offending key named in error_detail, which may be null. */
RG_API rg_status rg_train_options_from_json(
    const char *text,
    rg_train_options *out,
    char *error_detail,
    size_t error_detail_size
);

/* Renders a segmented word, so a caller can show how input will be read. */
RG_API char *rg_segments_to_json(const rg_segment *segments, size_t count);

RG_API rg_status rg_train_model(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_multi_model **out
);
RG_API void rg_multi_model_free(rg_multi_model *model);
/* Cognate sets that carried fewer than two forms, and so contributed no
 * correspondence. Not an error: a form whose cognates are in lects the corpus
 * did not sample has nothing to align against, and every cognate-coded
 * wordlist has some. Worth reading as a proportion of the corpus, because a
 * high one means the lect sample, not the method, is deciding the result. */
RG_API size_t rg_multi_model_unpaired_set_count(const rg_multi_model *model);
/* Borrowed; valid while the model lives. Never NULL for a trained model. */
RG_API const rg_corpus_fit *rg_multi_model_fit(const rg_multi_model *model);
/* Every published table is handed out whole: the rows and how many, borrowed
 * and valid while the model lives. A table is sorted by a stable key at
 * publication, so a caller may binary-search it; the order rules were decided
 * in is on each row's evidence, not in the table's order.
 *
 * They were a count function and an index function each, twenty-two of them,
 * every one a null check and an array index -- so each caller wrote its own
 * loop calling a function per row and dereferencing the result without
 * checking the null those functions document. */
RG_API const char *const *rg_multi_model_lects(const rg_multi_model *model, size_t *count);
RG_API const rg_multi_class_row *rg_multi_model_unconditioned_classes(
    const rg_multi_model *model,
    size_t *count
);
RG_API const rg_multi_class_row *rg_multi_model_conditioned_classes(
    const rg_multi_model *model,
    size_t *count
);
/* Classes that look like one change, grouped. Borrowed, valid while the model
 * lives. See rg_proposed_event_row: a proposal, not a verdict, and the member
 * classes remain published.
 *
 * Ordered by pooled `count`, heaviest first, ties broken on the first member's
 * class id. Not a binary-searchable key like the other tables': what a reader
 * wants from this one is which grouping carries the most evidence, and the
 * grouping axis it came from -- which is what the order used to reflect -- is
 * not a claim about strength. */
RG_API const rg_proposed_event_row *rg_multi_model_proposed_events(
    const rg_multi_model *model,
    size_t *count
);

RG_API const rg_multi_cross_dimensional_row *rg_multi_model_cross_dimensional_rows(
    const rg_multi_model *model,
    size_t *count
);
/* The pair models keep an index accessor: the table holds each pair's owned
 * strings alongside the published row, so there is no array of rows to hand
 * out. */
RG_API size_t rg_multi_model_pair_model_count(const rg_multi_model *model);
RG_API const rg_multi_pair_model_row *rg_multi_model_pair_model_at(
    const rg_multi_model *model,
    size_t index
);
RG_API rg_status rg_find_cognate_outliers(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_multi_model *model,
    const rg_train_options *options,
    int top_k,
    int max_chunk_size,
    rg_cognate_outlier_row **out,
    size_t *out_count
);
RG_API void rg_cognate_outlier_rows_free(rg_cognate_outlier_row *rows, size_t count);

/* Whether two lects in this corpus are transcribed by sources that disagree
 * about where a segment ends.
 *
 * The commonest way a comparative dataset goes wrong and the least visible.
 * Both transcriptions are valid IPA, every grapheme resolves, `rg_corpus_load_*`
 * accepts them and the model that comes out reports deaffrication, loss of
 * aspiration and loss of vowel length as clean, well-supported correspondences.
 * None of it happened. The shuffled baseline does not catch it either and
 * cannot: a baseline separates a pattern from chance, and this pattern is
 * perfectly systematic, which is what a sound law is.
 *
 * Reports; never refuses, and is not a verdict. A corpus can honestly contain
 * one language with affricates and one without, and only the person who
 * assembled it can tell that from two sources disagreeing. `corroborated`
 * against `forms` is the number to read: it says how often the other lect
 * actually writes the pieces where this one writes the whole.
 *
 * Rows come strongest-evidence first, ties broken on their own text so two
 * runs over one corpus print the same thing. Free with
 * rg_transcription_drift_rows_free. Zero rows is the common case and is not an
 * error. */
RG_API rg_status rg_find_transcription_drift(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    rg_transcription_drift_row **out,
    size_t *out_count
);
RG_API void rg_transcription_drift_rows_free(rg_transcription_drift_row *rows, size_t count);

/* ---- Translation table (IBM Model 1) ----
 *
 * An unconstrained probability table P(t|s) learned by EM from a corpus of
 * form pairs. Each source position independently maps to a target position or
 * to null; there is no monotonicity constraint, so disjoint mappings (e.g.
 * reduplication, infixation, long-range metathesis) fall out naturally.
 *
 * A table can also be built from merkmal priors alone (no corpus), in which
 * case P(t|s) is the softmax of the feature distance. That table answers
 * queries lazily unless precompute is called. */

typedef enum rg_direction {
    RG_DIR_FORWARD = 0,
    RG_DIR_BACKWARD,
    RG_DIR_SYMMETRIC
} rg_direction;

typedef struct rg_translation_table_options {
    /* EM iteration cap. 0 = dynamic default (converge within threshold). */
    int max_iterations;
    /* Prior probability of mapping to null (deletion). Negative = derive from
     * the gap cost (RG_DEFAULT_GAP_COST). */
    double null_prior;
    /* Relative change in corpus log-likelihood below which EM stops. */
    double convergence_threshold;
    /* Softmax temperature for the merkmal-distance prior. 0 = 1.0. */
    double temperature;
} rg_translation_table_options;

RG_API void rg_translation_table_options_init_defaults(
    rg_translation_table_options *options
);

typedef struct rg_translation_assignment {
    size_t source_index;
    size_t target_index;
    double probability;
} rg_translation_assignment;

typedef struct rg_translation_alignment {
    rg_translation_assignment *assignments;
    size_t count;
    double score;
} rg_translation_alignment;

/* Train on a corpus of form pairs. Both directions (forward and backward) are
 * trained so that symmetric queries are available. */
RG_API rg_status rg_train_translation_table(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_translation_table_options *options,
    rg_translation_table **out
);

/* Build from merkmal priors only (no corpus). Queries are lazy: each call to
 * rg_translation_probability computes the softmax on the fly. Call
 * rg_translation_table_precompute to materialise the full matrix. */
RG_API rg_status rg_translation_table_from_prior(
    const rg_context *ctx,
    rg_translation_table **out
);

/* Materialise the full |V_s| x |V_t| probability matrix for a prior-only
 * table, so repeated queries are O(1) lookups. */
RG_API rg_status rg_translation_table_precompute(
    rg_translation_table *table,
    const char **source_inventory,
    size_t source_count,
    const char **target_inventory,
    size_t target_count
);

/* Query a single-pair probability. direction selects P(t|s), P(s|t), or their
 * geometric mean. Returns 0.0 for unknown graphemes. */
RG_API double rg_translation_probability(
    const rg_translation_table *table,
    const char *source,
    const char *target,
    rg_direction direction
);

/* Per-position Viterbi assignment for a specific form pair. Each source
 * position is assigned to the target position (or null) with highest
 * probability under the chosen direction. RG_DIR_SYMMETRIC (geometric
 * mean of forward and backward) is the recommended default. */
RG_API rg_status rg_translation_align(
    const rg_translation_table *table,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    rg_direction direction,
    rg_translation_alignment **out
);

/* Sum of log-probabilities over the Viterbi assignment, divided by the
 * source length. Lower (more negative) means worse. */
RG_API rg_status rg_translation_score(
    const rg_translation_table *table,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    rg_direction direction,
    double *score
);

RG_API void rg_translation_table_free(rg_translation_table *table);
RG_API void rg_translation_alignment_free(rg_translation_alignment *alignment);

/* The source and target vocabularies the table was trained on (or precomputed
 * for). Returns 0 for a lazy prior-only table that has not been precomputed. */
RG_API const char *const *rg_translation_table_source_vocab(
    const rg_translation_table *table,
    size_t *count
);
RG_API const char *const *rg_translation_table_target_vocab(
    const rg_translation_table *table,
    size_t *count
);

#ifdef __cplusplus
}
#endif

#endif
