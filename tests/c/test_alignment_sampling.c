/* Sampled alignments (forward filtering, backward sampling).
 *
 * The draw must (1) reduce to the cheapest path as the temperature goes to 0,
 * (2) repeat for one seed, (3) vary across seeds on an ambiguous pair, and
 * (4) choose paths in proportion to exp(-cost / T). */
#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SEG 8

typedef struct word {
    rg_segment segs[MAX_SEG];
    size_t n;
} word;

static char *dup(const char *s) {
    char *out = (char *)malloc(strlen(s) + 1);
    assert(out != 0);
    strcpy(out, s);
    return out;
}

static word make(const char *text) {
    word w;
    char *copy = dup(text);
    char *tok = strtok(copy, " ");
    memset(&w, 0, sizeof(w));
    while (tok != 0) {
        assert(w.n < MAX_SEG);
        w.segs[w.n++].grapheme = dup(tok);
        tok = strtok(0, " ");
    }
    free(copy);
    return w;
}

static rg_form form(const char *lect, const word *w) {
    rg_form f;
    memset(&f, 0, sizeof(f));
    f.lect_id = lect;
    f.segments = w->segs;
    f.segment_count = w->n;
    return f;
}

/* An alignment as text: each link as source/target graphemes. */
static void signature(const rg_alignment *a, char *out, size_t size) {
    size_t i;
    size_t k;
    out[0] = '\0';
    for (i = 0; i < rg_alignment_link_count(a); i++) {
        const rg_link *link = rg_alignment_link_at(a, i);
        strncat(out, "[", size - strlen(out) - 1);
        for (k = 0; k < link->source_count; k++) {
            strncat(out, link->source[k].grapheme, size - strlen(out) - 1);
        }
        strncat(out, "/", size - strlen(out) - 1);
        for (k = 0; k < link->target_count; k++) {
            strncat(out, link->target[k].grapheme, size - strlen(out) - 1);
        }
        strncat(out, "]", size - strlen(out) - 1);
    }
}

int main(void) {
    static const char *corpus[][2] = {
        {"a k t a", "a t t a"}, {"o k t o", "o t t o"}, {"a k t u", "a t t u"},
        {"p a t a", "p a t a"}, {"k a p a", "k a p a"}, {"t a k a", "t a k a"},
        {"a p t a", "a t t a"}, {"m a k t o", "m a t t o"}, {"n o k t e", "n o t t e"},
        {"s a p o", "s a p o"}, {"l u p a", "l u p a"}, {"a k t e", "a t e"},
    };
    const size_t corpus_count = sizeof(corpus) / sizeof(corpus[0]);
    word src[32];
    word tgt[32];
    rg_form_pair pairs[32];
    rg_context *ctx = 0;
    rg_pairwise_model *model = 0;
    rg_train_options options;
    rg_alignment *best = 0;
    char best_sig[512];
    char sig[512];
    size_t i;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    for (i = 0; i < corpus_count; i++) {
        src[i] = make(corpus[i][0]);
        tgt[i] = make(corpus[i][1]);
        pairs[i].source = form("A", &src[i]);
        pairs[i].target = form("B", &tgt[i]);
        pairs[i].weight = 1.0;
    }
    rg_train_options_init_defaults(&options);
    assert(rg_train_pairwise(ctx, pairs, corpus_count, &options, &model) == RG_OK);

    /* The ambiguous pair: k t ~ t t, and a k t e ~ a t e with a loss. */
    {
        word s = make("a k t e");
        word t = make("a t e");
        rg_form fs = form("A", &s);
        rg_form ft = form("B", &t);
        rg_train_options sampled = options;
        char seen[16][512];
        double cost[16];
        int counts[16];
        size_t distinct = 0;
        int draws = 4000;
        int d;

        assert(rg_align_forms_with_model(ctx, model, &options, &fs, &ft, 0, &best) == RG_OK);
        signature(best, best_sig, sizeof(best_sig));

        /* (1) A vanishing temperature is the cheapest path. */
        sampled.alignment_sampling_temperature = 1e-6;
        for (d = 0; d < 20; d++) {
            rg_alignment *a = 0;
            sampled.alignment_sampling_seed = d;
            assert(rg_align_forms_with_model(ctx, model, &sampled, &fs, &ft, 0, &a) == RG_OK);
            signature(a, sig, sizeof(sig));
            assert(strcmp(sig, best_sig) == 0);
            rg_alignment_free(a);
        }

        /* (2) One seed, one draw. */
        sampled.alignment_sampling_temperature = 1.0;
        sampled.alignment_sampling_seed = 7;
        {
            rg_alignment *a = 0;
            rg_alignment *b = 0;
            char sa[512];
            char sb[512];
            assert(rg_align_forms_with_model(ctx, model, &sampled, &fs, &ft, 0, &a) == RG_OK);
            assert(rg_align_forms_with_model(ctx, model, &sampled, &fs, &ft, 0, &b) == RG_OK);
            signature(a, sa, sizeof(sa));
            signature(b, sb, sizeof(sb));
            assert(strcmp(sa, sb) == 0);
            rg_alignment_free(a);
            rg_alignment_free(b);
        }

        /* (3) and (4): draws over seeds. */
        memset(counts, 0, sizeof(counts));
        for (d = 0; d < draws; d++) {
            rg_alignment *a = 0;
            size_t k;
            sampled.alignment_sampling_seed = d;
            assert(rg_align_forms_with_model(ctx, model, &sampled, &fs, &ft, 0, &a) == RG_OK);
            signature(a, sig, sizeof(sig));
            for (k = 0; k < distinct; k++) {
                if (strcmp(seen[k], sig) == 0) {
                    break;
                }
            }
            if (k == distinct && distinct < 16) {
                strcpy(seen[distinct], sig);
                assert(rg_alignment_cost_with_model(ctx, model, &options, a, &cost[distinct]) == RG_OK);
                distinct++;
            }
            if (k < 16) {
                counts[k]++;
            }
            rg_alignment_free(a);
        }
        assert(distinct >= 2);
        {
            /* The two most frequent paths: their frequency ratio against the
             * cost difference, within sampling error. */
            size_t top = 0;
            size_t second = 1;
            size_t k;
            double observed;
            double expected;
            for (k = 0; k < distinct; k++) {
                if (counts[k] > counts[top]) {
                    top = k;
                }
            }
            second = top == 0 ? 1 : 0;
            for (k = 0; k < distinct; k++) {
                if (k != top && counts[k] > counts[second]) {
                    second = k;
                }
            }
            observed = log((double)counts[top] / (double)counts[second]);
            expected = cost[second] - cost[top];
            printf("distinct=%zu top=%s (%d, cost %.4f) second=%s (%d, cost %.4f) "
                   "log-ratio observed %.3f expected %.3f\n",
                   distinct, seen[top], counts[top], cost[top], seen[second], counts[second],
                   cost[second], observed, expected);
            if (counts[second] >= 50) {
                assert(fabs(observed - expected) < 0.25);
            }
        }
    }

    /* (5) Soft EM's E-step: the forward-backward link posteriors are the
     * marginals of the very distribution the sampler draws from. */
    {
        word s = make("a k t e");
        word t = make("a t e");
        rg_form fs = form("A", &s);
        rg_form ft = form("B", &t);
        rg_train_options sampled = options;
        rg_link_posterior *post = 0;
        size_t post_count = 0;
        double freq[MAX_SEG][MAX_SEG];
        double row_sum[MAX_SEG];
        int draws = 4000;
        int d;
        size_t q;
        memset(freq, 0, sizeof(freq));
        memset(row_sum, 0, sizeof(row_sum));
        assert(rg_align_link_posteriors(ctx, model, &options, &fs, &ft, 0, 1.0, &post, &post_count) == RG_OK);
        assert(post_count > 0);
        sampled.alignment_sampling_temperature = 1.0;
        for (d = 0; d < draws; d++) {
            rg_alignment *a = 0;
            size_t li;
            size_t si = 0;
            size_t ti = 0;
            sampled.alignment_sampling_seed = 10000 + d;
            assert(rg_align_forms_with_model(ctx, model, &sampled, &fs, &ft, 0, &a) == RG_OK);
            for (li = 0; li < rg_alignment_link_count(a); li++) {
                const rg_link *link = rg_alignment_link_at(a, li);
                if (link->source_count == 1 && link->target_count == 1) {
                    freq[si][ti] += 1.0 / (double)draws;
                }
                si += link->source_count;
                ti += link->target_count;
            }
            rg_alignment_free(a);
        }
        for (q = 0; q < post_count; q++) {
            const rg_link_posterior *p = &post[q];
            assert(p->source_index < s.n && p->target_index < t.n);
            assert(p->probability > 0.0 && p->probability <= 1.0);
            row_sum[p->source_index] += p->probability;
            printf("posterior %zu~%zu %.4f sampled %.4f\n", p->source_index, p->target_index,
                   p->probability, freq[p->source_index][p->target_index]);
            assert(fabs(p->probability - freq[p->source_index][p->target_index]) < 0.03);
            freq[p->source_index][p->target_index] = 0.0;
        }
        /* No sampled 1-to-1 link the posteriors missed. */
        for (q = 0; q < MAX_SEG * MAX_SEG; q++) {
            assert(freq[q / MAX_SEG][q % MAX_SEG] < 0.01);
        }
        for (q = 0; q < s.n; q++) {
            assert(row_sum[q] <= 1.0 + 1e-9);
        }
        rg_link_posteriors_free(post);

        /* A vanishing temperature puts all the mass on the cheapest path. */
        assert(rg_align_link_posteriors(ctx, model, &options, &fs, &ft, 0, 1e-6, &post, &post_count) == RG_OK);
        {
            size_t li;
            size_t si = 0;
            size_t ti = 0;
            size_t ones = 0;
            for (li = 0; li < rg_alignment_link_count(best); li++) {
                const rg_link *link = rg_alignment_link_at(best, li);
                if (link->source_count == 1 && link->target_count == 1) {
                    for (q = 0; q < post_count; q++) {
                        if (post[q].source_index == si && post[q].target_index == ti) {
                            assert(post[q].probability > 0.999);
                            ones++;
                        }
                    }
                }
                si += link->source_count;
                ti += link->target_count;
            }
            assert(ones > 0);
        }
        rg_link_posteriors_free(post);
        assert(rg_align_link_posteriors(ctx, model, &options, &fs, &ft, 0, 0.0, &post, &post_count) == RG_ERR_INVALID_ARGUMENT);
    }

    /* Soft EM trains, and a negative temperature is refused. */
    {
        rg_train_options soft = options;
        rg_pairwise_model *soft_model = 0;
        soft.alignment_posterior_temperature = 0.25;
        assert(rg_train_pairwise(ctx, pairs, corpus_count, &soft, &soft_model) == RG_OK);
        rg_pairwise_model_free(soft_model);
        soft.alignment_posterior_temperature = -1.0;
        assert(rg_train_pairwise(ctx, pairs, corpus_count, &soft, &soft_model) == RG_ERR_INVALID_ARGUMENT);
    }

    rg_alignment_free(best);
    rg_pairwise_model_free(model);
    rg_context_free(ctx);
    printf("alignment_sampling: ok\n");
    return 0;
}
