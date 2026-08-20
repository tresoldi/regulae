#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void test_prior_only(void) {
    rg_context *ctx = 0;
    rg_translation_table *table = 0;
    double p_identity;
    double p_close;
    double p_far;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(rg_translation_table_from_prior(ctx, &table) == RG_OK);

    p_identity = rg_translation_probability(table, "p", "p", RG_DIR_SYMMETRIC);
    p_close = rg_translation_probability(table, "p", "b", RG_DIR_SYMMETRIC);
    p_far = rg_translation_probability(table, "p", "a", RG_DIR_SYMMETRIC);
    assert(p_identity > p_close);
    assert(p_close > p_far);
    assert(p_far > 0.0);

    assert(rg_translation_probability(table, "QQZZ", "p", RG_DIR_SYMMETRIC) == 0.0);

    rg_translation_table_free(table);
    rg_context_free(ctx);
    printf("  prior_only: ok\n");
}

static void test_precompute(void) {
    rg_context *ctx = 0;
    rg_translation_table *table = 0;
    const char *source_inv[] = {"p", "b", "t", "d", "k", "a", "i", "u"};
    const char *target_inv[] = {"p", "b", "t", "d", "k", "a", "i", "u"};
    size_t src_count = 0;
    size_t tgt_count = 0;
    const char *const *src_vocab;
    const char *const *tgt_vocab;
    double lazy_p;
    double precomp_p;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(rg_translation_table_from_prior(ctx, &table) == RG_OK);

    lazy_p = rg_translation_probability(table, "p", "b", RG_DIR_SYMMETRIC);

    assert(rg_translation_table_precompute(table, source_inv, 8, target_inv, 8) == RG_OK);

    precomp_p = rg_translation_probability(table, "p", "b", RG_DIR_SYMMETRIC);
    assert(precomp_p > 0.0);

    src_vocab = rg_translation_table_source_vocab(table, &src_count);
    tgt_vocab = rg_translation_table_target_vocab(table, &tgt_count);
    assert(src_count == 8);
    assert(tgt_count == 8);
    assert(src_vocab != 0);
    assert(tgt_vocab != 0);

    (void)lazy_p;

    rg_translation_table_free(table);
    rg_context_free(ctx);
    printf("  precompute: ok\n");
}

static void test_train_simple(void) {
    rg_context *ctx = 0;
    rg_translation_table *table = 0;

    rg_segment src1[] = {{"p", 0, 0, 0}, {"a", 0, 0, 0}};
    rg_segment tgt1[] = {{"b", 0, 0, 0}, {"a", 0, 0, 0}};
    rg_segment src2[] = {{"p", 0, 0, 0}, {"i", 0, 0, 0}};
    rg_segment tgt2[] = {{"b", 0, 0, 0}, {"i", 0, 0, 0}};
    rg_segment src3[] = {{"t", 0, 0, 0}, {"a", 0, 0, 0}};
    rg_segment tgt3[] = {{"d", 0, 0, 0}, {"a", 0, 0, 0}};

    rg_form_pair pairs[3];
    double p_pb;
    double p_pa;
    double p_td;

    memset(pairs, 0, sizeof(pairs));
    pairs[0].source.segments = src1;
    pairs[0].source.segment_count = 2;
    pairs[0].target.segments = tgt1;
    pairs[0].target.segment_count = 2;
    pairs[1].source.segments = src2;
    pairs[1].source.segment_count = 2;
    pairs[1].target.segments = tgt2;
    pairs[1].target.segment_count = 2;
    pairs[2].source.segments = src3;
    pairs[2].source.segment_count = 2;
    pairs[2].target.segments = tgt3;
    pairs[2].target.segment_count = 2;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(rg_train_translation_table(ctx, pairs, 3, 0, &table) == RG_OK);

    p_pb = rg_translation_probability(table, "p", "b", RG_DIR_FORWARD);
    p_pa = rg_translation_probability(table, "p", "a", RG_DIR_FORWARD);
    p_td = rg_translation_probability(table, "t", "d", RG_DIR_FORWARD);

    assert(p_pb > 0.0);
    assert(p_pb > p_pa);
    assert(p_td > 0.0);

    rg_translation_table_free(table);
    rg_context_free(ctx);
    printf("  train_simple: ok\n");
}

static void test_align(void) {
    rg_context *ctx = 0;
    rg_translation_table *table = 0;
    rg_translation_alignment *alignment = 0;

    rg_segment src1[] = {{"p", 0, 0, 0}, {"a", 0, 0, 0}};
    rg_segment tgt1[] = {{"b", 0, 0, 0}, {"a", 0, 0, 0}};
    rg_segment src2[] = {{"p", 0, 0, 0}, {"i", 0, 0, 0}};
    rg_segment tgt2[] = {{"b", 0, 0, 0}, {"i", 0, 0, 0}};

    rg_form_pair pairs[2];
    double score = 0.0;

    memset(pairs, 0, sizeof(pairs));
    pairs[0].source.segments = src1;
    pairs[0].source.segment_count = 2;
    pairs[0].target.segments = tgt1;
    pairs[0].target.segment_count = 2;
    pairs[1].source.segments = src2;
    pairs[1].source.segment_count = 2;
    pairs[1].target.segments = tgt2;
    pairs[1].target.segment_count = 2;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(rg_train_translation_table(ctx, pairs, 2, 0, &table) == RG_OK);

    assert(rg_translation_align(table, src1, 2, tgt1, 2, RG_DIR_SYMMETRIC, &alignment) == RG_OK);
    assert(alignment != 0);
    assert(alignment->count == 2);
    assert(alignment->assignments[0].target_index == 0);
    assert(alignment->assignments[1].target_index == 1);
    rg_translation_alignment_free(alignment);

    assert(rg_translation_align(table, src1, 2, tgt1, 2, RG_DIR_FORWARD, &alignment) == RG_OK);
    assert(alignment->count == 2);
    rg_translation_alignment_free(alignment);

    assert(rg_translation_score(table, src1, 2, tgt1, 2, RG_DIR_SYMMETRIC, &score) == RG_OK);
    assert(score < 0.0);

    rg_translation_table_free(table);
    rg_context_free(ctx);
    printf("  align: ok\n");
}

static void test_symmetry(void) {
    rg_context *ctx = 0;
    rg_translation_table *table = 0;

    rg_segment src1[] = {{"p", 0, 0, 0}, {"a", 0, 0, 0}};
    rg_segment tgt1[] = {{"b", 0, 0, 0}, {"a", 0, 0, 0}};

    rg_form_pair pairs[1];
    double fwd;
    double bwd;
    double sym;

    memset(pairs, 0, sizeof(pairs));
    pairs[0].source.segments = src1;
    pairs[0].source.segment_count = 2;
    pairs[0].target.segments = tgt1;
    pairs[0].target.segment_count = 2;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(rg_train_translation_table(ctx, pairs, 1, 0, &table) == RG_OK);

    fwd = rg_translation_probability(table, "p", "b", RG_DIR_FORWARD);
    bwd = rg_translation_probability(table, "p", "b", RG_DIR_BACKWARD);
    sym = rg_translation_probability(table, "p", "b", RG_DIR_SYMMETRIC);

    assert(fwd > 0.0);
    assert(bwd > 0.0);
    assert(sym > 0.0);
    assert(fabs(sym - sqrt(fwd * bwd)) < 1e-12);

    rg_translation_table_free(table);
    rg_context_free(ctx);
    printf("  symmetry: ok\n");
}

static void test_null_args(void) {
    assert(rg_train_translation_table(0, 0, 0, 0, 0) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_translation_table_from_prior(0, 0) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_translation_probability(0, "p", "b", RG_DIR_FORWARD) == 0.0);
    assert(rg_translation_align(0, 0, 0, 0, 0, RG_DIR_SYMMETRIC, 0) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_translation_score(0, 0, 0, 0, 0, RG_DIR_SYMMETRIC, 0) == RG_ERR_INVALID_ARGUMENT);

    rg_translation_table_free(0);
    rg_translation_alignment_free(0);

    printf("  null_args: ok\n");
}

static void test_vocab_accessors(void) {
    rg_context *ctx = 0;
    rg_translation_table *table = 0;
    size_t count = 99;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(rg_translation_table_from_prior(ctx, &table) == RG_OK);

    assert(rg_translation_table_source_vocab(table, &count) == 0);
    assert(count == 0);
    assert(rg_translation_table_target_vocab(table, &count) == 0);
    assert(count == 0);

    rg_translation_table_free(table);
    rg_context_free(ctx);
    printf("  vocab_accessors: ok\n");
}

static void test_empty_corpus(void) {
    rg_context *ctx = 0;
    rg_translation_table *table = 0;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(rg_train_translation_table(ctx, 0, 0, 0, &table) == RG_OK);
    assert(table != 0);

    assert(rg_translation_probability(table, "p", "b", RG_DIR_FORWARD) == 0.0);

    rg_translation_table_free(table);
    rg_context_free(ctx);
    printf("  empty_corpus: ok\n");
}

int main(void) {
    printf("test_ibm1:\n");
    test_null_args();
    test_prior_only();
    test_precompute();
    test_train_simple();
    test_align();
    test_symmetry();
    test_vocab_accessors();
    test_empty_corpus();
    printf("all passed\n");
    return 0;
}
