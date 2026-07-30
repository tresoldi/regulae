#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <string.h>

static rg_segment seg(const char *g) {
    rg_segment s = {g, 0, 0, 0};
    return s;
}

static double must_score(rg_context *ctx, const rg_segment *src, size_t src_n, const rg_segment *tgt, size_t tgt_n) {
    double cost = -1.0;
    assert(rg_score_link(ctx, src, src_n, tgt, tgt_n, &cost) == RG_OK);
    return cost;
}

int main(void) {
    rg_context *ctx = 0;
    rg_segment p = seg("p");
    rg_segment b = seg("b");
    rg_segment k = seg("k");
    rg_segment a = seg("a");
    rg_segment kt[] = {{"k", 0, 0, 0}, {"t", 0, 0, 0}};
    rg_segment tt[] = {{"t", 0, 0, 0}, {"t", 0, 0, 0}};
    rg_feature_displacement *disp = 0;
    size_t disp_count = 0;
    double identity;
    double close;
    double far;
    double symmetric;
    double asymmetric;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    identity = must_score(ctx, &p, 1, &p, 1);
    assert(fabs(identity) < 1e-12);
    close = must_score(ctx, &p, 1, &b, 1);
    far = must_score(ctx, &p, 1, &a, 1);
    assert(close > identity);
    assert(close < far);
    assert(fabs(close - must_score(ctx, &b, 1, &p, 1)) < 1e-12);
    assert(fabs(must_score(ctx, &p, 1, 0, 0) - 0.5) < 1e-12);
    assert(fabs(must_score(ctx, 0, 0, &p, 1) - 0.5) < 1e-12);
    assert(fabs(must_score(ctx, 0, 0, 0, 0)) < 1e-12);
    symmetric = must_score(ctx, kt, 2, tt, 2);
    asymmetric = must_score(ctx, kt, 2, tt, 1);
    assert(asymmetric > symmetric);

    assert(rg_compute_displacement(ctx, p, p, &disp, &disp_count) == RG_OK);
    assert(disp_count == 0);
    rg_feature_displacement_free(disp, disp_count);
    disp = 0;
    assert(rg_compute_displacement(ctx, p, b, &disp, &disp_count) == RG_OK);
    assert(disp_count > 0);
    rg_feature_displacement_free(disp, disp_count);
    disp = 0;
    k.grapheme = "QQZZ";
    assert(rg_compute_displacement(ctx, k, p, &disp, &disp_count) == RG_ERR_UNKNOWN_GRAPHEME);
    assert(rg_score_link(ctx, &k, 1, &p, 1, &identity) == RG_ERR_UNKNOWN_GRAPHEME);
    assert(rg_score_link(0, &p, 1, &p, 1, &identity) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_score_link(ctx, &p, 1, &p, 1, 0) == RG_ERR_INVALID_ARGUMENT);
    rg_context_free(ctx);
    return 0;
}
