/* Fails if the suite was built with assertions disabled.
 *
 * Every other test in tests/c states its expectations with assert(), and most
 * of them put the library call inside the assert:
 *
 *     assert(rg_compute_syllable_breaks(ctx, &form, &breaks, &count) == RG_OK);
 *
 * With NDEBUG defined that statement is removed entirely, taking the call with
 * it. The suite then passes while exercising nothing — a worse outcome than any
 * red build, because it looks like evidence. This test makes that condition
 * loud instead of invisible.
 *
 * It checks the property the suite actually depends on — that assert evaluates
 * its argument — rather than only that NDEBUG is undefined. */

#include <assert.h>
#include <stdio.h>

int main(void)
{
#ifdef NDEBUG
    fprintf(stderr,
            "tests were compiled with NDEBUG: assert() is inert, so the suite\n"
            "would pass without calling into the library at all.\n"
            "See the -UNDEBUG option applied to the test targets.\n");
    return 1;
#else
    int evaluated = 0;

    /* The side effect is the test. This file exists to catch a build where
     * NDEBUG has deleted the assertions, and the only way to see that is an
     * assert that does something observable. */
    /* NOLINTNEXTLINE(bugprone-assert-side-effect) */
    assert((evaluated = 1) == 1);

    if (!evaluated) {
        fprintf(stderr, "assert() did not evaluate its argument\n");
        return 1;
    }

    printf("assertions are live\n");
    return 0;
#endif
}
