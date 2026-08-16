#!/usr/bin/env python3
"""Runs regulae over the field's shared reference wordlists, one family at a time.

`testdata/soundlaws/` asks whether regulae can find changes nobody disputes, on
corpora built so that exactly one answer is right. This asks the other question:
what happens on the data historical linguists actually have. Every dataset here
is a published CLDF wordlist with expert cognate judgements over segmented
forms, and none of them was built to be easy.

The point is not a score. There is no gold standard for "which conditioned
environments are real in Dravidian", and a number claiming there is would be
inventing one. What the survey reports is the shape of the answer -- how many
correspondences, how many of them conditioned, how far the corpus sits from its
own shuffled baseline, and how many of the conditioned rules survive being
compared against it. Those are the numbers a reader has to look at anyway, and
collecting them across fifty families says things no single family can:

  * that the corpus-level verdict and the rule-level verdict disagree, often.
    A wordlist can sit forty standard deviations below its shuffle -- so its
    cognates are certainly cognates -- and still have almost none of its
    conditioned rules clear the level the same search reaches on noise.
  * that the class counts scale with the number of lects and the number of
    sets and not with how much has been discovered, which is why they are not
    reported as a result anywhere in this repository.
  * which families cost what, so a slow corpus is a known quantity rather than
    a surprise.

Nothing is vendored. Every dataset carries its own licence and citation, so
this reads a local clone -- see `scripts/lexibank.py` -- and the numbers depend
on which revision of it you have. A recorded run lives in
`docs/family_survey.md` with the date it was taken.

Usage:
    scripts/families.py --clone ~/lexibank_clone            # curated list
    scripts/families.py --all                               # every usable one
    scripts/families.py --permutations 10                   # add the baseline
    scripts/families.py --lects 8 -o docs/family_survey.md
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lexibank  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent

# Curated, and the third column is the reason. A survey that listed every
# dataset in alphabetical order would be a list of what happens to be
# available; this is a claim about coverage, and each row has to defend itself.
#
# Read down the last column and the survey is a set of questions: does the
# method behave the same way on a family with tone as on one without, on a
# dialect continuum as on a five-thousand-year-old family, on a wordlist of
# two hundred concepts as on one of five thousand.
FAMILIES = [
    # --- Indo-European and Europe -------------------------------------
    ("starostinpie", "Indo-European", "the family the comparative method was built on"),
    ("meloniromance", "Romance", "shallow, dense, and with the ancestor attested"),
    ("saenkoromance", "Romance", "a second Romance coding, to see what the coding decides"),
    ("iecor", "Indo-European", "the phylogenetics reference wordlist"),
    ("syrjaenenuralic", "Uralic", "vowel harmony and consonant gradation"),
    ("zhivlovobugrian", "Ob-Ugric", "a small, deep, well-studied branch"),
    # --- Central and North Asia ---------------------------------------
    ("savelyevturkic", "Turkic", "rounding harmony -- the change that exposed the old feature list"),
    ("hruschkaturkic", "Turkic", "a second Turkic sample with different judgements"),
    ("oskolskayatungusic", "Tungusic", "harmony, and a member of a contested macro-family"),
    ("robbeetstriangulation", "Transeurasian", "a proposed grouping the field has not settled"),
    ("leekoreanic", "Koreanic", "great internal depth in a very small family"),
    ("hattorijaponic", "Japonic", "pitch accent as a dimension beside the segments"),
    ("leejaponic", "Japonic", "dialect-level divergence, densely sampled"),
    ("leeainu", "Ainu", "internal comparison within an isolate"),
    # --- Sinitic and mainland Southeast Asia --------------------------
    ("liusinitic", "Sinitic", "tone; the home ground of cross-dimensional discovery"),
    ("houchinese", "Sinitic", "a second Sinitic sample"),
    ("hsiuhmongmien", "Hmong-Mien", "tone and register"),
    ("starostinhmongmien", "Hmong-Mien", "the same family, independently coded"),
    ("mannburmish", "Burmish", "tone in Tibeto-Burman"),
    ("yanglalo", "Loloish", "the densest tonal sample available"),
    ("luangthongkumkaren", "Karenic", "tone, and heavy contact with Tai"),
    ("sagartst", "Sino-Tibetan", "deep and contested"),
    ("zhangrgyalrong", "Rgyalrongic", "elaborate morphology on short roots"),
    ("bodtkhobwa", "Kho-Bwa", "a small family sampled very deeply"),
    ("sidwellvietic", "Vietic", "tonogenesis caught in progress -- the textbook case"),
    ("sidwellbahnaric", "Bahnaric", "register, the stage before tone"),
    ("deepadungpalaung", "Palaungic", "register again, in a different branch"),
    ("nagarajakhasian", "Khasian", "Austroasiatic without tone"),
    ("dunnaslian", "Aslian", "Austroasiatic under heavy Malay contact"),
    # --- South Asia, Africa, the Near East ----------------------------
    ("dravlex", "Dravidian", "retroflexion as a system-wide contrast"),
    ("kitchensemitic", "Semitic", "root-and-pattern morphology over three consonants"),
    ("felekesemitic", "Ethiosemitic", "a second Semitic sample"),
    ("ratcliffearabic", "Arabic", "a dialect continuum rather than a tree"),
    ("gravinachadic", "Chadic", "tone in Afro-Asiatic"),
    ("grollemundbantu", "Bantu", "noun-class prefixes and spirantisation"),
    # --- Austronesia and the Pacific ----------------------------------
    ("walworthpolynesian", "Polynesian", "exceptionless mergers, tiny inventories"),
    ("blustaustronesian", "Austronesian", "the deep end of the family"),
    ("abvdphilippines", "Austronesian", "shallow and very widely sampled"),
    ("smithborneo", "Austronesian", "dense sampling within one island"),
    ("robinsonap", "Alor-Pantar", "Papuan, no established outside relatives"),
    ("mcelhanonhuon", "Huon", "Papuan, a second sample"),
    ("bowernpny", "Pama-Nyungan", "Australia, with a long contact history"),
    # --- The Americas -------------------------------------------------
    ("utoaztecan", "Uto-Aztecan", "a long north-south chain"),
    ("wichmannmixezoquean", "Mixe-Zoquean", "small, well-reconstructed"),
    ("mattercariban", "Cariban", "Amazonia"),
    ("tuled", "Tupian", "Amazonia, large and carefully coded"),
    ("constenlachibchan", "Chibchan", "Central America"),
    ("chaconarawakan", "Arawakan", "Amazonia, a wide-ranging family"),
    ("oliveiraprotopanoan", "Panoan", "reconstructed proto-forms in the wordlist"),
    ("crossandean", "Andean", "Quechuan and Aymaran, related or not"),
    # --- Built for a purpose rather than for a family ------------------
    ("kesslersignificance", "control", "assembled to test whether resemblances are chance"),
    ("bdpa", "many", "the benchmark database of phonetic alignments"),
]


# Chao tone letters, as a dataset writes them into a `Segments` cell. Counting
# them is how the survey knows which wordlists could have yielded a
# cross-dimensional rule -- "five of forty-three families" is a much weaker
# claim than "five of the six that transcribe tone at all", and most of this
# list does not transcribe tone.
CHAO = "\u00b9\u00b2\u00b3\u2070\u2074\u2075\u2076\u2077\u2078\u2079"


def convert(clone, name, max_lects):
    rows, stats = lexibank.convert(clone, name, max_lects=max_lects)
    kept = []
    for cognate_id, lect, segments in rows:
        cleaned = lexibank.clean_segments(segments)
        if cleaned is None:
            stats["markup"] += 1
            continue
        if any(c in CHAO for c in cleaned):
            stats["toned_forms"] += 1
        kept.append((cognate_id, lect, cleaned))
    return kept, stats


def train(cli, path, permutations, limit):
    argv = [cli, "train", str(path), "--json"]
    if permutations:
        argv += ["--permutations", str(permutations)]
    started = time.time()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=limit)
    except subprocess.TimeoutExpired:
        # Reported rather than raised. Which corpora are impractical at a given
        # lect count is one of the things the survey is for -- `meloniromance`
        # carries five thousand cognate sets and is a different kind of run
        # from a two-hundred-set wordlist, and saying so is more useful than
        # quietly leaving it out or waiting an hour for it.
        return None, f"over {limit}s", time.time() - started
    elapsed = time.time() - started
    if proc.returncode != 0:
        # Kept apart from the timeout, and the reason kept. This collapsed both
        # into "too slow" for one afternoon and reported `iecor` as a corpus
        # that takes over two hundred seconds; it takes one, and then refuses a
        # dental-diacritic rhotic the feature system does not cover. The two
        # need different work from whoever reads the survey, and the second
        # names the grapheme that has to be dealt with.
        # The refusal is not necessarily the first thing on stderr. A corpus
        # with doublets in it announces them before training starts, and taking
        # the first line reported Tungusic as failing because it has doublets,
        # which it does and which is not a failure. The refusal names its
        # grapheme wherever it lands, so look for that first.
        detail = proc.stderr.strip().splitlines()
        match = re.search(r'unknown grapheme "([^"]*)"', proc.stderr)
        if match:
            return None, f"grapheme `{match.group(1)}` not covered", elapsed
        markup = re.search(r'"([^"]*)" is CLDF/CLTS markup', proc.stderr)
        if markup:
            return None, f"`{markup.group(1)}` is markup, not a sound", elapsed
        return None, (detail[-1] if detail else "training failed"), elapsed
    payload = json.loads(proc.stdout)
    fit = dict(payload.get("fit", payload))
    # Not in rg_corpus_fit, and worth a column of its own on this list: half
    # the families here were chosen because they have tone, and how many
    # cross-dimensional rules a tonal corpus yields is the question they were
    # chosen to ask.
    fit["cross_dimensional"] = len(payload.get("cross_dimensional") or [])
    return fit, None, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone", default="~/lexibank_clone")
    parser.add_argument("--cli", default=str(REPO / "build" / "c" / "regulae"))
    parser.add_argument("--lects", type=int, default=6,
                        help="lects per dataset, most-attested first (default 6)")
    parser.add_argument("--permutations", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=240,
                        help="seconds per dataset before it is recorded as too slow")
    parser.add_argument("--all", action="store_true",
                        help="every dataset with judgements and segments, not the curated list")
    parser.add_argument("--only", help="comma-separated dataset names")
    parser.add_argument("--out", "-o")
    args = parser.parse_args()

    if args.only:
        wanted = [(name, "", "") for name in args.only.split(",")]
    elif args.all:
        wanted = [(name, "", "") for name in lexibank.usable(args.clone)]
    else:
        wanted = FAMILIES

    lines = []
    skipped = []
    work = tempfile.mkdtemp(prefix="regulae-families-")
    for name, family, why in wanted:
        try:
            rows, stats = convert(args.clone, name, args.lects)
        except Exception as error:                     # a dataset can be absent
            skipped.append((name, family, str(error)))
            continue
        if not rows:
            # Cognate judgements over forms with no `Segments` column. The
            # dataset is not unusable in general -- it is unusable by anything
            # that reads segments, which is every method in this repository.
            skipped.append((name, family, "no segmented forms"))
            continue
        path = pathlib.Path(work) / f"{name}.tsv"
        with path.open("w", encoding="utf-8") as handle:
            handle.write("cognate_id\tlect_id\tsegments\n")
            for row in rows:
                handle.write("\t".join(row) + "\n")
        fit, failure, elapsed = train(args.cli, path, args.permutations, args.timeout)
        if fit is None:
            reason = failure
            if reason.startswith("over "):
                reason += f" at {args.lects} lects ({stats['sets']} cognate sets)"
            reason = reason.replace("regulae: ", "")
            skipped.append((name, family, reason))
            print(f"{name:<26} {reason[:70]}", file=sys.stderr)
            continue
        lines.append({
            "name": name, "family": family, "why": why,
            "sets": fit.get("scored_set_count", 0),
            "lects": stats["lects_used"],
            "available": stats["lects_available"],
            "dropped": stats.get("markup", 0),
            "toned": stats.get("toned_forms", 0),
            "uncond": fit.get("unconditioned_class_count", 0),
            "cond": fit.get("conditioned_class_count", 0),
            "xdim": fit.get("cross_dimensional", 0),
            # Forms with no vowel and no syllabic consonant, given one so the
            # syllable predicates have something to hold of. A corpus with many
            # of these has syllable-conditioned rules resting on a guess, and
            # on a real wordlist the count is rarely zero.
            "guessed": fit.get("inferred_nucleus_form_count", 0),
            "syllabified": fit.get("syllabified_form_count", 0),
            "cost": fit.get("cost_per_segment", 0.0),
            "z": fit.get("cost_per_segment_z", 0.0),
            "above": fit.get("rules_above_noise", 0),
            "measured": fit.get("rules_measured", 0),
            "pair_above": fit.get("pairwise_rules_above_noise", 0),
            "pair_measured": fit.get("pairwise_rules_measured", 0),
            "seconds": elapsed,
        })
        print(f"{name:<26} {lines[-1]['sets']:>5} sets  "
              f"{lines[-1]['uncond']:>5} cls  {lines[-1]['cond']:>4} cond  "
              f"cost={lines[-1]['cost']:>7.3f}  {elapsed:>6.1f}s", file=sys.stderr)

    report = render(lines, skipped, args)
    if args.out:
        pathlib.Path(args.out).write_text(report)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    return 0


def render(lines, skipped, args):
    """The report, with the readings that hold across runs stated as readings
    and the ones that do not computed from the run itself.

    A table of numbers with no prose is a table nobody reads twice. What the
    survey is for is a handful of statements about the method that only show up
    across families, and each of them has to be recoverable from the columns
    beside it -- so the summary counts what it claims rather than asserting it.
    """
    out = ["# Family survey", "",
           "<!-- Recorded by scripts/families.py. Not checked by CI: the numbers",
           "     depend on which revision of the Lexibank clone produced them. -->",
           ""]
    out.append(f"{len(lines)} published wordlists with expert cognate judgements over "
               f"segmented forms, {args.lects} lects each (the most widely attested), "
               f"trained with default options"
               + (f" and {args.permutations} shuffles." if args.permutations else "."))
    out.append("")
    out.append("There is no score here and there could not be. Nobody has a gold "
               "standard for *which conditioned environments are real in Dravidian*, "
               "and a number claiming there is would be inventing one. What the "
               "survey reports is the shape of the answer, across enough families "
               "that the shape itself says something.")
    out.append("")

    transcribes_tone = [r for r in lines if r["toned"] > 0]
    tonal = [r for r in transcribes_tone if r["xdim"] > 0]
    silent = [r for r in transcribes_tone if r["xdim"] == 0]
    guessed = [r for r in lines if r["guessed"] > 0]
    out.append("## What to read off it")
    out.append("")
    out.append("**Class counts scale with the corpus, not with the discovery.** They "
               "rise with the number of lects and the number of sets, and they rise "
               "on shuffled data too. Nothing in this repository reports them as a "
               "result, and the column is here so that the point is visible rather "
               "than asserted.")
    out.append("")
    out.append(f"**Tone.** {len(transcribes_tone)} of the {len(lines)} wordlists "
               f"transcribe tone at all -- most of the world's tone languages are "
               f"represented here by sources that do not write it -- and "
               f"{len(tonal)} of those {len(transcribes_tone)} yielded "
               f"cross-dimensional rules. Until 2026-08-16 the number was zero, on "
               f"every one of them: the CLDF `Segments` column writes Chao tone as a "
               f"token of its own (`t\u02b0 u \u2075\u00b9`) and the pre-segmented "
               f"loader kept it as a segment, so no segment carried a tone and the "
               f"stage that exists for tonogenesis had nothing to read.")
    if silent:
        out.append("")
        out.append("Still silent, with tone in the transcription and no "
                   "cross-dimensional rule out of it: "
                   + ", ".join(f"`{r['name']}`" for r in silent)
                   + ". Worth a look rather than an explanation -- zero is a "
                     "legitimate answer, and it is also what the directionality "
                     "limitation recorded in `experiments/README.md` produces, "
                     "since the stage searches one side of a lect pair and the "
                     "side is decided by which lect id sorts first.")
    out.append("")
    if guessed:
        out.append(f"**{len(guessed)} wordlists contain forms with no nucleus** -- no "
                   f"vowel and no syllabic consonant -- which are given one so the "
                   f"syllable predicates have something to hold of. A "
                   f"syllable-conditioned rule on those corpora rests on that guess. "
                   f"The column is `guessed/syllabified`.")
        out.append("")
    if args.permutations:
        pairs = [r for r in lines if r["measured"] > 0]
        thin = [r for r in pairs if r["above"] * 4 < r["measured"]]
        out.append(f"**The corpus-level verdict and the rule-level verdict disagree, "
                   f"often.** {len(thin)} of the {len(pairs)} wordlists that had rules "
                   f"measured had fewer than a quarter of them stand above the level "
                   f"the same search reaches on the corpus shuffled -- while sitting "
                   f"tens of standard deviations below their own baseline on "
                   f"`cost/seg`. The cognates are cognates; most of the environments "
                   f"are not distinguishable from artefacts of having looked.")
        out.append("")

    header = ("| dataset | family | sets | classes | conditioned | cross-dim |"
              " nucleus guessed | cost/seg |")
    rule = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    if args.permutations:
        header += " z | above noise | per pair |"
        rule += " ---: | ---: | ---: |"
    out += ["## The run", "", header, rule]
    for row in lines:
        guessed_cell = (f"{row['guessed']}/{row['syllabified']}"
                        if row["syllabified"] else str(row["guessed"]))
        cells = (f"| `{row['name']}` | {row['family']} | {row['sets']} | "
                 f"{row['uncond']} | {row['cond']} | {row['xdim']} | "
                 f"{guessed_cell} | {row['cost']:.3f} |")
        if args.permutations:
            cells += (f" {row['z']:.1f} | {row['above']}/{row['measured']} |"
                      f" {row['pair_above']}/{row['pair_measured']} |")
        out.append(cells)
    if skipped:
        out += ["", "## Not surveyed", "",
                "Each row is a thing to fix rather than a dataset to drop. A "
                "grapheme the feature system does not cover is one line in "
                "merkmal, and `docs/grapheme_coverage.md` has the standing "
                "measurement of how many tokens each one costs; a corpus that "
                "runs long is a corpus to sample; a dataset with cognate "
                "judgements but no segmented forms is invisible to every "
                "method in this repository and worth saying so about.",
                "", "| dataset | why |", "| --- | --- |"]
        for name, _, why in skipped:
            out.append(f"| `{name}` | {why} |")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
