#!/usr/bin/env python3
"""Generates web/corpora.js: every corpus the page can load, keyed by path.

The example list and the guide both need corpus text, and embedding it twice
would ship several corpora in two places that could drift. This is the single
source; both reference entries by path.

Corpora are grouped for the example list. "Start here" is a ladder: each entry
shows something the previous one cannot. "Not readable in this format yet"
carries the grapheme that blocks it, so the list says why rather than just
failing when picked.

No count appears in a hand-written description. Every number the list shows is
read off the model the corpus trains to and written into `stats`, because the
three that were written by hand had all gone stale: the ladder offered "97
cognates, 22 conditioned classes" for Latin and Spanish where the build
published 10, "79 accidental correspondences" where it published 95, and "the
four environments" for final devoicing where it published 6. Regenerating this
file reproduced them exactly, since the prose is the input rather than the
output. Prose says what a corpus is for; `stats` says what it does.

Usage:
    scripts/corpora.py            # rewrite web/corpora.js
    scripts/corpora.py --check    # exit non-zero if it would change
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = REPO / "web" / "corpora.js"

# The ladder, in order. Each earns its place by showing a capability the
# previous entries cannot.
LADDER = [
    ("experiments/finnish_estonian/cognates.tsv",
     "Finnish and Estonian",
     "Two closely related lects. A first look at what a correspondence class is."),
    ("experiments/latin_spanish/cognates.tsv",
     "Latin and Spanish",
     "The environments are discovered, not declared: no rule list is supplied "
     "anywhere, and none of the conditioning below was named in advance."),
    ("testdata/soundlaws/grimm.tsv",
     "Grimm's Law",
     "Three shifts running at once and not one of them conditioned. Finding a "
     "single correspondence proves little; the claim is the chain shift whole, "
     "and every rung of it flat."),
    ("testdata/soundlaws/graded_1_adjacent.tsv",
     "One change, one adjacent trigger",
     "The same shape with conditioning that is really there, so the environment "
     "has to be recovered rather than declined."),
    ("experiments/contaminated_cognates_synthetic/cognates.tsv",
     "Contaminated cognates",
     "Confidence weighting and outlier ranking on deliberately noisy data."),
    ("testdata/corpora/real_romance_4lect.tsv",
     "Romance, four lects",
     "Latin, Spanish, French and Italian reconciled into classes binding all four."),
    # The last three are what the tool looks like when it is right to find
    # nothing, and they belong on the ladder for the same reason the contrast
    # environments belong in a fixture. A reader who has only seen it succeed
    # cannot tell success from output.
    ("testdata/restraint/chance.tsv",
     "Two unrelated lects",
     "Accidental correspondences and no selected environment, from wordlists "
     "with no history between them. The pairings match their shuffled baseline."),
    ("testdata/soundlaws/final_devoicing.tsv",
     "German final devoicing",
     "A neutralisation: /t/ and /d/ merge word-finally, so every environment "
     "reported for it is a correlate, and each falls below the baseline."),
    ("testdata/diagnostics/drift.tsv",
     "One language, two transcriptions",
     "The same forty words segmented two ways. Reads as deaffrication, loss of "
     "aspiration and loss of length -- none of which happened, and nothing catches it."),
]

DESCRIPTIONS = {
    "arabic_hebrew": "Semitic, with conditioned classes.",
    "georgian_svan": "Kartvelian. The richest conditioning in the set.",
    "latin_french": "Romance, a second daughter for comparison.",
    "latin_italian": "Romance, a third daughter.",
    "oe_english": "Old English against the modern language.",
    "swahili_zulu": "Bantu.",
    "turkish_azerbaijani": "Turkic, closely related.",
    "ppn_hawaiian": "Proto-Polynesian and Hawaiian.",
    "mandarin_historical": "Middle Chinese and Mandarin, tone held in separate columns.",
    "harmony_synthetic": "Vowel harmony, synthetic.",
    "umlaut_synthetic": "Umlaut, synthetic.",
    "length_conditioned_synthetic": "Length conditioning, synthetic.",
}

# One per change the field settled long ago, built so that exactly one answer
# is right and the wrong answers are available. They are globbed rather than
# listed, so a fixture added to testdata/soundlaws/ reaches the page without a
# second edit here; a name missing from this map gets no description rather
# than a wrong one.
SOUND_LAWS = {
    "compensatory_lengthening": "A lost segment paid for by the vowel before it.",
    "conditioned_confound": "Two environments the corpus cannot tell apart, and says so.",
    "final_devoicing": "German. A neutralisation, so every environment is a correlate.",
    "grassmann": "Of two aspirates in a word the first loses aspiration. Needs an "
                 "existential environment: the trigger is neither adjacent nor at a fixed distance.",
    "great_vowel_shift": "English. A chain shift with no conditioning anywhere in it.",
    "grimm": "PIE to Proto-Germanic. Three unconditioned shifts at once.",
    "lenition": "Western Romance. Voiceless stops voice between vowels and nowhere else.",
    "metathesis_adjacent": "Two segments swap. One event, not two substitutions.",
    "metathesis_distant": "The same, over a longer span.",
    "morphological_rhotacism": "Rhotacism where the boundary is morphological, "
                               "supplied by the corpus and never inferred.",
    "opaque_umlaut": "The trigger is gone from the surface by the time the change is visible.",
    "place_assimilation": "A nasal takes the place of what follows it.",
    "place_dissimilation": "Two like places, and one of them moves.",
    "rhotacism": "Latin. /s/ to /r/ between vowels, with the three environments "
                 "that do not rhotacise present so the wrong answer is available.",
    "rhotacism_reordered": "The same corpus with each set's rows swapped. Row order "
                           "is not data and must not move the model.",
    "rounding_harmony": "Rounding spreads across a word.",
    "verner": "The residue Grimm's Law leaves over, conditioned by where the accent was.",
}


def sound_law_description(stem):
    if stem in SOUND_LAWS:
        return SOUND_LAWS[stem]
    if stem.startswith("graded_"):
        return ("A rung on the conditioning ladder: the same corpus shape with the "
                "environment made harder to state.")
    return ""


def corpus_format(path):
    if path.suffix == ".csv":
        return "arcaverborum"
    if "experiments/" in path.as_posix():
        return "wide"
    return "tsv"


def probe(cli, path):
    """Trains a corpus to find out whether the page can load it, what it
    publishes, and if it cannot be read, which grapheme stops it.

    `--json` rather than the summary because the counts the list shows have to
    come off the model rather than off a person's memory of it. It costs one
    training run either way.
    """
    result = subprocess.run(
        [str(cli), "train", "--json", "--format", corpus_format(path), str(REPO / path)],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode == 0:
        fit = json.loads(result.stdout)["fit"]
        return {"readable": True, "stats": stats_line(fit)}
    # Both refusals the CLI can report name the offending token: a sound the
    # feature system does not cover, and CLDF/CLTS markup that never
    # transcribed one.
    match = (re.search(r'unknown grapheme "([^"]*)"', result.stderr) or
             re.search(r'"([^"]*)" is CLDF/CLTS markup', result.stderr))
    return {"readable": False, "blockedBy": match.group(1) if match else None}


def stats_line(fit):
    """What the corpus actually publishes, in the order a reader needs it:
    how much went in, how much came out, and whether the fit is any good."""
    conditioned = int(round(fit["conditioned_class_count"]))
    parts = [
        count_of(fit["lect_count"], "lect", "lects"),
        count_of(fit["scored_set_count"], "cognate set", "cognate sets"),
        count_of(fit["unconditioned_class_count"], "class", "classes"),
        "none conditioned" if conditioned == 0 else f"{conditioned} conditioned",
        f"cost/segment {fit['cost_per_segment']:.2f}".replace("-", "−"),
    ]
    return " · ".join(parts)


def count_of(value, singular, plural):
    count = int(round(value))
    return f"{count} {singular if count == 1 else plural}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cli", default=str(REPO / "build" / "c" / "regulae"))
    args = parser.parse_args()

    cli = pathlib.Path(args.cli)
    if not cli.exists():
        print(f"corpora: CLI not found at {cli}; build it first", file=sys.stderr)
        return 2

    paths = [pathlib.Path(entry[0]) for entry in LADDER]
    for pattern in ("experiments/*/cognates.tsv", "testdata/soundlaws/*.tsv"):
        for path in sorted(REPO.glob(pattern)):
            relative = path.relative_to(REPO)
            if relative not in paths:
                paths.append(relative)

    ladder_paths = {entry[0]: (entry[1], entry[2]) for entry in LADDER}
    corpora = {}
    entries = []
    for path in paths:
        key = path.as_posix()
        probed = probe(cli, path)
        corpora[key] = (REPO / path).read_text(encoding="utf-8")

        if key in ladder_paths:
            label, description = ladder_paths[key]
            group = "Start here"
        elif key.startswith("testdata/soundlaws/"):
            stem = path.stem
            label = stem.replace("_", " ")
            description = sound_law_description(stem)
            group = "Sound laws" if probed["readable"] else "Not readable in this format yet"
        else:
            name = path.parent.name
            label = name.replace("_", " ")
            description = DESCRIPTIONS.get(name, "")
            group = "More corpora" if probed["readable"] else "Not readable in this format yet"

        entry = {
            "path": key,
            "label": label,
            "description": description,
            "group": group,
            "format": corpus_format(path),
            "readable": probed["readable"],
        }
        if probed["readable"]:
            entry["stats"] = probed["stats"]
        elif probed["blockedBy"]:
            entry["blockedBy"] = probed["blockedBy"]
        entries.append(entry)

    # Ladder first, then the settled changes, then the rest, then blocked;
    # stable within each group.
    order = {"Start here": 0, "Sound laws": 1, "More corpora": 2,
             "Not readable in this format yet": 3}
    entries.sort(key=lambda e: order[e["group"]])

    text = (
        "// Generated by scripts/corpora.js. Do not edit by hand.\n"
        "//\n"
        "// Every corpus the page can load, keyed by repository path, plus the\n"
        "// grouped list the example picker shows. The guide references these by\n"
        "// path rather than embedding its own copies.\n"
        "const CORPORA = " + json.dumps(corpora, indent=2, ensure_ascii=False) + ";\n\n"
        "const CORPUS_LIST = " + json.dumps(entries, indent=2, ensure_ascii=False) + ";\n"
    ).replace("scripts/corpora.js", "scripts/corpora.py")

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print("corpora: web/corpora.js is out of date; run scripts/corpora.py",
                  file=sys.stderr)
            return 1
        print("corpora: up to date")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    readable = sum(1 for e in entries if e["readable"])
    print(f"corpora: wrote {OUTPUT.relative_to(REPO)} "
          f"({len(entries)} corpora, {readable} readable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
