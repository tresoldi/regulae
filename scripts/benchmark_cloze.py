#!/usr/bin/env python3
"""Leave-one-reflex-out cloze, regulae against lingrex CoPaR, on shared families.

A reference tool has to say where it stands against the field's existing one.
lingrex's CoPaR clusters aligned columns into correspondence patterns and imputes
a missing reflex from them; regulae trains a conditioned correspondence model and
predicts a held-out reflex from it. This runs both on the same Lexibank families,
holds out one lect's reflex per cognate set (the same set for both tools), and
scores an exact whole-reflex match the same way, so the two numbers are
comparable.

The point is not a leaderboard -- exact-form cloze is a hard, noisy metric and
the two tools answer overlapping but different questions -- but a stated
comparison: regulae recovers a comparable share of held-out reflexes while also
reporting the environment each correspondence is conditioned on and whether it
stands against a null, which CoPaR does not.

Both take an integer cognate id; regulae reads its own corpus TSV and CoPaR reads
a lingpy wordlist built from the same rows. CoPaR predicts only the sites its
patterns cover, so its denominator can be smaller than the held-out count; the
`n` column is what it actually predicted.

Usage:
    scripts/lexibank.py <dataset> --clone ~/lexibank_clone -o <name>.tsv
    scripts/benchmark_cloze.py <name>=<name>.tsv [more...] --folds 3

Needs `lingpy` and `lingrex` (`pip install lingrex`), and a built `regulae`.
"""
import csv, json, subprocess, collections, argparse
from lingpy import Wordlist
from lingpy.sequence.sound_classes import prosodic_string
from lingrex.copar import CoPaR


def read_rows(path):
    out = []
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            tokens = row["segments"].strip()
            if tokens:
                out.append((row["lect_id"], row["cognate_id"], tokens.split()))
    return out


def hold_one_per_set(rows, seed):
    """One held-out (lect, cognate_id) -> true tokens, per set covering >= 3 lects."""
    by_cognate = collections.defaultdict(list)
    for lect, cognate, tokens in rows:
        by_cognate[cognate].append((lect, tokens))
    held = {}
    for cognate, members in sorted(by_cognate.items()):
        if len(members) >= 3:
            members = sorted(members)
            pick = members[seed % len(members)]
            held[(pick[0], cognate)] = pick[1]
    return held


def copar_cloze(rows, held):
    data = {0: ["doculect", "concept", "ipa", "tokens", "structure", "cogid"]}
    index = 1
    cognate_ids = {}
    for lect, cognate, tokens in rows:
        if (lect, cognate) in held:
            continue
        if cognate not in cognate_ids:
            cognate_ids[cognate] = len(cognate_ids) + 1
        try:
            structure = " ".join(prosodic_string(tokens, _output="CcV"))
        except Exception:
            structure = " ".join("c" for _ in tokens)
        data[index] = [lect, cognate, "".join(tokens), " ".join(tokens),
                       structure, cognate_ids[cognate]]
        index += 1
    cop = CoPaR(Wordlist(data), ref="cogid", segments="tokens", minrefs=2,
                transcription="ipa", structure="structure")
    cop.align()
    cop.get_sites()
    cop.cluster_sites()
    cop.sites_to_pattern()
    words, _, _ = cop.predict_words()
    inverse = {value: key for key, value in cognate_ids.items()}
    correct = total = 0
    for numeric_cognate, predictions in words.items():
        cognate = inverse.get(numeric_cognate)
        for lect, predicted in predictions.items():
            if (lect, cognate) not in held:
                continue
            total += 1
            tokens = predicted.split() if isinstance(predicted, str) else list(predicted)
            # CoPaR marks an unpredicted site "\u00d8" and an alignment gap "-",
            # and a pattern compatible with several sounds "a|b"; take its top
            # option and drop the non-predictions, then require the whole reflex.
            guess = [t.split("|")[0] for t in tokens if t not in ("-", "\u00d8")]
            if guess == held[(lect, cognate)]:
                correct += 1
    return correct, total


def regulae_cloze(binary, path, folds):
    completed = subprocess.run(
        [binary, "train", "--predictive-folds", str(folds), "--json", path],
        capture_output=True, text=True)
    model = json.loads(completed.stdout)
    lolo = model["fit"]["predictive"].get("leave_one_lect_out_conditioned", {})
    return lolo.get("top1_coverage"), lolo.get("observations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("families", nargs="+", help="name=path.tsv")
    parser.add_argument("--binary", default="./build/c/regulae")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(f"| {'family':20} | {'sets':>4} | {'lects':>5} | {'regulae':>7} | "
          f"{'CoPaR':>6} | {'n':>4} |")
    print(f"| {'-'*20} | {'-'*4} | {'-'*5} | {'-'*7} | {'-'*6} | {'-'*4} |")
    for spec in args.families:
        name, path = spec.split("=", 1)
        rows = read_rows(path)
        sets = len({c for _, c, _ in rows})
        lects = len({l for l, _, _ in rows})
        held = hold_one_per_set(rows, args.seed)
        reg, _ = regulae_cloze(args.binary, path, args.folds)
        ok, total = copar_cloze(rows, held)
        reg_str = f"{reg:.3f}" if reg is not None else "n/a"
        cop_str = f"{ok/total:.3f}" if total else "n/a"
        print(f"| {name:20} | {sets:4d} | {lects:5d} | {reg_str:>7} | "
              f"{cop_str:>6} | {total:4d} |", flush=True)


if __name__ == "__main__":
    main()
