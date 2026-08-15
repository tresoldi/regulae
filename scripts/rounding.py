import pathlib, itertools
FRONT_ROUND = ["y", "ø"]; FRONT_PLAIN = ["i", "e"]
ONSETS = ["k", "t", "m", "n", "l", "s"]
rows = ["cognate_id\tlect_id\tsegments"]
i = 0
for o, v2 in itertools.product(ONSETS, FRONT_ROUND + FRONT_PLAIN):
    for v1 in ["a", "o"]:
        applies = v2 in FRONT_ROUND
        rows.append(f"w{i:02d}\tproto\t{o} {v1} p {v2}")
        rows.append(f"w{i:02d}\tdaughter\t{o} {v1} {'f' if applies else 'p'} {v2}")
        i += 1
pathlib.Path("testdata/soundlaws/rounding_harmony.tsv").write_text("\n".join(rows) + "\n")
print(f"{i} sets, {i//2} showing the change")
