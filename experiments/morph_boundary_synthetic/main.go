// Command morph_boundary_synthetic trains the same synthetic corpus twice —
// once without morpheme-boundary annotations and once with them — and
// compares the promoted chunk tables to show that morpheme boundaries prevent
// morphology-straddling chunks from being promoted.
//
// Run with:
//
//	cd experiments/morph_boundary_synthetic && go run .
package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	regulae "github.com/tresoldi/regulae"
)

// parseForm builds a Form for lect from the IPA column (ipa) and the breaks
// field (breaksField). The IPA may contain "+" markers (morpheme boundaries
// inline); they are stripped when counting segment positions. breaksField is
// either "-" (no breaks) or a comma-separated list of integer positions.
// When withBoundaries is false, breaks are ignored (breaksField treated as "-").
func parseForm(lect, ipa, breaksField string, withBoundaries bool) regulae.Form {
	// Strip "+" markers; positions count over the filtered list.
	var segs []regulae.Segment
	for _, ch := range ipa {
		if ch != '+' {
			segs = append(segs, regulae.Segment{Grapheme: string(ch)})
		}
	}

	if !withBoundaries || breaksField == "-" || breaksField == "" {
		return regulae.Form{LectID: lect, Segments: segs}
	}

	var breaks []int
	for _, tok := range strings.Split(breaksField, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		n, err := strconv.Atoi(tok)
		if err != nil {
			continue
		}
		breaks = append(breaks, n)
	}
	return regulae.Form{LectID: lect, Segments: segs, MorphemeBreaks: breaks}
}

type row struct {
	proto         string
	derived       string
	protoBreaks   string
	derivedBreaks string
}

func loadRows(path string) ([]row, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	if !sc.Scan() {
		return nil, fmt.Errorf("empty file: %s", path)
	}
	header := strings.TrimSpace(sc.Text())
	want := "gloss\tproto\tderived\tproto_breaks\tderived_breaks"
	if header != want {
		return nil, fmt.Errorf("unexpected header: %q", header)
	}

	var rows []row
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) != 5 {
			continue
		}
		rows = append(rows, row{
			proto:         parts[1],
			derived:       parts[2],
			protoBreaks:   parts[3],
			derivedBreaks: parts[4],
		})
	}
	return rows, sc.Err()
}

func buildPairs(rows []row, withBoundaries bool) []regulae.FormPair {
	pairs := make([]regulae.FormPair, len(rows))
	for i, r := range rows {
		pairs[i] = regulae.FormPair{
			Src: parseForm("proto", r.proto, r.protoBreaks, withBoundaries),
			Tgt: parseForm("derived", r.derived, r.derivedBreaks, withBoundaries),
		}
	}
	return pairs
}

func summariseChunks(multi *regulae.MultiLectModel) string {
	pair, ok := multi.PairwiseModel("proto", "derived")
	if !ok {
		return "  (pairwise model not found)"
	}
	if len(pair.ChunkTable.Entries) == 0 {
		return "  (none promoted)"
	}

	type entry struct {
		key  string
		cost float64
	}
	var entries []entry
	for k, cost := range pair.ChunkTable.Entries {
		entries = append(entries, entry{k, cost})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].cost < entries[j].cost })

	var lines []string
	for _, e := range entries {
		cp := pair.ChunkTable.Pairs[e.key]
		var sb, tb strings.Builder
		for _, seg := range cp.Src {
			sb.WriteString(seg.Grapheme)
		}
		for _, seg := range cp.Tgt {
			tb.WriteString(seg.Grapheme)
		}
		lines = append(lines, fmt.Sprintf("  (%s, %s): cost=%.3f", sb.String(), tb.String(), e.cost))
	}
	return strings.Join(lines, "\n")
}

func main() {
	path := "cognates.tsv"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}

	rows, err := loadRows(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load error:", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d pairs.\n\n", len(rows))

	// Train without morpheme boundaries.
	pairsNo := buildPairs(rows, false)
	corpusNo := regulae.CognateSetsFromPairs(pairsNo, [2]string{"proto", "derived"}, "mb")
	modelNo, err := regulae.TrainModel(corpusNo, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error (no boundaries):", err)
		os.Exit(1)
	}
	fmt.Println("Without morpheme boundaries — chunks promoted:")
	fmt.Println(summariseChunks(modelNo))
	fmt.Println()

	// Train with morpheme boundaries.
	pairsYes := buildPairs(rows, true)
	corpusYes := regulae.CognateSetsFromPairs(pairsYes, [2]string{"proto", "derived"}, "mb")
	modelYes, err := regulae.TrainModel(corpusYes, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error (with boundaries):", err)
		os.Exit(1)
	}
	fmt.Println("With morpheme boundaries — chunks promoted:")
	fmt.Println(summariseChunks(modelYes))
	fmt.Println()

	// Compute dropped chunks (in no-boundary but not in with-boundary).
	pairNo, okNo := modelNo.PairwiseModel("proto", "derived")
	pairYes, okYes := modelYes.PairwiseModel("proto", "derived")
	if !okNo || !okYes {
		fmt.Println("Boundary-rejected chunks: (models unavailable)")
		return
	}

	type chunkKey struct{ src, tgt string }
	inNo := map[chunkKey]struct{}{}
	for k := range pairNo.ChunkTable.Entries {
		cp := pairNo.ChunkTable.Pairs[k]
		var sb, tb strings.Builder
		for _, seg := range cp.Src {
			sb.WriteString(seg.Grapheme)
		}
		for _, seg := range cp.Tgt {
			tb.WriteString(seg.Grapheme)
		}
		inNo[chunkKey{sb.String(), tb.String()}] = struct{}{}
	}
	inYes := map[chunkKey]struct{}{}
	for k := range pairYes.ChunkTable.Entries {
		cp := pairYes.ChunkTable.Pairs[k]
		var sb, tb strings.Builder
		for _, seg := range cp.Src {
			sb.WriteString(seg.Grapheme)
		}
		for _, seg := range cp.Tgt {
			tb.WriteString(seg.Grapheme)
		}
		inYes[chunkKey{sb.String(), tb.String()}] = struct{}{}
	}

	var dropped []chunkKey
	for ck := range inNo {
		if _, ok := inYes[ck]; !ok {
			dropped = append(dropped, ck)
		}
	}
	sort.Slice(dropped, func(i, j int) bool {
		if dropped[i].src != dropped[j].src {
			return dropped[i].src < dropped[j].src
		}
		return dropped[i].tgt < dropped[j].tgt
	})

	fmt.Printf("Boundary-rejected chunks (%d):\n", len(dropped))
	for _, ck := range dropped {
		fmt.Printf("  (%s, %s)\n", ck.src, ck.tgt)
	}
}
