// Command finnish_estonian trains a learned sound-correspondence model on the
// 40-pair Finnish→Estonian cognate corpus in cognates.tsv and prints a
// human-readable report. Demonstrates vowel-length, harmony, and final-vowel
// correspondences.
//
// Run with:
//
//	cd experiments/finnish_estonian && go run .
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

var multi = []string{
	"ɑː", "eː", "iː", "oː", "uː", "æː", "øː", "yː",
}

func parseForm(ipa string) []regulae.Segment {
	runes := []rune(ipa)
	var segs []regulae.Segment
	i := 0
	for i < len(runes) {
		hit := false
		for _, cluster := range multi {
			cr := []rune(cluster)
			if i+len(cr) <= len(runes) && string(runes[i:i+len(cr)]) == cluster {
				segs = append(segs, regulae.Segment{Grapheme: cluster})
				i += len(cr)
				hit = true
				break
			}
		}
		if !hit {
			segs = append(segs, regulae.Segment{Grapheme: string(runes[i])})
			i++
		}
	}
	return segs
}

func parseBreaks(s string) []int {
	if s == "-" || s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	var out []int
	for _, p := range parts {
		n, err := strconv.Atoi(strings.TrimSpace(p))
		if err == nil {
			out = append(out, n)
		}
	}
	return out
}

type labeledPair struct {
	gloss string
	src   regulae.Form
	tgt   regulae.Form
}

func loadCorpus(path string) ([]labeledPair, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	if !sc.Scan() {
		return nil, fmt.Errorf("empty file")
	}
	header := strings.Split(strings.TrimSpace(sc.Text()), "\t")
	hasBreaks := len(header) >= 5 && header[3] == "finnish_breaks" && header[4] == "estonian_breaks"

	var out []labeledPair
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) < 3 {
			continue
		}
		src := regulae.Form{LectID: "finnish", Segments: parseForm(parts[1])}
		tgt := regulae.Form{LectID: "estonian", Segments: parseForm(parts[2])}
		if hasBreaks && len(parts) >= 5 {
			src.MorphemeBreaks = parseBreaks(parts[3])
			tgt.MorphemeBreaks = parseBreaks(parts[4])
		}
		out = append(out, labeledPair{gloss: parts[0], src: src, tgt: tgt})
	}
	return out, sc.Err()
}

func main() {
	path := "cognates.tsv"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	labeled, err := loadCorpus(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load error:", err)
		os.Exit(1)
	}

	pairs := make([]regulae.FormPair, len(labeled))
	for i, lp := range labeled {
		pairs[i] = regulae.FormPair{Src: lp.src, Tgt: lp.tgt}
	}

	fmt.Printf("Loaded %d Finnish → Estonian cognate pairs.\n\n", len(labeled))

	cs := regulae.CognateSetsFromPairs(pairs, [2]string{"finnish", "estonian"}, "fe")
	multiModel, err := regulae.TrainModel(cs, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}

	trained, ok := multiModel.PairwiseModel("finnish", "estonian")
	if !ok {
		fmt.Fprintln(os.Stderr, "pairwise model not found")
		os.Exit(1)
	}

	opts := regulae.DefaultFormatModelOptions()
	opts.TopSegments = 20
	fmt.Println(regulae.FormatModel(trained, opts))

	// Source segments with competing target correspondences (unconditioned only).
	type tgtCount struct {
		tgt   string
		count float64
	}
	bySrc := map[string][]tgtCount{}
	for k, count := range trained.SegmentTable.Counts {
		cc := trained.SegmentTable.Corr[k]
		if cc.Context.ConstraintCount() != 0 {
			continue
		}
		bySrc[cc.Src] = append(bySrc[cc.Src], tgtCount{cc.Tgt, count})
	}

	// Sort by descending total count, keep top 10 with total >= 2.
	type srcEntry struct {
		src   string
		tgts  []tgtCount
		total float64
	}
	var entries []srcEntry
	for src, tgts := range bySrc {
		total := 0.0
		for _, tc := range tgts {
			total += tc.count
		}
		if total < 2 {
			continue
		}
		sort.Slice(tgts, func(i, j int) bool { return tgts[i].count > tgts[j].count })
		entries = append(entries, srcEntry{src, tgts, total})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].total > entries[j].total })

	fmt.Println("\nSource segments with competing target correspondences:")
	shown := 0
	for _, e := range entries {
		if shown >= 10 {
			break
		}
		top := e.tgts
		if len(top) > 3 {
			top = top[:3]
		}
		var parts []string
		for _, tc := range top {
			parts = append(parts, fmt.Sprintf("%s×%.0f", tc.tgt, tc.count))
		}
		extra := ""
		if len(e.tgts) > 3 {
			extra = fmt.Sprintf(" (+%d more)", len(e.tgts)-3)
		}
		fmt.Printf("  %q: %s%s  [total %.0f]\n", e.src, strings.Join(parts, ", "), extra, e.total)
		shown++
	}
}
