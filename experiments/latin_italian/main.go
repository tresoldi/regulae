// Command latin_italian trains a learned sound-correspondence model on the
// Latin→Italian cognate corpus in cognates.tsv and prints a human-readable
// report. Italian is the most conservative Romance branch: it preserves
// geminate consonants, keeps final vowels, and has a milder reduction profile
// than Spanish or French.
//
// Run with:
//
//	cd experiments/latin_italian && go run .
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

var multi = []string{"tʃ", "dʒ", "ɲ", "ʎ"}

func parseSegments(ipa string) []regulae.Segment {
	runes := []rune(ipa)
	var segs []regulae.Segment
	i := 0
	for i < len(runes) {
		matched := false
		for _, cluster := range multi {
			cr := []rune(cluster)
			if i+len(cr) <= len(runes) && string(runes[i:i+len(cr)]) == cluster {
				segs = append(segs, regulae.Segment{Grapheme: cluster})
				i += len(cr)
				matched = true
				break
			}
		}
		if !matched {
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
	var out []int
	for _, p := range strings.Split(s, ",") {
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
	hasBreaks := len(header) >= 5 && header[3] == "latin_breaks" && header[4] == "italian_breaks"

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
		src := regulae.Form{LectID: "latin", Segments: parseSegments(parts[1])}
		tgt := regulae.Form{LectID: "italian", Segments: parseSegments(parts[2])}
		if hasBreaks && len(parts) >= 5 {
			src.MorphemeBreaks = parseBreaks(parts[3])
			tgt.MorphemeBreaks = parseBreaks(parts[4])
		}
		out = append(out, labeledPair{gloss: parts[0], src: src, tgt: tgt})
	}
	return out, sc.Err()
}

func reportCosts(label string, labeled []labeledPair, model *regulae.LearnedModel) {
	priorTotal, learnedTotal := 0.0, 0.0
	for _, lp := range labeled {
		aP, _ := regulae.AlignForms(lp.src, lp.tgt, "descriptive", 0, nil)
		cP, _ := regulae.AlignmentCost(aP, "descriptive", nil)
		aL, _ := regulae.AlignForms(lp.src, lp.tgt, "descriptive", 0, model)
		cL, _ := regulae.AlignmentCost(aL, "descriptive", model)
		priorTotal += cP
		learnedTotal += cL
	}
	fmt.Printf("%s:\n", label)
	fmt.Printf("  Prior-only total cost: %.2f\n", priorTotal)
	fmt.Printf("  Learned-model total cost: %.2f\n", learnedTotal)
	fmt.Printf("  Reduction:     %.2f\n", priorTotal-learnedTotal)
}

func showAlignments(labeled []labeledPair, model *regulae.LearnedModel) {
	toShow := []string{
		"father", "night", "eight", "milk", "head",
		"sky", "five", "wolf", "water", "king", "tooth",
	}
	byGloss := map[string]labeledPair{}
	for _, lp := range labeled {
		byGloss[lp.gloss] = lp
	}
	fmt.Println()
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Selected alignments (trained model)")
	fmt.Println(strings.Repeat("=", 70))
	for _, gloss := range toShow {
		lp, ok := byGloss[gloss]
		if !ok {
			continue
		}
		al, _ := regulae.AlignForms(lp.src, lp.tgt, "descriptive", 0, model)
		fmt.Printf("\n[%s]\n%s\n", gloss, regulae.FormatAlignment(al, "descriptive", true, false))
	}
}

func findContextProblems(model *regulae.LearnedModel) {
	bySrc := map[string]map[string]float64{}
	for k, count := range model.SegmentTable.Counts {
		cc := model.SegmentTable.Corr[k]
		if cc.Context.ConstraintCount() != 0 {
			continue
		}
		if bySrc[cc.Src] == nil {
			bySrc[cc.Src] = map[string]float64{}
		}
		bySrc[cc.Src][cc.Tgt] = count
	}

	type entry struct {
		src    string
		topTwo []struct {
			tgt string
			n   float64
		}
		total float64
	}
	var competing []entry
	for src, targets := range bySrc {
		if len(targets) < 2 {
			continue
		}
		type kv struct {
			k string
			v float64
		}
		var sorted []kv
		for t, n := range targets {
			sorted = append(sorted, kv{t, n})
		}
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].v > sorted[j].v })
		if sorted[1].v < 2.0 {
			continue
		}
		total := 0.0
		for _, kv := range sorted {
			total += kv.v
		}
		top2 := []struct {
			tgt string
			n   float64
		}{{sorted[0].k, sorted[0].v}, {sorted[1].k, sorted[1].v}}
		competing = append(competing, entry{src, top2, total})
	}
	sort.Slice(competing, func(i, j int) bool { return competing[i].total > competing[j].total })

	fmt.Println()
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Unconditioned source segments with competing targets")
	fmt.Println(strings.Repeat("=", 70))
	if len(competing) == 0 {
		fmt.Println("  (none above threshold)")
		return
	}
	limit := 10
	if len(competing) < limit {
		limit = len(competing)
	}
	for _, e := range competing[:limit] {
		var parts []string
		topSum := 0.0
		for _, t := range e.topTwo {
			parts = append(parts, fmt.Sprintf("%s×%d", t.tgt, int(t.n)))
			topSum += t.n
		}
		others := e.total - topSum
		tail := ""
		if others > 0 {
			tail = fmt.Sprintf(" (+%d others)", int(others))
		}
		fmt.Printf("  %q: %s%s  [total %d]\n", e.src, strings.Join(parts, ", "), tail, int(e.total))
	}
}

func showContextSplits(model *regulae.LearnedModel) {
	type condEntry struct {
		cc    regulae.ConditionedCorrespondence
		count float64
	}
	bySrc := map[string][]condEntry{}
	total := 0
	for k, count := range model.SegmentTable.Counts {
		cc := model.SegmentTable.Corr[k]
		if cc.Context.ConstraintCount() == 0 {
			continue
		}
		bySrc[cc.Src] = append(bySrc[cc.Src], condEntry{cc, count})
		total++
	}

	fmt.Println()
	fmt.Println(strings.Repeat("=", 70))
	fmt.Printf("Context-conditioned splits (%d total)\n", total)
	fmt.Println(strings.Repeat("=", 70))

	var srcs []string
	for s := range bySrc {
		srcs = append(srcs, s)
	}
	sort.Strings(srcs)
	for _, src := range srcs {
		entries := bySrc[src]
		sort.Slice(entries, func(i, j int) bool { return entries[i].count > entries[j].count })
		for _, e := range entries {
			ctx := e.cc.Context
			var ctxParts []string
			if ctx.Position != "" {
				ctxParts = append(ctxParts, "pos="+ctx.Position)
			}
			if len(ctx.Preceding) > 0 {
				feats := make([]string, len(ctx.Preceding))
				for i, c := range ctx.Preceding {
					feats[i] = c.Feature
				}
				ctxParts = append(ctxParts, "prec=["+strings.Join(feats, ",")+"]")
			}
			if len(ctx.Following) > 0 {
				feats := make([]string, len(ctx.Following))
				for i, c := range ctx.Following {
					feats[i] = c.Feature
				}
				ctxParts = append(ctxParts, "foll=["+strings.Join(feats, ",")+"]")
			}
			fmt.Printf("  %s → %s: %d  %s\n", e.cc.Src, e.cc.Tgt, int(e.count), strings.Join(ctxParts, " "))
		}
	}
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

	fmt.Printf("Loaded %d Latin → Italian cognate pairs.\n\n", len(labeled))
	fmt.Println("Training learned model...")

	cs := regulae.CognateSetsFromPairs(pairs, [2]string{"latin", "italian"}, "li")
	multiModel, err := regulae.TrainModel(cs, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}

	trained, ok := multiModel.PairwiseModel("latin", "italian")
	if !ok {
		fmt.Fprintln(os.Stderr, "pairwise model not found")
		os.Exit(1)
	}
	fmt.Println()

	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Trained model summary")
	fmt.Println(strings.Repeat("=", 70))
	opts := regulae.DefaultFormatModelOptions()
	opts.TopSegments = 25
	opts.TopDisplacements = 8
	fmt.Println(regulae.FormatModel(trained, opts))

	fmt.Println()
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Headline numbers")
	fmt.Println(strings.Repeat("=", 70))
	reportCosts("Full corpus", labeled, trained)

	showAlignments(labeled, trained)
	findContextProblems(trained)
	showContextSplits(trained)
}
