// Command oe_english trains a learned sound-correspondence model on the
// Old English → Modern English cognate corpus in cognates.tsv and prints a
// human-readable report. The corpus exercises the Great Vowel Shift, consonant
// losses (/kn-/, /gn-/, /x/), fricative voicing, and palatalization.
//
// Run with:
//
//	cd experiments/oe_english && go run .
package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	regulae "github.com/tresoldi/regulae"
)

// parseSegments splits an IPA string into segments, handling the multi-char
// graphemes used in the Old English / Modern English corpus. Long vowels and
// diphthongs are consumed as single graphemes; the length mark ː is kept as
// part of the grapheme.
func parseSegments(ipa string) []regulae.Segment {
	multi := []string{
		"aː", "æː", "ɑː", "eː", "iː", "oː", "ɔː", "uː", "yː",
		"aɪ", "oʊ", "aʊ", "ɔɪ", "eɪ",
		"tʃ", "dʒ",
	}
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

// parseBreaks parses a comma-separated list of integers from a TSV column.
// Returns nil if the column is "-" or empty.
func parseBreaks(s string) []int {
	if s == "" || s == "-" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
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
	hasBreaks := len(header) >= 4 && header[3] == "old_english_breaks"

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
		gloss, oe, me := parts[0], parts[1], parts[2]
		var oeBreaks []int
		if hasBreaks && len(parts) >= 4 {
			oeBreaks = parseBreaks(parts[3])
		}
		out = append(out, labeledPair{
			gloss: gloss,
			src: regulae.Form{
				LectID:         "old_english",
				Segments:       parseSegments(oe),
				MorphemeBreaks: oeBreaks,
			},
			tgt: regulae.Form{
				LectID:   "modern_english",
				Segments: parseSegments(me),
			},
		})
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

	fmt.Printf("Loaded %d Old English → Modern English cognate pairs.\n\n", len(labeled))

	fmt.Println("Training learned model...")
	cognateSets := regulae.CognateSetsFromPairs(pairs, [2]string{"old_english", "modern_english"}, "oe_english")
	multiModel, err := regulae.TrainModel(cognateSets, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}
	trained, ok := multiModel.PairwiseModel("old_english", "modern_english")
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
	opts.TopDisplacements = 10
	fmt.Println(regulae.FormatModel(trained, opts))

	// Cost comparison.
	fmt.Println()
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Headline numbers")
	fmt.Println(strings.Repeat("=", 70))
	priorTotal, learnedTotal := 0.0, 0.0
	for _, lp := range labeled {
		aP, _ := regulae.AlignForms(lp.src, lp.tgt, "descriptive", 0, nil)
		cP, _ := regulae.AlignmentCost(aP, "descriptive", nil)
		aL, _ := regulae.AlignForms(lp.src, lp.tgt, "descriptive", 0, trained)
		cL, _ := regulae.AlignmentCost(aL, "descriptive", trained)
		priorTotal += cP
		learnedTotal += cL
	}
	fmt.Println("Full corpus:")
	fmt.Printf("  Prior-only total cost: %.2f\n", priorTotal)
	fmt.Printf("  Learned-model total cost: %.2f\n", learnedTotal)
	fmt.Printf("  Reduction:     %.2f\n", priorTotal-learnedTotal)

	// Selected alignments illustrating specific phenomena.
	toShow := []string{
		"stone",  // GVS: ā → oʊ
		"tooth",  // GVS: ō → uː
		"house",  // GVS: ū → aʊ
		"wife",   // GVS: ī → aɪ
		"see",    // GVS: ē → iː
		"knee",   // initial kn- → n-
		"knight", // kn- + x loss
		"night",  // x loss
		"light",  // x loss + GVS
		"mother", // intervocalic θ → ð
		"cheese", // palatalization k → tʃ
		"church", // palatalization
		"blood",  // shortening ō → ʌ
		"foot",   // shortening ō → ʊ
		"fish",   // sk → ʃ
	}
	fmt.Println()
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Selected alignments (trained model)")
	fmt.Println(strings.Repeat("=", 70))
	byGloss := map[string]labeledPair{}
	for _, lp := range labeled {
		byGloss[lp.gloss] = lp
	}
	for _, gloss := range toShow {
		lp, ok := byGloss[gloss]
		if !ok {
			continue
		}
		al, _ := regulae.AlignForms(lp.src, lp.tgt, "descriptive", 0, trained)
		fmt.Printf("\n[%s]\n%s\n", gloss, regulae.FormatAlignment(al, "descriptive", true, false))
	}
}
