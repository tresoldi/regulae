// Command turkish_azerbaijani trains a learned sound-correspondence model on
// the Turkish → Azerbaijani cognate corpus in cognates.tsv and prints a
// human-readable report. The corpus exercises word-initial k~q/g alternation,
// word-final k~χ, ğ fortition, and Turkic vowel harmony.
//
// Run with:
//
//	cd experiments/turkish_azerbaijani && go run .
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
// graphemes used in the Turkish/Azerbaijani corpus.
func parseSegments(ipa string) []regulae.Segment {
	multi := []string{
		"tʃ", "dʒ",
		"aː", "eː", "iː", "oː", "uː",
		"øː", "yː", "æː", "əː", "ɑː",
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
	hasBreaks := len(header) >= 5 && header[3] == "turkish_breaks" && header[4] == "azerbaijani_breaks"

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
		gloss, tr, az := parts[0], parts[1], parts[2]
		var trBreaks, azBreaks []int
		if hasBreaks && len(parts) >= 5 {
			trBreaks = parseBreaks(parts[3])
			azBreaks = parseBreaks(parts[4])
		}
		out = append(out, labeledPair{
			gloss: gloss,
			src: regulae.Form{
				LectID:         "turkish",
				Segments:       parseSegments(tr),
				MorphemeBreaks: trBreaks,
			},
			tgt: regulae.Form{
				LectID:         "azerbaijani",
				Segments:       parseSegments(az),
				MorphemeBreaks: azBreaks,
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

	fmt.Printf("Loaded %d Turkish → Azerbaijani cognate pairs.\n\n", len(labeled))

	cognateSets := regulae.CognateSetsFromPairs(pairs, [2]string{"turkish", "azerbaijani"}, "turkish_azerbaijani")
	multiModel, err := regulae.TrainModel(cognateSets, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}
	trained, ok := multiModel.PairwiseModel("turkish", "azerbaijani")
	if !ok {
		fmt.Fprintln(os.Stderr, "pairwise model not found")
		os.Exit(1)
	}

	opts := regulae.DefaultFormatModelOptions()
	opts.TopSegments = 20
	fmt.Println(regulae.FormatModel(trained, opts))
}
