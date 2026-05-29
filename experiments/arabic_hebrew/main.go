// Command arabic_hebrew trains a learned sound-correspondence model on the
// 32-pair Arabic→Hebrew cognate corpus in cognates.tsv and prints a
// human-readable report. This is an experiment (a report, not a pass/fail
// test): it shows what the framework recovers on real Proto-Semitic data.
//
// Run with:
//
//	cd experiments/arabic_hebrew && go run .
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	regulae "github.com/tresoldi/regulae"
)

// multi-char IPA clusters used in this corpus
var multi = []string{
	"aː", "eː", "iː", "oː", "uː",
	"tsˤ", "ts", "dˤ", "tˤ", "sˤ", "zˤ", "ðˤ",
	"tʃ", "dʒ",
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
	// header consumed; columns: gloss, arabic, hebrew[, arabic_breaks, hebrew_breaks]
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
		out = append(out, labeledPair{
			gloss: parts[0],
			src:   regulae.Form{LectID: "arabic", Segments: parseForm(parts[1])},
			tgt:   regulae.Form{LectID: "hebrew", Segments: parseForm(parts[2])},
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

	fmt.Printf("Loaded %d Arabic → Hebrew cognate pairs.\n", len(labeled))

	corpus := regulae.CognateSetsFromPairs(pairs, [2]string{"arabic", "hebrew"}, "ah")
	multi, err := regulae.TrainModel(corpus, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}

	trained, ok := multi.PairwiseModel("arabic", "hebrew")
	if !ok {
		fmt.Fprintln(os.Stderr, "pairwise model not found")
		os.Exit(1)
	}

	opts := regulae.DefaultFormatModelOptions()
	opts.TopSegments = 20
	fmt.Println(regulae.FormatModel(trained, opts))
}
