// Command latin_spanish trains a learned sound-correspondence model on the
// ~95-pair Latin→Spanish cognate corpus in cognates.tsv and prints a
// human-readable report. This is an experiment (a report, not a pass/fail
// test): it shows what the framework recovers on real data.
//
// Run with:
//
//	cd experiments/latin_spanish && go run .
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	regulae "github.com/tresoldi/regulae"
)

// parseSegments splits an IPA string into segments, handling the multi-character
// graphemes used in this corpus.
func parseSegments(ipa string) []regulae.Segment {
	multi := []string{"tʃ", "dʒ", "nj", "ts"}
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
			src:   regulae.Form{LectID: "latin", Segments: parseSegments(parts[1])},
			tgt:   regulae.Form{LectID: "spanish", Segments: parseSegments(parts[2])},
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

	corpus := make([]regulae.FormPair, len(labeled))
	for i, lp := range labeled {
		corpus[i] = regulae.FormPair{Src: lp.src, Tgt: lp.tgt}
	}
	model, err := regulae.TrainPairwise(corpus, nil, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}

	fmt.Printf("Latin → Spanish experiment: %d cognate pairs\n\n", len(labeled))
	fmt.Println(regulae.FormatModel(model, regulae.DefaultFormatModelOptions()))

	// Cost reduction: prior-only vs learned.
	priorTotal, learnedTotal := 0.0, 0.0
	for _, lp := range labeled {
		aP, _ := regulae.AlignForms(lp.src, lp.tgt, "descriptive", 0, nil)
		cP, _ := regulae.AlignmentCost(aP, "descriptive", nil)
		aL, _ := regulae.AlignForms(lp.src, lp.tgt, "descriptive", 0, model)
		cL, _ := regulae.AlignmentCost(aL, "descriptive", model)
		priorTotal += cP
		learnedTotal += cL
	}
	fmt.Printf("\nPrior-only total cost:    %.2f\n", priorTotal)
	fmt.Printf("Learned-model total cost: %.2f\n", learnedTotal)
	fmt.Printf("Reduction:                %.2f\n", priorTotal-learnedTotal)

	// A few illustrative alignments.
	fmt.Println("\nSelected alignments (trained model):")
	byGloss := map[string]labeledPair{}
	for _, lp := range labeled {
		byGloss[lp.gloss] = lp
	}
	for _, gloss := range []string{"father", "mother", "night", "head", "tooth"} {
		lp, ok := byGloss[gloss]
		if !ok {
			continue
		}
		al, _ := regulae.AlignForms(lp.src, lp.tgt, "descriptive", 0, model)
		fmt.Printf("\n[%s]\n%s\n", gloss, regulae.FormatAlignment(al, "descriptive", true, false))
	}
}
