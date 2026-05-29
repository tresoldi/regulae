// Command ppn_hawaiian trains a learned sound-correspondence model on the
// Proto-Polynesian → Hawaiian cognate corpus in cognates.tsv and prints a
// human-readable report. The corpus exercises near-exceptionless mergers:
// *t → k, *k → ʔ, *f → h, *ŋ → n, *s → h.
//
// Run with:
//
//	cd experiments/ppn_hawaiian && go run .
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	regulae "github.com/tresoldi/regulae"
)

// parseSegments splits an IPA string into segments, handling multi-char
// graphemes that appear in the PPN/Hawaiian corpus.
func parseSegments(ipa string) []regulae.Segment {
	multi := []string{"tʃ", "dʒ", "ts"}
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
	// header: gloss ppn hawaiian
	var out []labeledPair
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) != 3 {
			continue
		}
		out = append(out, labeledPair{
			gloss: parts[0],
			src:   regulae.Form{LectID: "ppn", Segments: parseSegments(parts[1])},
			tgt:   regulae.Form{LectID: "hawaiian", Segments: parseSegments(parts[2])},
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

	fmt.Printf("Loaded %d Proto-Polynesian → Hawaiian cognate pairs.\n\n", len(labeled))

	fmt.Println("Training learned model...")
	cognateSets := regulae.CognateSetsFromPairs(pairs, [2]string{"ppn", "hawaiian"}, "ppn_hawaiian")
	multiModel, err := regulae.TrainModel(cognateSets, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}
	trained, ok := multiModel.PairwiseModel("ppn", "hawaiian")
	if !ok {
		fmt.Fprintln(os.Stderr, "pairwise model not found")
		os.Exit(1)
	}
	fmt.Println()

	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Trained model summary")
	fmt.Println(strings.Repeat("=", 70))
	opts := regulae.DefaultFormatModelOptions()
	opts.TopSegments = 20
	opts.TopDisplacements = 8
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
		"one",    // *t → k merger
		"three",  // *t → k + *r → l
		"four",   // *f → h
		"seven",  // *f → h + *t → k
		"person", // complex: *t → k, *ŋ → n
		"mouth",  // *ŋ → n
		"skin",   // *k → ʔ
		"chief",  // *r → l + *k → ʔ
		"bird",   // conservation
		"ten",    // conservation + *f → h
		"go",     // f→h in isolation
		"canoe",  // k→ʔ
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
