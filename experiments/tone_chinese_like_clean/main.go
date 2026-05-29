// Command tone_chinese_like_clean trains a model on a 72-pair synthetic
// corpus with a clean tonogenesis rule (voiceless initial → tone preserved;
// voiced/sonorant initial → tone += 3). This validates cross-dimensional
// discovery on unambiguous signal.
//
// Tone notation: trailing digits stripped and attached to the last vowel
// (vowel set: aeiou) or last segment if no vowel found.
//
// Run with:
//
//	cd experiments/tone_chinese_like_clean && go run .
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	regulae "github.com/tresoldi/regulae"
)

var vowels = map[rune]bool{
	'a': true, 'e': true, 'i': true, 'o': true, 'u': true,
}

// parseTonedForm splits an IPA string like "pi1" into segments, attaching
// the trailing tone digits to the last vowel (or last segment if no vowel).
func parseTonedForm(ipa string) []regulae.Segment {
	runes := []rune(ipa)
	i := len(runes)
	for i > 0 && runes[i-1] >= '0' && runes[i-1] <= '9' {
		i--
	}
	segmental := runes[:i]
	tone := ""
	if i < len(runes) {
		tone = string(runes[i:])
	}

	segs := make([]regulae.Segment, len(segmental))
	for j, r := range segmental {
		segs[j] = regulae.Segment{Grapheme: string(r)}
	}

	if tone != "" {
		attached := false
		for j := len(segs) - 1; j >= 0; j-- {
			if vowels[[]rune(segs[j].Grapheme)[0]] {
				segs[j].Tone = tone
				attached = true
				break
			}
		}
		if !attached && len(segs) > 0 {
			segs[len(segs)-1].Tone = tone
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
			src:   regulae.Form{LectID: "mandarin-like", Segments: parseTonedForm(parts[1])},
			tgt:   regulae.Form{LectID: "cantonese-like", Segments: parseTonedForm(parts[2])},
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

	fmt.Printf("Loaded %d synthetic clean Mandarin/Cantonese-like pairs.\n", len(labeled))
	fmt.Println()
	fmt.Println("NOTE: rule-explicit synthetic fixture. Designed to validate")
	fmt.Println("cross-dimensional discovery commit loop on clean signal.")
	fmt.Println()

	pairs := make([]regulae.FormPair, len(labeled))
	for i, lp := range labeled {
		pairs[i] = regulae.FormPair{Src: lp.src, Tgt: lp.tgt}
	}
	cognateSets := regulae.CognateSetsFromPairs(pairs, [2]string{"mandarin-like", "cantonese-like"}, "pair")
	multiModel, err := regulae.TrainModel(cognateSets, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}
	trained, ok := multiModel.PairwiseModel("mandarin-like", "cantonese-like")
	if !ok {
		fmt.Fprintln(os.Stderr, "no pairwise model for mandarin-like/cantonese-like")
		os.Exit(1)
	}

	opts := regulae.DefaultFormatModelOptions()
	opts.TopSegments = 20
	fmt.Println(regulae.FormatModel(trained, opts))

	fmt.Println()
	fmt.Println("Cross-dimensional rules in detail:")
	for i, rule := range trained.CrossDimensionalTable.Entries {
		fmt.Printf("#%d: src[%s@%s] → tgt[%s=%s@%+d]  count=%.0f  conf=%.2f\n",
			i,
			rule.SrcFeature.Feature,
			rule.SrcPosition,
			rule.TgtDimension,
			rule.TgtValue,
			rule.TgtPositionOffset,
			rule.Count,
			rule.Confidence,
		)
	}

	priorTotal := 0.0
	learnedTotal := 0.0
	for _, lp := range labeled {
		aP, _ := regulae.AlignForms(lp.src, lp.tgt, "descriptive", 0, nil)
		cP, _ := regulae.AlignmentCost(aP, "descriptive", nil)
		aL, _ := regulae.AlignForms(lp.src, lp.tgt, "descriptive", 0, trained)
		cL, _ := regulae.AlignmentCost(aL, "descriptive", trained)
		priorTotal += cP
		learnedTotal += cL
	}
	fmt.Println()
	fmt.Printf("Prior-only total cost:  %.2f\n", priorTotal)
	fmt.Printf("Learned-model total cost: %.2f\n", learnedTotal)
}
