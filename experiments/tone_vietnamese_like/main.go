// Command tone_vietnamese_like trains a learned model on a synthetic
// Vietnamese-like corpus (Hanoi vs. Saigon dialect, 6-tone system) where
// certain source tones lower by 1 in the target.
//
// Tone notation: trailing digits stripped and attached to the last vowel
// (vowel set: aeiouɛɔəɪʊ) or last segment if no vowel found.
//
// Run with:
//
//	cd experiments/tone_vietnamese_like && go run .
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
	'ɛ': true, 'ɔ': true, 'ə': true, 'ɪ': true, 'ʊ': true,
}

// parseTonedForm splits an IPA string like "mot6" into segments, attaching
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
			src:   regulae.Form{LectID: "hanoi-like", Segments: parseTonedForm(parts[1])},
			tgt:   regulae.Form{LectID: "saigon-like", Segments: parseTonedForm(parts[2])},
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

	fmt.Printf("Loaded %d Vietnamese-like toned pairs.\n", len(labeled))
	fmt.Println()
	fmt.Println("NOTE: synthetic fixture, not real dialectal data.")
	fmt.Println()

	pairs := make([]regulae.FormPair, len(labeled))
	for i, lp := range labeled {
		pairs[i] = regulae.FormPair{Src: lp.src, Tgt: lp.tgt}
	}
	cognateSets := regulae.CognateSetsFromPairs(pairs, [2]string{"hanoi-like", "saigon-like"}, "pair")
	multiModel, err := regulae.TrainModel(cognateSets, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}
	trained, ok := multiModel.PairwiseModel("hanoi-like", "saigon-like")
	if !ok {
		fmt.Fprintln(os.Stderr, "no pairwise model for hanoi-like/saigon-like")
		os.Exit(1)
	}

	fmt.Println()
	fmt.Println(regulae.FormatModel(trained, regulae.DefaultFormatModelOptions()))

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
