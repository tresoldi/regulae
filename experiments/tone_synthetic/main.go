// Command tone_synthetic trains a learned sound-correspondence model on 30
// hand-constructed tonal pairs (src tone 55 or 11 → tgt tone 33) and prints
// a report showing that the framework recovers the two tonal correspondences.
//
// Tone notation: a run of trailing digits (e.g. "pa55") is stripped and
// attached to the last vowel in the segmental string.
//
// Run with:
//
//	cd experiments/tone_synthetic && go run .
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
	'ɛ': true, 'ɔ': true, 'ə': true,
}

// parseTonedForm splits an IPA string like "pa55" into segments, attaching
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

	chars := make([]string, len(segmental))
	for j, r := range segmental {
		chars[j] = string(r)
	}

	segs := make([]regulae.Segment, len(chars))
	for j, c := range chars {
		segs[j] = regulae.Segment{Grapheme: c}
	}

	if tone != "" {
		// Attach to the last vowel; fall back to the last segment.
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
			src:   regulae.Form{LectID: "src", Segments: parseTonedForm(parts[1])},
			tgt:   regulae.Form{LectID: "tgt", Segments: parseTonedForm(parts[2])},
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

	fmt.Printf("Loaded %d synthetic toned pairs.\n", len(labeled))
	fmt.Println()

	// Show sample parses.
	fmt.Println("Sample parses:")
	limit := 3
	if len(labeled) < limit {
		limit = len(labeled)
	}
	for _, lp := range labeled[:limit] {
		var srcDesc, tgtDesc []string
		for _, s := range lp.src.Segments {
			if s.Tone != "" {
				srcDesc = append(srcDesc, fmt.Sprintf("%s[T=%s]", s.Grapheme, s.Tone))
			} else {
				srcDesc = append(srcDesc, s.Grapheme)
			}
		}
		for _, s := range lp.tgt.Segments {
			if s.Tone != "" {
				tgtDesc = append(tgtDesc, fmt.Sprintf("%s[T=%s]", s.Grapheme, s.Tone))
			} else {
				tgtDesc = append(tgtDesc, s.Grapheme)
			}
		}
		fmt.Printf("  [%s] %s  ~  %s\n", lp.gloss, strings.Join(srcDesc, " "), strings.Join(tgtDesc, " "))
	}
	fmt.Println()

	fmt.Println("Training learned model...")
	pairs := make([]regulae.FormPair, len(labeled))
	for i, lp := range labeled {
		pairs[i] = regulae.FormPair{Src: lp.src, Tgt: lp.tgt}
	}
	cognateSets := regulae.CognateSetsFromPairs(pairs, [2]string{"src", "tgt"}, "pair")
	multiModel, err := regulae.TrainModel(cognateSets, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}
	trained, ok := multiModel.PairwiseModel("src", "tgt")
	if !ok {
		fmt.Fprintln(os.Stderr, "no pairwise model for src/tgt")
		os.Exit(1)
	}
	fmt.Println()

	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Trained model summary")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println(regulae.FormatModel(trained, regulae.DefaultFormatModelOptions()))

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
	fmt.Printf("Prior-only total cost: %.2f\n", priorTotal)
	fmt.Printf("Learned-model total cost: %.2f\n", learnedTotal)
	fmt.Printf("Reduction:     %.2f\n", priorTotal-learnedTotal)

	fmt.Println()
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Sample alignments")
	fmt.Println(strings.Repeat("=", 70))
	limit = 4
	if len(labeled) < limit {
		limit = len(labeled)
	}
	for _, lp := range labeled[:limit] {
		fmt.Println()
		fmt.Printf("[%s]\n", lp.gloss)
		al, _ := regulae.AlignForms(lp.src, lp.tgt, "descriptive", 0, trained)
		fmt.Println(regulae.FormatAlignment(al, "descriptive", true, false))
	}
}
