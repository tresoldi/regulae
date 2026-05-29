// Command tone_3way_synthetic trains a multi-lect model on 80 three-way
// synthetic tonogenesis cognate sets (proto + daughter_a + daughter_b) and
// prints a report. Both daughters undergo independent tonogenesis conditioned
// on proto-initial voicing, with different target tones.
//
// Tone notation: trailing digits stripped and attached to the last vowel
// (vowel set: aeiou) or last segment if no vowel found.
//
// Run with:
//
//	cd experiments/tone_3way_synthetic && go run .
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

func loadCorpus(path string) ([]regulae.CognateSet, error) {
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
	var out []regulae.CognateSet
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) < 4 {
			continue
		}
		gloss, proto, a, b := parts[0], parts[1], parts[2], parts[3]
		cs := regulae.CognateSet{
			CognateID: gloss,
			Forms: map[string]regulae.Form{
				"proto": {
					LectID:   "proto",
					Segments: parseTonedForm(proto),
				},
				"daughter_a": {
					LectID:   "daughter_a",
					Segments: parseTonedForm(a),
				},
				"daughter_b": {
					LectID:   "daughter_b",
					Segments: parseTonedForm(b),
				},
			},
			FormsOrder: []string{"proto", "daughter_a", "daughter_b"},
			Confidence: 1.0,
		}
		out = append(out, cs)
	}
	return out, sc.Err()
}

func main() {
	path := "cognates.tsv"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	corpus, err := loadCorpus(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load error:", err)
		os.Exit(1)
	}

	fmt.Printf("Loaded %d 3-way synthetic tonogenesis sets.\n", len(corpus))
	fmt.Println()
	fmt.Println("Expected: 2 daughters each undergo independent tonogenesis")
	fmt.Println("          on proto-initial-voicing, with different target tones.")
	fmt.Println()

	model, err := regulae.TrainModel(corpus, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}

	fmt.Println(regulae.FormatMultiLectModel(model, 10, 10))

	fmt.Println()
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Multi-lect cross-dimensional rules: %d\n", len(model.CrossDimensionalTable.Entries))
	fmt.Println(strings.Repeat("=", 60))
	for i, rule := range model.CrossDimensionalTable.Entries {
		fmt.Printf("#%d: %s[%s@%s] → %s[%s=%s@%+d]  count=%.0f/%.0f  conf=%.2f\n",
			i,
			rule.SrcLect,
			rule.SrcFeature.Feature,
			rule.SrcPosition,
			rule.TgtLect,
			rule.TgtDimension,
			rule.TgtValue,
			rule.TgtPositionOffset,
			rule.Count,
			rule.SrcCount,
			rule.Confidence,
		)
	}
}
