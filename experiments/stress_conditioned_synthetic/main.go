// Command stress_conditioned_synthetic trains a multi-lect model on 36
// synthetic CVCV pairs embodying stress-conditioned vowel lowering
// (stressed e → ɛ, stressed o → ɔ; unstressed vowels preserved) and prints
// the recovered correspondences together with stress-conditioned segment
// entries.
//
// Parsing: IPA primary-stress mark ˈ marks the onset of a stressed syllable;
// the stress is attached to the next vowel nucleus encountered. Syllable-
// boundary dashes (-) are dropped from the segmentation.
//
// Run with:
//
//	cd experiments/stress_conditioned_synthetic && go run .
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode/utf8"

	regulae "github.com/tresoldi/regulae"
)

var vowels = map[rune]bool{
	'a': true, 'e': true, 'i': true, 'o': true, 'u': true,
	'ɛ': true, 'ɔ': true, 'ɐ': true, 'ə': true, 'ɨ': true,
	'ɯ': true, 'æ': true, 'ø': true, 'y': true, 'ɪ': true,
	'ʊ': true, 'ɑ': true, 'ɒ': true,
}

func parseForm(ipa string) []regulae.Segment {
	var segs []regulae.Segment
	pendingStress := ""
	i := 0
	for i < len(ipa) {
		r, size := utf8.DecodeRuneInString(ipa[i:])
		i += size
		if r == 'ˈ' {
			pendingStress = "+"
			continue
		}
		if r == '-' {
			continue
		}
		if vowels[r] && pendingStress != "" {
			segs = append(segs, regulae.Segment{Grapheme: string(r), Stress: pendingStress})
			pendingStress = ""
		} else {
			segs = append(segs, regulae.Segment{Grapheme: string(r)})
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
			src:   regulae.Form{LectID: "proto", Segments: parseForm(parts[1])},
			tgt:   regulae.Form{LectID: "derived", Segments: parseForm(parts[2])},
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

	fmt.Printf("Loaded %d stress-conditioned synthetic pairs.\n", len(labeled))
	fmt.Println("Rules: stressed e → ɛ, stressed o → ɔ; unstressed preserved.")
	fmt.Println()

	pairs := make([]regulae.FormPair, len(labeled))
	for i, lp := range labeled {
		pairs[i] = regulae.FormPair{Src: lp.src, Tgt: lp.tgt}
	}
	cognateSets := regulae.CognateSetsFromPairs(pairs, [2]string{"proto", "derived"}, "pair")
	multiModel, err := regulae.TrainModel(cognateSets, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}

	trained, ok := multiModel.PairwiseModel("proto", "derived")
	if !ok {
		fmt.Fprintln(os.Stderr, "no pairwise model for proto/derived")
		os.Exit(1)
	}

	opts := regulae.DefaultFormatModelOptions()
	opts.TopSegments = 20
	fmt.Println(regulae.FormatModel(trained, opts))

	fmt.Println()
	fmt.Println("Stress-conditioned segment entries:")
	for k, count := range trained.SegmentTable.Counts {
		cc := trained.SegmentTable.Corr[k]
		ctx := cc.Context
		hasStress := false
		for _, fc := range ctx.SelfStress {
			if fc.Feature == "stress" {
				hasStress = true
				break
			}
		}
		if !hasStress {
			for _, fc := range ctx.PrecedingStress {
				if fc.Feature == "stress" {
					hasStress = true
					break
				}
			}
		}
		if !hasStress {
			for _, fc := range ctx.FollowingStress {
				if fc.Feature == "stress" {
					hasStress = true
					break
				}
			}
		}
		if !hasStress {
			continue
		}
		var parts []string
		if len(ctx.SelfStress) > 0 {
			var ss []string
			for _, fc := range ctx.SelfStress {
				ss = append(ss, fmt.Sprintf("%s:%s", fc.Feature, fc.Value))
			}
			parts = append(parts, "self="+strings.Join(ss, ","))
		}
		if len(ctx.PrecedingStress) > 0 {
			var ps []string
			for _, fc := range ctx.PrecedingStress {
				ps = append(ps, fmt.Sprintf("%s:%s", fc.Feature, fc.Value))
			}
			parts = append(parts, "prec_s="+strings.Join(ps, ","))
		}
		if len(ctx.FollowingStress) > 0 {
			var fs []string
			for _, fc := range ctx.FollowingStress {
				fs = append(fs, fmt.Sprintf("%s:%s", fc.Feature, fc.Value))
			}
			parts = append(parts, "fol_s="+strings.Join(fs, ","))
		}
		fmt.Printf("  %s -> %s: count=%.1f  [%s]\n", cc.Src, cc.Tgt, count, strings.Join(parts, " | "))
	}
}
