// Command length_conditioned_synthetic trains a multi-lect model on 30
// synthetic VCV pairs embodying a length-conditioned rule (b → β after a long
// vowel, b → b after a short vowel) and prints the recovered correspondences
// together with any length-conditioned segment entries.
//
// Long vowels (aː, eː, iː, oː, uː) are treated as single multi-character
// graphemes matching the Python parse.
//
// Run with:
//
//	cd experiments/length_conditioned_synthetic && go run .
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	regulae "github.com/tresoldi/regulae"
)

// multiGraphemes lists the multi-character graphemes used in this corpus.
var multiGraphemes = []string{"aː", "eː", "iː", "oː", "uː"}

func parseForm(ipa string) []regulae.Segment {
	runes := []rune(ipa)
	var segs []regulae.Segment
	i := 0
	for i < len(runes) {
		matched := false
		for _, cluster := range multiGraphemes {
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

	fmt.Printf("Loaded %d length-conditioned synthetic pairs.\n", len(labeled))
	fmt.Println("Rule: b → β / [+long] _")
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
	fmt.Println("Length-conditioned segment entries:")
	for k, count := range trained.SegmentTable.Counts {
		cc := trained.SegmentTable.Corr[k]
		ctx := cc.Context
		isLengthConditioned := false
		for _, fc := range ctx.Preceding {
			if fc.Feature == "long" {
				isLengthConditioned = true
				break
			}
		}
		if !isLengthConditioned {
			for _, fc := range ctx.Following {
				if fc.Feature == "long" {
					isLengthConditioned = true
					break
				}
			}
		}
		if !isLengthConditioned {
			continue
		}
		var parts []string
		if len(ctx.Preceding) > 0 {
			var ps []string
			for _, fc := range ctx.Preceding {
				ps = append(ps, fmt.Sprintf("%s:%s", fc.Feature, fc.Value))
			}
			parts = append(parts, "prec="+strings.Join(ps, ","))
		}
		if len(ctx.Following) > 0 {
			var fs []string
			for _, fc := range ctx.Following {
				fs = append(fs, fmt.Sprintf("%s:%s", fc.Feature, fc.Value))
			}
			parts = append(parts, "foll="+strings.Join(fs, ","))
		}
		fmt.Printf("  %s -> %s: count=%.1f  [%s]\n", cc.Src, cc.Tgt, count, strings.Join(parts, " | "))
	}
}
