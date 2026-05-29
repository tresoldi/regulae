// Command harmony_synthetic trains a multi-lect model on 45 synthetic CVCV
// pairs embodying a progressive vowel-harmony rule (a → o when the preceding
// syllable's vowel is back) and prints the recovered correspondences together
// with any long-range conditioned entries. This is a report experiment, not a
// pass/fail test.
//
// Run with:
//
//	cd experiments/harmony_synthetic && go run .
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	regulae "github.com/tresoldi/regulae"
)

type labeledPair struct {
	gloss string
	src   regulae.Form
	tgt   regulae.Form
}

func parseForm(ipa string) []regulae.Segment {
	runes := []rune(ipa)
	segs := make([]regulae.Segment, 0, len(runes))
	for _, r := range runes {
		segs = append(segs, regulae.Segment{Grapheme: string(r)})
	}
	return segs
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

	fmt.Printf("Loaded %d synthetic harmony pairs.\n", len(labeled))
	fmt.Println()
	fmt.Println("Rule: a → o when previous syllable's vowel is back (u, o).")
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
	fmt.Println("Long-range conditioned entries in detail:")
	for k, count := range trained.SegmentTable.Counts {
		cc := trained.SegmentTable.Corr[k]
		ctx := cc.Context
		lrTotal := len(ctx.PrecedingAtDistance) +
			len(ctx.FollowingAtDistance) +
			len(ctx.SomewherePreceding) +
			len(ctx.SomewhereFollowing) +
			len(ctx.SameSyllable) +
			len(ctx.NextSyllable) +
			len(ctx.PreviousSyllable)
		if lrTotal == 0 {
			continue
		}
		var parts []string
		for _, dc := range ctx.PrecedingAtDistance {
			parts = append(parts, fmt.Sprintf("pre@%d=%s:%s", dc.Offset, dc.Constraint.Feature, dc.Constraint.Value))
		}
		for _, dc := range ctx.FollowingAtDistance {
			parts = append(parts, fmt.Sprintf("fol@%d=%s:%s", dc.Offset, dc.Constraint.Feature, dc.Constraint.Value))
		}
		for _, fc := range ctx.SomewherePreceding {
			parts = append(parts, fmt.Sprintf("s_pre=%s:%s", fc.Feature, fc.Value))
		}
		for _, fc := range ctx.SomewhereFollowing {
			parts = append(parts, fmt.Sprintf("s_fol=%s:%s", fc.Feature, fc.Value))
		}
		for _, fc := range ctx.SameSyllable {
			parts = append(parts, fmt.Sprintf("same_syl=%s:%s", fc.Feature, fc.Value))
		}
		for _, fc := range ctx.NextSyllable {
			parts = append(parts, fmt.Sprintf("next_syl=%s:%s", fc.Feature, fc.Value))
		}
		for _, fc := range ctx.PreviousSyllable {
			parts = append(parts, fmt.Sprintf("prev_syl=%s:%s", fc.Feature, fc.Value))
		}
		fmt.Printf("  %s -> %s: %.0f  [%s]\n", cc.Src, cc.Tgt, count, strings.Join(parts, ", "))
	}
}
