// Command mandarin_historical trains a learned sound-correspondence model on
// 40 Middle Chinese → Mandarin cognate pairs, illustrating tonogenesis via
// onset-voicing redistribution.
//
// Run with:
//
//	cd experiments/mandarin_historical && go run .
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode/utf8"

	regulae "github.com/tresoldi/regulae"
)

// Multi-character graphemes that appear in this corpus. Longer first.
var multiGraphemes = []string{
	"tsʰ", "tʃʰ", "tɕʰ",
	"tʰ", "pʰ", "kʰ",
	"ɻɻ",
	"tɕ", "dʑ", "tʃ", "dʒ", "ts", "dz",
	"aː", "eː", "iː", "oː", "uː",
}

// vowelChars are the characters that identify a vowel-like segment.
const vowelChars = "aeiouɑɒɔɛɪɨɯʊʉyøəɚɻɿ"

func isVowelLike(grapheme string) bool {
	for _, ch := range grapheme {
		for _, v := range vowelChars {
			if ch == v {
				return true
			}
		}
	}
	return false
}

// parseSegments splits an IPA string into Segment values, handling the
// multi-character graphemes listed above.
func parseSegments(ipa string) []regulae.Segment {
	var segs []regulae.Segment
	runes := []rune(ipa)
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
			_, size := utf8.DecodeRuneInString(string(runes[i:]))
			_ = size
			segs = append(segs, regulae.Segment{Grapheme: string(runes[i])})
			i++
		}
	}
	return segs
}

// parseFormWithTone parses an IPA string and attaches tone to the first
// vowel-like segment (the nucleus). If no vowel is found the tone is attached
// to the last segment. A tone of "-" leaves all segments untoned.
func parseFormWithTone(ipa, tone string) []regulae.Segment {
	bare := parseSegments(ipa)
	if tone == "-" || tone == "" {
		return bare
	}
	applied := false
	out := make([]regulae.Segment, len(bare))
	copy(out, bare)
	for i, seg := range out {
		if !applied && isVowelLike(seg.Grapheme) {
			out[i].Tone = tone
			applied = true
			break
		}
	}
	if !applied && len(out) > 0 {
		out[len(out)-1].Tone = tone
	}
	return out
}

type cognate struct {
	gloss string
	src   regulae.Form
	tgt   regulae.Form
}

func loadCorpus(path string) ([]cognate, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	// Read and validate header.
	if !sc.Scan() {
		return nil, fmt.Errorf("empty file: %s", path)
	}
	header := strings.TrimSpace(sc.Text())
	want := "gloss\tmiddle_chinese\tmc_tone\tmandarin\tmd_tone"
	if header != want {
		return nil, fmt.Errorf("unexpected header: %q", header)
	}

	var out []cognate
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) != 5 {
			continue
		}
		gloss, mc, mcTone, md, mdTone := parts[0], parts[1], parts[2], parts[3], parts[4]
		src := regulae.Form{
			LectID:   "middle_chinese",
			Segments: parseFormWithTone(mc, mcTone),
		}
		tgt := regulae.Form{
			LectID:   "mandarin",
			Segments: parseFormWithTone(md, mdTone),
		}
		out = append(out, cognate{gloss: gloss, src: src, tgt: tgt})
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
	fmt.Printf("Loaded %d Middle Chinese → Mandarin cognate pairs.\n", len(labeled))

	pairs := make([]regulae.FormPair, len(labeled))
	for i, c := range labeled {
		pairs[i] = regulae.FormPair{Src: c.src, Tgt: c.tgt}
	}

	corpus := regulae.CognateSetsFromPairs(pairs, [2]string{"middle_chinese", "mandarin"}, "mc_md")
	multi, err := regulae.TrainModel(corpus, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}

	trained, ok := multi.PairwiseModel("middle_chinese", "mandarin")
	if !ok {
		fmt.Fprintln(os.Stderr, "no pairwise model for middle_chinese/mandarin")
		os.Exit(1)
	}

	opts := regulae.DefaultFormatModelOptions()
	opts.TopSegments = 20
	fmt.Println(regulae.FormatModel(trained, opts))

	fmt.Println()
	fmt.Println("Cross-dimensional rules (tonogenesis):")
	for _, rule := range trained.CrossDimensionalTable.Entries {
		srcLabel := fmt.Sprintf("%s=%s@%s", rule.SrcFeature.Feature, rule.SrcFeature.Value, rule.SrcPosition)
		fmt.Printf("  %s -> %s=%s@+%d  count=%.0f/%.0f  conf=%.2f\n",
			srcLabel, rule.TgtDimension, rule.TgtValue, rule.TgtPositionOffset,
			rule.Count, rule.SrcCount, rule.Confidence)
	}
}
