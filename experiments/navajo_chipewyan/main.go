// Command navajo_chipewyan trains a learned sound-correspondence model on the
// Navajo → Chipewyan (Dëne Sųłiné) Athabaskan cognate corpus in cognates.tsv
// and prints a human-readable report. The corpus exercises the classic
// interdental split (Proto-Athabaskan *ts ↔ Chipewyan tθ/θ), glottal
// preservation, and the rich Athabaskan ejective/aspirated consonant inventory.
//
// Run with:
//
//	cd experiments/navajo_chipewyan && go run .
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"

	regulae "github.com/tresoldi/regulae"
)

// Multi-character graphemes for Athabaskan; longer prefixes first.
var multiChar = []string{
	"tθʼ", "tʃʼ", "tsʼ", "tɬʼ",
	"tθ", "tʃʰ", "tɬʰ", "tsʰ",
	"tʼ", "pʼ", "kʼ",
	"tʰ", "pʰ", "kʰ",
	"dʒ", "dɮ", "dz",
	"aː", "eː", "iː", "oː", "uː",
	"ãː", "ẽː", "ĩː", "õː", "ũː",
	"ã", "ẽ", "ĩ", "õ", "ũ",
}

const (
	combAcute = '́' // combining acute accent → tone H
	combGrave = '̀' // combining grave accent → tone L
	combTilde = '̃' // combining tilde → nasalisation (attaches to grapheme)
)

// normalise applies NFD decomposition so tone marks are separate code points,
// then recomposes base + combining tilde (nasalisation) into precomposed
// characters while leaving acute/grave (tone marks) decomposed for extraction.
func normalise(raw string) string {
	decomposed := norm.NFD.String(raw)
	runes := []rune(decomposed)
	var result []rune
	i := 0
	for i < len(runes) {
		ch := runes[i]
		if i+1 < len(runes) && runes[i+1] == combTilde {
			// base + combining tilde → precomposed nasalised form.
			combined := norm.NFC.String(string([]rune{ch, combTilde}))
			result = append(result, []rune(combined)...)
			i += 2
		} else {
			result = append(result, ch)
			i++
		}
	}
	return string(result)
}

// parseSegments tokenises an IPA string into regulae.Segments.
// Multi-char graphemes (affricates, aspirated stops, ejectives, long vowels,
// nasalised vowels) are consumed as single units. Combining acute/grave
// accents are extracted as tone values "H"/"L"; a trailing ː is appended to
// the grapheme.
func parseSegments(ipa string) []regulae.Segment {
	ipa = normalise(ipa)
	runes := []rune(ipa)
	var segs []regulae.Segment
	i := 0
	for i < len(runes) {
		// Match a multi-char prefix or consume a single rune.
		emitted := ""
		for _, cluster := range multiChar {
			cr := []rune(cluster)
			if i+len(cr) <= len(runes) && string(runes[i:i+len(cr)]) == cluster {
				emitted = cluster
				i += len(cr)
				break
			}
		}
		if emitted == "" {
			emitted = string(runes[i])
			i++
		}
		// Absorb trailing diacritics: tone marks, nasalisation tilde, length.
		tone := ""
		for i < len(runes) {
			mark := runes[i]
			switch mark {
			case combAcute:
				tone = "H"
				i++
			case combGrave:
				tone = "L"
				i++
			case 'ː':
				emitted = emitted + "ː"
				i++
			case combTilde:
				// Remaining combining tilde (shouldn't appear after normalise
				// but handle defensively).
				emitted = emitted + string(combTilde)
				i++
			default:
				if unicode.Is(unicode.Mn, mark) {
					// Any other combining mark — skip silently.
					i++
				} else {
					goto doneAbsorb
				}
			}
		}
	doneAbsorb:
		seg := regulae.Segment{Grapheme: emitted}
		if tone != "" {
			seg.Tone = tone
		}
		segs = append(segs, seg)
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
	// header: gloss navajo chipewyan
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
		gloss, nv, ch := parts[0], parts[1], parts[2]
		out = append(out, labeledPair{
			gloss: gloss,
			src:   regulae.Form{LectID: "navajo", Segments: parseSegments(nv)},
			tgt:   regulae.Form{LectID: "chipewyan", Segments: parseSegments(ch)},
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

	fmt.Printf("Loaded %d Navajo → Chipewyan cognate pairs.\n", len(labeled))

	cognateSets := regulae.CognateSetsFromPairs(pairs, [2]string{"navajo", "chipewyan"}, "navajo_chipewyan")
	multiModel, err := regulae.TrainModel(cognateSets, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}
	trained, ok := multiModel.PairwiseModel("navajo", "chipewyan")
	if !ok {
		fmt.Fprintln(os.Stderr, "pairwise model not found")
		os.Exit(1)
	}

	opts := regulae.DefaultFormatModelOptions()
	opts.TopSegments = 25
	fmt.Println(regulae.FormatModel(trained, opts))
}
