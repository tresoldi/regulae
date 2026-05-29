// Command contaminated_cognates_synthetic trains a multi-lect model on 25
// synthetic pairs (20 clean at confidence=1.0, 5 contaminated at
// confidence=0.0) and shows that the confidence weighting correctly suppresses
// the noise. A control run with all pairs at confidence=1.0 confirms the
// difference. This is a report experiment, not a pass/fail test.
//
// Run with:
//
//	cd experiments/contaminated_cognates_synthetic && go run .
package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"

	regulae "github.com/tresoldi/regulae"
)

func parseSegments(ipa string) []regulae.Segment {
	runes := []rune(ipa)
	segs := make([]regulae.Segment, 0, len(runes))
	for _, r := range runes {
		segs = append(segs, regulae.Segment{Grapheme: string(r)})
	}
	return segs
}

// loadCorpus loads the TSV with confidence weights.
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
	// parse confidence as float
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
		gloss, proto, derived := parts[0], parts[1], parts[2]
		conf := 0.0
		fmt.Sscanf(parts[3], "%f", &conf)
		src := regulae.Form{LectID: "proto", Segments: parseSegments(proto)}
		tgt := regulae.Form{LectID: "derived", Segments: parseSegments(derived)}
		out = append(out, regulae.CognateSet{
			CognateID:  gloss,
			Forms:      map[string]regulae.Form{"proto": src, "derived": tgt},
			FormsOrder: []string{"proto", "derived"},
			Confidence: conf,
		})
	}
	return out, sc.Err()
}

// allEqualConfidence returns a copy of corpus with every confidence set to 1.0.
func allEqualConfidence(corpus []regulae.CognateSet) []regulae.CognateSet {
	out := make([]regulae.CognateSet, len(corpus))
	for i, cs := range corpus {
		cs.Confidence = 1.0
		out[i] = cs
	}
	return out
}

// summarise prints p→f-style counts for the proto→derived pair.
func summarise(multiModel *regulae.MultiLectModel) {
	pair, ok := multiModel.PairwiseModel("proto", "derived")
	if !ok {
		fmt.Println("  (no proto/derived pairwise model)")
		return
	}
	type entry struct {
		cc    regulae.ConditionedCorrespondence
		count float64
	}
	var items []entry
	for k, count := range pair.SegmentTable.Counts {
		cc := pair.SegmentTable.Corr[k]
		if cc.Src != "p" || cc.Context.ConstraintCount() != 0 {
			continue
		}
		items = append(items, entry{cc, count})
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].count > items[j].count
	})
	fmt.Println("Segment table (p source):")
	for _, e := range items {
		fmt.Printf("  p -> %s: count=%.1f\n", e.cc.Tgt, e.count)
	}
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
	control := allEqualConfidence(corpus)

	contaminated := 0
	for _, cs := range corpus {
		if cs.Confidence == 0.0 {
			contaminated++
		}
	}
	fmt.Printf("Loaded %d cognates (%d contaminated at confidence=0).\n", len(corpus), contaminated)
	fmt.Println()

	fmt.Println("=== With confidence weighting (bad pairs at 0.0) ===")
	m, err := regulae.TrainModel(corpus, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}
	summarise(m)
	fmt.Println()

	fmt.Println("=== Control: all pairs at confidence=1.0 ===")
	mCtrl, err := regulae.TrainModel(control, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error (control):", err)
		os.Exit(1)
	}
	summarise(mCtrl)
}
