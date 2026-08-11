// Command goref dumps compact summaries of the Go reference implementation so
// the C port can be compared against it. It is a parity harness, not a product
// entry point, and is removed with the Go tree at M9.
//
// The summary format is shared with the C CLI's `train --summary`; the two are
// diffed by scripts/parity.sh.
package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/tresoldi/regulae"
)

func parseForm(lect, spec string) regulae.Form {
	var segs []regulae.Segment
	var morph []int
	for _, tok := range strings.Fields(spec) {
		if tok == "+" {
			if len(segs) > 0 {
				morph = append(morph, len(segs))
			}
			continue
		}
		segs = append(segs, regulae.Segment{Grapheme: tok})
	}
	return regulae.Form{LectID: lect, Segments: segs, MorphemeBreaks: morph}
}

func num(v float64) string {
	return fmt.Sprintf("%.6f", v)
}

func constraints(cs []regulae.FeatureConstraint) string {
	parts := make([]string, 0, len(cs))
	for _, c := range cs {
		parts = append(parts, c.Feature+":"+c.Value)
	}
	sort.Strings(parts)
	return strings.Join(parts, ";")
}

// contextKey renders a Context as a canonical string. Slot order is fixed and
// constraint lists are sorted, so two implementations that agree on content
// produce byte-identical keys.
func contextKey(c regulae.Context) string {
	var parts []string
	add := func(name, value string) {
		if value != "" {
			parts = append(parts, name+"="+value)
		}
	}
	preAt := make([]string, 0, len(c.PrecedingAtDistance))
	for _, d := range c.PrecedingAtDistance {
		preAt = append(preAt, fmt.Sprintf("%d@%s:%s", d.Offset, d.Constraint.Feature, d.Constraint.Value))
	}
	sort.Strings(preAt)
	folAt := make([]string, 0, len(c.FollowingAtDistance))
	for _, d := range c.FollowingAtDistance {
		folAt = append(folAt, fmt.Sprintf("%d@%s:%s", d.Offset, d.Constraint.Feature, d.Constraint.Value))
	}
	sort.Strings(folAt)

	add("pos", c.Position)
	add("morph", c.Morphological)
	add("pre", constraints(c.Preceding))
	add("fol", constraints(c.Following))
	add("preat", strings.Join(preAt, ";"))
	add("folat", strings.Join(folAt, ";"))
	add("swpre", constraints(c.SomewherePreceding))
	add("swfol", constraints(c.SomewhereFollowing))
	add("samesyl", constraints(c.SameSyllable))
	add("nextsyl", constraints(c.NextSyllable))
	add("prevsyl", constraints(c.PreviousSyllable))
	add("selfstress", constraints(c.SelfStress))
	add("prestress", constraints(c.PrecedingStress))
	add("folstress", constraints(c.FollowingStress))
	if len(parts) == 0 {
		return "-"
	}
	return strings.Join(parts, ",")
}

func sortedLects(segments map[string]string) []string {
	lects := make([]string, 0, len(segments))
	for l := range segments {
		lects = append(lects, l)
	}
	sort.Strings(lects)
	return lects
}

func segmentsKey(segments map[string]string) string {
	parts := []string{}
	for _, l := range sortedLects(segments) {
		parts = append(parts, l+":"+segments[l])
	}
	return strings.Join(parts, "|")
}

func contextsKey(segments map[string]string, contexts map[string]regulae.Context) string {
	parts := []string{}
	for _, l := range sortedLects(segments) {
		parts = append(parts, l+"="+contextKey(contexts[l]))
	}
	return strings.Join(parts, "|")
}

// loadCorpus reads the corpus path given as the second argument through the Go
// TSV loader, so a parity run compares loader behaviour as well as training.
// hasToneColumn reports whether a TSV declares a "tone" column, which decides
// between the frozen loader and the local tone-aware one.
func hasToneColumn(path string) bool {
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	lines := strings.SplitN(string(data), "\n", 2)
	if len(lines) == 0 {
		return false
	}
	for _, name := range strings.Split(strings.TrimRight(lines[0], "\r"), "\t") {
		if strings.TrimSpace(name) == "tone" {
			return true
		}
	}
	return false
}

// loadTonedTSV mirrors the C long-format loader, including its tone column:
// per-segment tones parallel to the segments cell, with "-" leaving a segment
// untoned. Built through regulae's public types only.
func loadTonedTSV(path string) ([]regulae.CognateSet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n")
	if len(lines) == 0 {
		return nil, fmt.Errorf("empty corpus")
	}
	index := map[string]int{}
	for i, name := range strings.Split(lines[0], "\t") {
		index[strings.TrimSpace(name)] = i
	}
	cognateCol, okC := index["cognate_id"]
	lectCol, okL := index["lect_id"]
	segCol, okS := index["segments"]
	if !okC || !okL || !okS {
		return nil, fmt.Errorf("missing a required column")
	}
	toneCol, hasTone := index["tone"]
	confCol, hasConf := index["confidence"]

	var order []string
	byID := map[string]*regulae.CognateSet{}
	for _, line := range lines[1:] {
		if strings.TrimSpace(line) == "" {
			continue
		}
		cols := strings.Split(line, "\t")
		get := func(i int) string {
			if i < 0 || i >= len(cols) {
				return ""
			}
			return strings.TrimSpace(cols[i])
		}
		id, lect := get(cognateCol), get(lectCol)
		if id == "" || lect == "" {
			continue
		}
		var segs []regulae.Segment
		for _, tok := range strings.Fields(get(segCol)) {
			if tok == "-" {
				continue
			}
			segs = append(segs, regulae.Segment{Grapheme: tok})
		}
		if len(segs) == 0 {
			continue
		}
		if hasTone {
			tones := strings.Fields(get(toneCol))
			if len(tones) > 0 && len(tones) != len(segs) {
				return nil, fmt.Errorf("%s/%s: %d tones for %d segments", id, lect, len(tones), len(segs))
			}
			for i, tone := range tones {
				if tone != "-" {
					segs[i].Tone = tone
				}
			}
		}
		set, seen := byID[id]
		if !seen {
			set = &regulae.CognateSet{
				CognateID:  id,
				Forms:      map[string]regulae.Form{},
				Confidence: 1.0,
			}
			byID[id] = set
			order = append(order, id)
		}
		if _, dup := set.Forms[lect]; dup {
			return nil, fmt.Errorf("%s: duplicate lect %s", id, lect)
		}
		set.Forms[lect] = regulae.Form{LectID: lect, Segments: segs}
		set.FormsOrder = append(set.FormsOrder, lect)
		if hasConf {
			if raw := get(confCol); raw != "" {
				var value float64
				if _, err := fmt.Sscanf(raw, "%g", &value); err == nil && value < set.Confidence {
					set.Confidence = value
				}
			}
		}
	}
	out := make([]regulae.CognateSet, 0, len(order))
	for _, id := range order {
		out = append(out, *byID[id])
	}
	return out, nil
}

func loadCorpus() []regulae.CognateSet {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: goref <summary|outliers> <corpus.tsv>")
		os.Exit(2)
	}
	path := os.Args[2]
	var corpus []regulae.CognateSet
	var err error
	// A .csv corpus is arcaverborum-format, which is where morpheme boundaries
	// come from; .tsv goes through the generic loader.
	if strings.HasSuffix(strings.ToLower(path), ".csv") {
		corpus, err = regulae.LoadArcaverborum(path, regulae.ArcaverborumLoadOptions{})
	} else if hasToneColumn(path) {
		// The frozen Go loader has no tone column, and adding one would be
		// adding behaviour to the reference. Reading it here instead keeps the
		// library frozen while letting a tone-bearing corpus reach the parity
		// harness, which is the only way to exercise cross-dimensional
		// discovery against the reference at all.
		corpus, err = loadTonedTSV(path)
	} else {
		corpus, err = regulae.LoadCognatesFromTSV(path, regulae.TSVLoadOptions{ConfidenceCol: "confidence"})
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	return corpus
}

func readCorpus(in *bufio.Scanner) []regulae.CognateSet {
	type entry struct {
		forms      map[string]regulae.Form
		order      []string
		confidence float64
	}
	byID := map[string]*entry{}
	var order []string
	for in.Scan() {
		line := in.Text()
		if strings.TrimSpace(line) == "" || strings.HasPrefix(line, "#") {
			continue
		}
		cols := strings.Split(line, "\t")
		if len(cols) < 3 {
			continue
		}
		cogID := strings.TrimSpace(cols[0])
		lectID := strings.TrimSpace(cols[1])
		form := parseForm(lectID, cols[2])
		if len(form.Segments) == 0 {
			continue
		}
		confidence := 1.0
		if len(cols) > 3 && strings.TrimSpace(cols[3]) != "" {
			fmt.Sscanf(strings.TrimSpace(cols[3]), "%g", &confidence)
		}
		e := byID[cogID]
		if e == nil {
			e = &entry{forms: map[string]regulae.Form{}, confidence: confidence}
			byID[cogID] = e
			order = append(order, cogID)
		}
		if confidence < e.confidence {
			e.confidence = confidence
		}
		if _, dup := e.forms[lectID]; dup {
			continue
		}
		e.forms[lectID] = form
		e.order = append(e.order, lectID)
	}
	out := make([]regulae.CognateSet, 0, len(order))
	for _, id := range order {
		e := byID[id]
		out = append(out, regulae.CognateSet{
			CognateID:  id,
			Forms:      e.forms,
			FormsOrder: e.order,
			Confidence: e.confidence,
		})
	}
	return out
}

func dumpModel(model *regulae.MultiLectModel) {
	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()

	fmt.Fprintf(w, "LECTS\t%s\n", strings.Join(model.LectIDs, " "))
	for _, class := range model.UnconditionedClasses {
		fmt.Fprintf(w, "UNCOND\t%d\t%s\t%s\t%s\t%s\n",
			class.ClassID,
			segmentsKey(class.Segments),
			num(class.Count),
			num(class.Confidence),
			strings.Join(class.SupportingCognates, ","))
	}
	for _, class := range model.ConditionedClasses {
		fmt.Fprintf(w, "COND\t%d\t%s\t%s\t%s\t%s\n",
			class.ClassID,
			segmentsKey(class.Segments),
			num(class.Count),
			num(class.Confidence),
			contextsKey(class.Segments, class.Contexts))
	}
	for _, rule := range model.CrossDimensionalTable.Entries {
		// The joint predicate is part of the rule's identity: without it a
		// two-predicate rule prints identically to the single-predicate rule
		// over the same first feature, and parity cannot see the difference.
		src := fmt.Sprintf("%s=%s@%s", rule.SrcFeature.Feature, rule.SrcFeature.Value, rule.SrcPosition)
		if rule.SrcFeature2 != nil && rule.SrcPosition2 != "" {
			src += fmt.Sprintf("&%s=%s@%s", rule.SrcFeature2.Feature, rule.SrcFeature2.Value, rule.SrcPosition2)
		}
		fmt.Fprintf(w, "XDIM\t%s>%s\t%s\t%s=%s@%d\t%s\t%s\t%s\n",
			rule.SrcLect, rule.TgtLect, src,
			rule.TgtDimension, rule.TgtValue, rule.TgtPositionOffset,
			num(rule.Count), num(rule.SrcCount), num(rule.Confidence))
	}
}

func main() {
	mode := "summary"
	if len(os.Args) > 1 {
		mode = os.Args[1]
	}
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 1024*1024), 1024*1024)

	switch mode {
	case "syllables":
		for in.Scan() {
			line := strings.TrimSpace(in.Text())
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			form := parseForm("x", line)
			breaks := regulae.ComputeSyllableBreaks(form, "descriptive")
			parts := make([]string, len(breaks))
			for i, b := range breaks {
				parts[i] = fmt.Sprint(b)
			}
			fmt.Printf("%s\t%s\n", line, strings.Join(parts, ","))
		}
	case "summary":
		corpus := loadCorpus()
		model, err := regulae.TrainModel(corpus, regulae.DefaultTrainOptions())
		if err != nil {
			fmt.Fprintf(os.Stderr, "train: %v\n", err)
			os.Exit(1)
		}
		dumpModel(model)
	case "outliers":
		corpus := loadCorpus()
		model, err := regulae.TrainModel(corpus, regulae.DefaultTrainOptions())
		if err != nil {
			fmt.Fprintf(os.Stderr, "train: %v\n", err)
			os.Exit(1)
		}
		reports, err := regulae.FindCognateOutliers(corpus, model, 0, 0)
		if err != nil {
			fmt.Fprintf(os.Stderr, "outliers: %v\n", err)
			os.Exit(1)
		}
		for _, r := range reports {
			fmt.Printf("OUTLIER\t%s\t%d\t%s\t%s\n", r.CognateID, r.NPairs, num(r.CostPerSegment), num(r.ZScore))
		}
	case "pairwise":
		corpus := loadCorpus()
		model, err := regulae.TrainModel(corpus, regulae.DefaultTrainOptions())
		if err != nil {
			fmt.Fprintf(os.Stderr, "train: %v\n", err)
			os.Exit(1)
		}
		for i := 0; i < len(model.LectIDs); i++ {
			for j := i + 1; j < len(model.LectIDs); j++ {
				a, b := model.LectIDs[i], model.LectIDs[j]
				pm, ok := model.PairwiseModel(a, b)
				if !ok || pm == nil {
					continue
				}
				var keys []string
				for k := range pm.SegmentTable.Counts {
					keys = append(keys, k)
				}
				sort.Strings(keys)
				for _, k := range keys {
					corr := pm.SegmentTable.Corr[k]
					fmt.Printf("SEG\t%s>%s\t%s\t%s\t%s\t%s\n",
						a, b, corr.Src, corr.Tgt, contextKey(corr.Context), num(pm.SegmentTable.Counts[k]))
				}
				var chunkKeys []string
				for k := range pm.ChunkTable.Entries {
					chunkKeys = append(chunkKeys, k)
				}
				sort.Strings(chunkKeys)
				for _, k := range chunkKeys {
					pair := pm.ChunkTable.Pairs[k]
					var src, tgt string
					for _, s := range pair.Src {
						src += s.Grapheme
					}
					for _, s := range pair.Tgt {
						tgt += s.Grapheme
					}
					fmt.Printf("CHUNK\t%s>%s\t%s\t%s\t%s\t%s\n",
						a, b, src, tgt, num(pm.ChunkTable.Entries[k]), num(pm.ChunkTable.ObservationCounts[k]))
				}
				var toneKeys []string
				toneOf := map[string]regulae.TonalCorrespondence{}
				for k := range pm.TonalTable.Counts {
					key := k.SrcTone + ">" + k.TgtTone
					toneKeys = append(toneKeys, key)
					toneOf[key] = k
				}
				sort.Strings(toneKeys)
				for _, key := range toneKeys {
					fmt.Printf("TONE\t%s>%s\t%s\t%s\n", a, b, key, num(pm.TonalTable.Counts[toneOf[key]]))
				}
				// The per-pair cross-dimensional table is where discovery
				// actually commits; the multi-lect table only shows what
				// survived lifting. Joint rules are visible only here.
				var xdimRows []string
				for _, rule := range pm.CrossDimensionalTable.Entries {
					src := fmt.Sprintf("%s=%s@%s", rule.SrcFeature.Feature, rule.SrcFeature.Value, rule.SrcPosition)
					if rule.SrcFeature2 != nil && rule.SrcPosition2 != "" {
						src += fmt.Sprintf("&%s=%s@%s", rule.SrcFeature2.Feature, rule.SrcFeature2.Value, rule.SrcPosition2)
					}
					xdimRows = append(xdimRows, fmt.Sprintf("XDIM\t%s>%s\t%s\t%s=%s@%d\t%s\t%s\t%s",
						a, b, src, rule.TgtDimension, rule.TgtValue, rule.TgtPositionOffset,
						num(rule.Count), num(rule.SrcCount), num(rule.Confidence)))
				}
				sort.Strings(xdimRows)
				for _, row := range xdimRows {
					fmt.Println(row)
				}
			}
		}
	case "malign":
		// Mirrors pairAlignmentEdges: sorted lect order, trained pair model.
		corpus := loadCorpus()
		model, err := regulae.TrainModel(corpus, regulae.DefaultTrainOptions())
		if err != nil {
			fmt.Fprintf(os.Stderr, "train: %v\n", err)
			os.Exit(1)
		}
		for _, cs := range corpus {
			var lects []string
			for l := range cs.Forms {
				lects = append(lects, l)
			}
			sort.Strings(lects)
			for i := 0; i < len(lects); i++ {
				for j := i + 1; j < len(lects); j++ {
					pm, ok := model.PairwiseModel(lects[i], lects[j])
					if !ok || pm == nil {
						continue
					}
					alignment, err := regulae.AlignForms(cs.Forms[lects[i]], cs.Forms[lects[j]], pm.FeatureSystem, 3, pm)
					if err != nil {
						fmt.Fprintf(os.Stderr, "align: %v\n", err)
						os.Exit(1)
					}
					var parts []string
					for _, link := range alignment.Links {
						var src, tgt string
						for _, s := range link.SourceChunk {
							src += s.Grapheme
						}
						for _, s := range link.TargetChunk {
							tgt += s.Grapheme
						}
						parts = append(parts, src+"/"+tgt)
					}
					mcost, _ := regulae.AlignmentCost(alignment, pm.FeatureSystem, pm)
					fmt.Printf("MALIGN\t%s\t%s>%s\t%.12f\t%s\n", cs.CognateID, lects[i], lects[j], mcost, strings.Join(parts, " "))
				}
			}
		}
	case "align":
		corpus := loadCorpus()
		for _, cs := range corpus {
			lects := cs.FormsOrder
			for i := 0; i < len(lects); i++ {
				for j := i + 1; j < len(lects); j++ {
					a := cs.Forms[lects[i]]
					b := cs.Forms[lects[j]]
					alignment, err := regulae.AlignForms(a, b, "descriptive", 0, nil)
					if err != nil {
						fmt.Fprintf(os.Stderr, "align: %v\n", err)
						os.Exit(1)
					}
					cost, err := regulae.AlignmentCost(alignment, "descriptive", nil)
					if err != nil {
						fmt.Fprintf(os.Stderr, "cost: %v\n", err)
						os.Exit(1)
					}
					var parts []string
					for _, link := range alignment.Links {
						var src, tgt string
						for _, s := range link.SourceChunk {
							src += s.Grapheme
						}
						for _, s := range link.TargetChunk {
							tgt += s.Grapheme
						}
						parts = append(parts, src+"/"+tgt)
					}
					fmt.Printf("ALIGN\t%s\t%s>%s\t%s\t%s\n", cs.CognateID, lects[i], lects[j], num(cost), strings.Join(parts, " "))
				}
			}
		}
	default:
		fmt.Fprintf(os.Stderr, "unknown mode %q\n", mode)
		os.Exit(2)
	}
}
