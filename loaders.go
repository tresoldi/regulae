package regulae

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Loaders for external cognate-set data formats: a generic TSV loader plus
// format-specific loaders for GLED and arcaverborum (Lexibank-derived) data.
// Each returns []CognateSet so downstream training is format-agnostic.

// parseSegments splits a space-separated segment string into Segments. Empty
// tokens and gap markers ("-") are dropped.
func parseSegments(raw string) []Segment {
	var out []Segment
	for _, tok := range strings.Fields(raw) {
		if tok == "-" {
			continue
		}
		out = append(out, Segment{Grapheme: tok})
	}
	return out
}

// parseAlignment parses a space-separated alignment column; "-" tokens become
// nil (gap), others become *Segment.
func parseAlignment(raw string) []*Segment {
	var out []*Segment
	for _, tok := range strings.Fields(raw) {
		if tok == "-" {
			out = append(out, nil)
		} else {
			s := Segment{Grapheme: tok}
			out = append(out, &s)
		}
	}
	return out
}

// parseArcaverborumSegments parses a Segments cell: space-separated graphemes
// with "+" marking morpheme boundaries. Returns the segments and the boundary
// positions (in the filtered sequence).
func parseArcaverborumSegments(raw string) ([]Segment, []int) {
	var segments []Segment
	var boundaries []int
	for _, tok := range strings.Fields(raw) {
		switch tok {
		case "+":
			if len(segments) > 0 {
				boundaries = append(boundaries, len(segments))
			}
		case "-":
			// gap marker, skip
		default:
			segments = append(segments, Segment{Grapheme: tok})
		}
	}
	return segments, boundaries
}

// dictReader reads a delimited file into header + rows, mirroring Python's
// csv.DictReader. Returns (header, rows). Lenient on quoting and field counts.
func dictReader(path string, delim rune) ([]string, [][]string, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer fh.Close()
	r := csv.NewReader(fh)
	r.Comma = delim
	r.LazyQuotes = true
	r.FieldsPerRecord = -1
	records, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(records) == 0 {
		return nil, nil, nil
	}
	return records[0], records[1:], nil
}

func columnIndex(header []string) map[string]int {
	idx := map[string]int{}
	for i, name := range header {
		idx[name] = i
	}
	return idx
}

func cell(row []string, idx map[string]int, col string) string {
	i, ok := idx[col]
	if !ok || i >= len(row) {
		return ""
	}
	return row[i]
}

// TSVLoadOptions configures LoadCognatesFromTSV. Empty column names fall back to
// the defaults ("cognate_id", "lect_id", "segments"). AlignmentCol/ConfidenceCol
// are optional (empty = not used).
type TSVLoadOptions struct {
	CognateIDCol  string
	LectIDCol     string
	SegmentsCol   string
	AlignmentCol  string
	ConfidenceCol string
}

func (o TSVLoadOptions) withDefaults() TSVLoadOptions {
	if o.CognateIDCol == "" {
		o.CognateIDCol = "cognate_id"
	}
	if o.LectIDCol == "" {
		o.LectIDCol = "lect_id"
	}
	if o.SegmentsCol == "" {
		o.SegmentsCol = "segments"
	}
	return o
}

// cognateAccumulator groups rows by cognate id while preserving first-seen
// orders.
type cognateAccumulator struct {
	order         []string
	forms         map[string]map[string]Form
	formsOrder    map[string][]string
	aligns        map[string]map[string][]*Segment
	boundaries    map[string]map[string][]int
	confidence    map[string]float64
	hasConfidence map[string]bool
}

func newCognateAccumulator() *cognateAccumulator {
	return &cognateAccumulator{
		forms:         map[string]map[string]Form{},
		formsOrder:    map[string][]string{},
		aligns:        map[string]map[string][]*Segment{},
		boundaries:    map[string]map[string][]int{},
		confidence:    map[string]float64{},
		hasConfidence: map[string]bool{},
	}
}

func (a *cognateAccumulator) ensure(cogID string) {
	if _, ok := a.forms[cogID]; !ok {
		a.order = append(a.order, cogID)
		a.forms[cogID] = map[string]Form{}
		a.aligns[cogID] = map[string][]*Segment{}
		a.boundaries[cogID] = map[string][]int{}
	}
}

// LoadCognatesFromTSV loads cognate sets from a TSV file (one row per
// (lect, cognate) pair). Rows sharing a cognate id are grouped; cognate sets
// are returned in first-appearance order.
func LoadCognatesFromTSV(path string, opts TSVLoadOptions) ([]CognateSet, error) {
	opts = opts.withDefaults()
	header, rows, err := dictReader(path, '\t')
	if err != nil {
		return nil, err
	}
	if header == nil {
		return nil, nil
	}
	idx := columnIndex(header)
	for _, col := range []string{opts.CognateIDCol, opts.LectIDCol, opts.SegmentsCol} {
		if _, ok := idx[col]; !ok {
			return nil, fmt.Errorf("TSV %s missing required column: %q", path, col)
		}
	}

	acc := newCognateAccumulator()
	for _, row := range rows {
		cogID := strings.TrimSpace(cell(row, idx, opts.CognateIDCol))
		lectID := strings.TrimSpace(cell(row, idx, opts.LectIDCol))
		if cogID == "" || lectID == "" {
			continue
		}
		segs := parseSegments(cell(row, idx, opts.SegmentsCol))
		if len(segs) == 0 {
			continue
		}
		acc.ensure(cogID)
		if _, dup := acc.forms[cogID][lectID]; dup {
			return nil, fmt.Errorf("TSV %s: cognate %q has duplicate row for lect %q", path, cogID, lectID)
		}
		acc.forms[cogID][lectID] = Form{LectID: lectID, Segments: segs}
		acc.formsOrder[cogID] = append(acc.formsOrder[cogID], lectID)

		if opts.AlignmentCol != "" {
			aligned := parseAlignment(cell(row, idx, opts.AlignmentCol))
			if len(aligned) > 0 {
				existing := acc.aligns[cogID]
				for _, v := range existing {
					if len(v) != len(aligned) {
						return nil, fmt.Errorf("TSV %s: cognate %q has alignment length mismatch across lects", path, cogID)
					}
					break
				}
				acc.aligns[cogID][lectID] = aligned
			}
		}
		if opts.ConfidenceCol != "" {
			rawConf := strings.TrimSpace(cell(row, idx, opts.ConfidenceCol))
			if rawConf != "" {
				c, perr := strconv.ParseFloat(rawConf, 64)
				if perr != nil {
					return nil, fmt.Errorf("TSV %s: cognate %q has non-numeric %s=%q", path, cogID, opts.ConfidenceCol, rawConf)
				}
				if !acc.hasConfidence[cogID] || c < acc.confidence[cogID] {
					acc.confidence[cogID] = c
					acc.hasConfidence[cogID] = true
				}
			}
		}
	}

	// The generic TSV loader keeps every cognate set regardless of lect count
	// (unlike LoadGLED/LoadArcaverborum, which default to min 2). minLects=1
	// drops nothing real (every set has >=1 form).
	return acc.build(1, false), nil
}

// build assembles the accumulated data into CognateSets. minLects drops sets
// with fewer than that many lects (use 0/1 to keep all); checkAlignLen discards
// alignment hints with inconsistent lengths.
func (a *cognateAccumulator) build(minLects int, dropMalformedAlign bool) []CognateSet {
	var out []CognateSet
	for _, cogID := range a.order {
		forms := a.forms[cogID]
		if minLects > 0 && len(forms) < minLects {
			continue
		}
		var aligns map[string][]*Segment
		if len(a.aligns[cogID]) > 0 {
			aligns = a.aligns[cogID]
			if dropMalformedAlign {
				lens := map[int]struct{}{}
				for _, v := range aligns {
					lens[len(v)] = struct{}{}
				}
				if len(lens) > 1 {
					aligns = nil
				}
			}
		}
		var bounds map[string][]int
		if len(a.boundaries[cogID]) > 0 {
			bounds = a.boundaries[cogID]
		}
		confidence := 1.0
		if a.hasConfidence[cogID] {
			confidence = a.confidence[cogID]
		}
		out = append(out, CognateSet{
			CognateID:          cogID,
			Forms:              forms,
			FormsOrder:         a.formsOrder[cogID],
			Alignments:         aligns,
			MorphemeBoundaries: bounds,
			Confidence:         confidence,
		})
	}
	return out
}

// GLEDLoadOptions configures LoadGLED. Family/Doculects are optional filters;
// MinLects defaults to 2 when zero.
type GLEDLoadOptions struct {
	Family    string
	Doculects map[string]bool
	MinLects  int
}

// LoadGLED loads GLED-format TSV data, grouping rows by COGSET and filtering by
// family and/or doculect. Uses the IPA column for segments and ALIGNMENT for
// the alignment hint.
func LoadGLED(path string, opts GLEDLoadOptions) ([]CognateSet, error) {
	minLects := opts.MinLects
	if minLects == 0 {
		minLects = 2
	}
	header, rows, err := dictReader(path, '\t')
	if err != nil {
		return nil, err
	}
	idx := columnIndex(header)
	for _, col := range []string{"DOCULECT", "FAMILY", "IPA", "COGSET"} {
		if _, ok := idx[col]; !ok {
			return nil, fmt.Errorf("GLED TSV %s: missing required column %q", path, col)
		}
	}

	acc := newCognateAccumulator()
	for _, row := range rows {
		if opts.Family != "" && cell(row, idx, "FAMILY") != opts.Family {
			continue
		}
		lectID := cell(row, idx, "DOCULECT")
		if opts.Doculects != nil && !opts.Doculects[lectID] {
			continue
		}
		cogID := strings.TrimSpace(cell(row, idx, "COGSET"))
		if cogID == "" {
			continue
		}
		segs := parseSegments(cell(row, idx, "IPA"))
		if len(segs) == 0 {
			continue
		}
		acc.ensure(cogID)
		if _, dup := acc.forms[cogID][lectID]; dup {
			continue // keep first reflex per (lect, cognate)
		}
		acc.forms[cogID][lectID] = Form{LectID: lectID, Segments: segs}
		acc.formsOrder[cogID] = append(acc.formsOrder[cogID], lectID)
		if raw := cell(row, idx, "ALIGNMENT"); raw != "" {
			acc.aligns[cogID][lectID] = parseAlignment(raw)
		}
	}
	return acc.build(minLects, true), nil
}

// ArcaverborumLoadOptions configures LoadArcaverborum. Dataset/LanguageIDs/Family
// are optional filters; MinLects defaults to 2 when zero.
type ArcaverborumLoadOptions struct {
	Dataset     string
	LanguageIDs map[string]bool
	Family      string
	MinLects    int
}

// LoadArcaverborum loads arcaverborum-format (merged Lexibank CLDF) data,
// auto-detecting TSV vs CSV by extension, grouping by the first Cognacy id, and
// recording morpheme boundaries from "+" tokens in Segments.
func LoadArcaverborum(path string, opts ArcaverborumLoadOptions) ([]CognateSet, error) {
	minLects := opts.MinLects
	if minLects == 0 {
		minLects = 2
	}
	delim := ','
	switch strings.ToLower(filepath.Ext(path)) {
	case ".tsv", ".txt":
		delim = '\t'
	}
	header, rows, err := dictReader(path, delim)
	if err != nil {
		return nil, err
	}
	if header == nil {
		return nil, nil
	}
	idx := columnIndex(header)
	for _, col := range []string{"Language_ID", "Segments", "Cognacy"} {
		if _, ok := idx[col]; !ok {
			return nil, fmt.Errorf("arcaverborum file %s: missing required column %q", path, col)
		}
	}

	acc := newCognateAccumulator()
	for _, row := range rows {
		if opts.Dataset != "" && cell(row, idx, "Dataset") != opts.Dataset {
			continue
		}
		if opts.Family != "" && cell(row, idx, "Family") != opts.Family {
			continue
		}
		lectID := cell(row, idx, "Language_ID")
		if lectID == "" {
			continue
		}
		if opts.LanguageIDs != nil && !opts.LanguageIDs[lectID] {
			continue
		}
		rawCog := cell(row, idx, "Cognacy")
		if rawCog == "" || rawCog == "<NA>" || rawCog == "NA" {
			continue
		}
		cogID := strings.TrimSpace(strings.SplitN(rawCog, ";", 2)[0])
		if cogID == "" {
			continue
		}
		segs, morphBounds := parseArcaverborumSegments(cell(row, idx, "Segments"))
		if len(segs) == 0 {
			continue
		}
		acc.ensure(cogID)
		if _, dup := acc.forms[cogID][lectID]; dup {
			continue
		}
		acc.forms[cogID][lectID] = Form{LectID: lectID, Segments: segs}
		acc.formsOrder[cogID] = append(acc.formsOrder[cogID], lectID)
		if len(morphBounds) > 0 {
			acc.boundaries[cogID][lectID] = morphBounds
		}
		if raw := cell(row, idx, "Alignment"); raw != "" && raw != "<NA>" && raw != "NA" {
			acc.aligns[cogID][lectID] = parseAlignment(raw)
		}
	}
	return acc.build(minLects, true), nil
}
