// Command arcaverborum_polynesian loads a 7-way Polynesian dataset from
// arcaverborum (Walworth's Proto-Polynesian wordlist) and trains a multi-lect
// model. With Tongan as anchor, the pipeline disambiguates the Hawaiian/Samoan
// glottal stop merger.
//
// Data: Zenodo archive 10.5281/zenodo.17294927, CoreCog collection. The loader
// expects an unpacked forms.csv at:
//
//	/tmp/arcaverborum_corecog/arcaverborum-A-corecog-20251008/forms.csv
//
// Pass an alternate path as os.Args[1].
//
// NOTE: this experiment requires an external data file that is not bundled in
// the repository. Running without the file will print an error to stderr and
// exit 1.
//
// Run with:
//
//	cd experiments/arcaverborum_polynesian && go run .
package main

import (
	"fmt"
	"os"
	"sort"

	regulae "github.com/tresoldi/regulae"
)

const defaultCSVPath = "/tmp/arcaverborum_corecog/arcaverborum-A-corecog-20251008/forms.csv"

var polyLects = map[string]bool{
	"walworthpolynesian_Hawaiian": true,
	"walworthpolynesian_Samoan":   true,
	"walworthpolynesian_Tongan":   true,
	"walworthpolynesian_Maori":    true,
	"walworthpolynesian_Tahitian": true,
	"walworthpolynesian_Niuean":   true,
	"walworthpolynesian_Tuvalu":   true,
}

// shortName maps full lect IDs to display abbreviations.
var shortName = map[string]string{
	"walworthpolynesian_Hawaiian": "Haw",
	"walworthpolynesian_Samoan":   "Sam",
	"walworthpolynesian_Tongan":   "Ton",
	"walworthpolynesian_Maori":    "Mao",
	"walworthpolynesian_Tahitian": "Tah",
	"walworthpolynesian_Niuean":   "Niu",
	"walworthpolynesian_Tuvalu":   "Tuv",
}

func short(lect string) string {
	if s, ok := shortName[lect]; ok {
		return s
	}
	return lect
}

func main() {
	csvPath := defaultCSVPath
	if len(os.Args) > 1 {
		csvPath = os.Args[1]
	}

	if _, err := os.Stat(csvPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "arcaverborum data not found at %s.\n", csvPath)
		fmt.Fprintf(os.Stderr, "Download from https://doi.org/10.5281/zenodo.17294927 (CoreCog collection) and unpack.\n")
		os.Exit(1)
	}

	fmt.Printf("Loading arcaverborum walworthpolynesian from %s...\n", csvPath)
	corpus, err := regulae.LoadArcaverborum(csvPath, regulae.ArcaverborumLoadOptions{
		Dataset:     "walworthpolynesian",
		LanguageIDs: polyLects,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "load error:", err)
		os.Exit(1)
	}

	lectsPresent := map[string]int{}
	sizeHist := map[int]int{}
	for _, cs := range corpus {
		for lect := range cs.Forms {
			lectsPresent[lect]++
		}
		sizeHist[len(cs.Forms)]++
	}

	fmt.Printf("Loaded %d Polynesian cognate sets.\n", len(corpus))

	// per-lect coverage with short names.
	lects := make([]string, 0, len(lectsPresent))
	for l := range lectsPresent {
		lects = append(lects, l)
	}
	sort.Strings(lects)
	fmt.Print("  per-lect coverage: {")
	for i, l := range lects {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%s: %d", short(l), lectsPresent[l])
	}
	fmt.Println("}")

	sizes := make([]int, 0, len(sizeHist))
	for s := range sizeHist {
		sizes = append(sizes, s)
	}
	sort.Ints(sizes)
	fmt.Print("  cognate set sizes: map[")
	for i, s := range sizes {
		if i > 0 {
			fmt.Print(" ")
		}
		fmt.Printf("%d:%d", s, sizeHist[s])
	}
	fmt.Println("]")
	fmt.Println()

	fmt.Println("Training multi-lect model...")
	model, err := regulae.TrainModel(corpus, regulae.DefaultTrainOptions())
	if err != nil {
		fmt.Fprintln(os.Stderr, "train error:", err)
		os.Exit(1)
	}

	// Display lect IDs using short names.
	shortLects := make([]string, len(model.LectIDs))
	for i, l := range model.LectIDs {
		shortLects[i] = short(l)
	}
	fmt.Printf("  lects: %v\n", shortLects)
	fmt.Printf("  pairwise models: %d\n", len(model.PairwiseModels))
	fmt.Printf("  unconditioned classes: %d\n", len(model.UnconditionedClasses))
	fmt.Printf("  conditioned classes:   %d\n", len(model.ConditionedClasses))
	fmt.Println()

	fmt.Println("--- Top 25 unconditioned classes overall ---")
	for i, klass := range model.UnconditionedClasses {
		if i >= 25 {
			break
		}
		segs := sortedSegStrShort(klass.Segments)
		fmt.Printf("  [%d-way] count=%5.1f  %s\n", len(klass.Segments), klass.Count, segs)
	}
	fmt.Println()

	fmt.Println("--- Classes with Tongan:k (proto *k anchor) ---")
	var tonK []regulae.MultiLectCorrespondenceClass
	for _, klass := range model.UnconditionedClasses {
		if klass.Segments["walworthpolynesian_Tongan"] == "k" {
			tonK = append(tonK, klass)
		}
	}
	sort.Slice(tonK, func(i, j int) bool { return tonK[i].Count > tonK[j].Count })
	for i, klass := range tonK {
		if i >= 15 {
			break
		}
		fmt.Printf("  count=%5.1f  %s\n", klass.Count, sortedSegStrShort(klass.Segments))
	}
	fmt.Println()

	fmt.Println("--- Classes with Samoan:ʔ (glottal — mixed *ʔ and *k) ---")
	var samGlot []regulae.MultiLectCorrespondenceClass
	for _, klass := range model.UnconditionedClasses {
		if klass.Segments["walworthpolynesian_Samoan"] == "ʔ" {
			samGlot = append(samGlot, klass)
		}
	}
	sort.Slice(samGlot, func(i, j int) bool { return samGlot[i].Count > samGlot[j].Count })
	for i, klass := range samGlot {
		if i >= 15 {
			break
		}
		fmt.Printf("  count=%5.1f  %s\n", klass.Count, sortedSegStrShort(klass.Segments))
	}
	fmt.Println()

	fmt.Println("--- Classes with Hawaiian:ʔ (glottal — mixed *ʔ and *k) ---")
	var hawGlot []regulae.MultiLectCorrespondenceClass
	for _, klass := range model.UnconditionedClasses {
		if klass.Segments["walworthpolynesian_Hawaiian"] == "ʔ" {
			hawGlot = append(hawGlot, klass)
		}
	}
	sort.Slice(hawGlot, func(i, j int) bool { return hawGlot[i].Count > hawGlot[j].Count })
	for i, klass := range hawGlot {
		if i >= 15 {
			break
		}
		fmt.Printf("  count=%5.1f  %s\n", klass.Count, sortedSegStrShort(klass.Segments))
	}
	fmt.Println()

	// Key diagnostic: Sam:ʔ + Ton:k (merger disambiguation).
	var mergerDisambig []regulae.MultiLectCorrespondenceClass
	for _, klass := range model.UnconditionedClasses {
		if klass.Segments["walworthpolynesian_Samoan"] == "ʔ" &&
			klass.Segments["walworthpolynesian_Tongan"] == "k" {
			mergerDisambig = append(mergerDisambig, klass)
		}
	}
	sort.Slice(mergerDisambig, func(i, j int) bool { return mergerDisambig[i].Count > mergerDisambig[j].Count })
	fmt.Printf("--- DIAGNOSTIC: Sam:ʔ + Ton:k classes (merger-disambig signal): %d ---\n", len(mergerDisambig))
	for _, klass := range mergerDisambig {
		fmt.Printf("  count=%5.1f  %s\n", klass.Count, sortedSegStrShort(klass.Segments))
	}
	fmt.Println()

	fmt.Println("--- Top 20 conditioned classes ---")
	cond := make([]regulae.MultiLectCorrespondenceClass, len(model.ConditionedClasses))
	copy(cond, model.ConditionedClasses)
	sort.Slice(cond, func(i, j int) bool { return cond[i].Count > cond[j].Count })
	for i, klass := range cond {
		if i >= 20 {
			break
		}
		segs := sortedSegStrShort(klass.Segments)
		ctxStr := formatContextsShort(klass)
		fmt.Printf("  count=%5.1f  %s   [%s]\n", klass.Count, segs, ctxStr)
	}
}

// sortedSegStrShort renders a Segments map using short lect names.
func sortedSegStrShort(segs map[string]string) string {
	keys := make([]string, 0, len(segs))
	for l := range segs {
		keys = append(keys, l)
	}
	sort.Strings(keys)
	out := ""
	for i, l := range keys {
		if i > 0 {
			out += " "
		}
		out += short(l) + ":" + segs[l]
	}
	return out
}

// formatContextsShort renders per-lect contexts using short names.
func formatContextsShort(klass regulae.MultiLectCorrespondenceClass) string {
	if klass.Contexts == nil {
		return "no constraints"
	}
	lects := make([]string, 0, len(klass.Contexts))
	for l := range klass.Contexts {
		lects = append(lects, l)
	}
	sort.Strings(lects)

	var parts []string
	for _, lect := range lects {
		ctx := klass.Contexts[lect]
		if ctx.ConstraintCount() == 0 {
			continue
		}
		var ps []string
		if ctx.Position != "" {
			ps = append(ps, "pos="+ctx.Position)
		}
		if len(ctx.Preceding) > 0 {
			var fs []string
			for _, c := range ctx.Preceding {
				fs = append(fs, c.Feature)
			}
			ps = append(ps, "prec=["+joinStr(fs, ",")+"]")
		}
		if len(ctx.Following) > 0 {
			var fs []string
			for _, c := range ctx.Following {
				fs = append(fs, c.Feature)
			}
			ps = append(ps, "foll=["+joinStr(fs, ",")+"]")
		}
		if len(ps) > 0 {
			parts = append(parts, short(lect)+": "+joinStr(ps, " "))
		}
	}
	if len(parts) == 0 {
		return "no constraints"
	}
	return joinStr(parts, " | ")
}

func joinStr(ss []string, sep string) string {
	out := ""
	for i, s := range ss {
		if i > 0 {
			out += sep
		}
		out += s
	}
	return out
}
