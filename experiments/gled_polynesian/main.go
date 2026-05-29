// Command gled_polynesian loads a 6-way Polynesian subset from GLED and
// trains a multi-lect model, focusing on the Hawaiian k merger disambiguation.
//
// The GLED TSV is expected at /tmp/gled_clone/releases/20221127/gled.tsv.
// Pass an alternate path as os.Args[1].
//
// NOTE: this experiment requires an external GLED data file that is not
// bundled in the repository. Running without the file will print an error to
// stderr and exit 1.
//
// Run with:
//
//	cd experiments/gled_polynesian && go run .
package main

import (
	"fmt"
	"os"
	"sort"

	regulae "github.com/tresoldi/regulae"
)

const defaultGLEDPath = "/tmp/gled_clone/releases/20221127/gled.tsv"

var polyDoculects = map[string]bool{
	"HAWAIIAN_2": true,
	"MAORI":      true,
	"SAMOAN":     true,
	"TAHITIAN":   true,
	"RAROTONGAN": true,
	"RAPA_NUI":   true,
}

func main() {
	gledPath := defaultGLEDPath
	if len(os.Args) > 1 {
		gledPath = os.Args[1]
	}

	if _, err := os.Stat(gledPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "GLED data not found at %s.\n", gledPath)
		fmt.Fprintf(os.Stderr, "Clone https://github.com/tresoldi/gled and update the path.\n")
		os.Exit(1)
	}

	fmt.Printf("Loading GLED Polynesian subset from %s...\n", gledPath)
	corpus, err := regulae.LoadGLED(gledPath, regulae.GLEDLoadOptions{
		Family:    "Austronesian",
		Doculects: polyDoculects,
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

	lects := make([]string, 0, len(lectsPresent))
	for l := range lectsPresent {
		lects = append(lects, l)
	}
	sort.Strings(lects)
	fmt.Print("  per-lect coverage: map[")
	for i, l := range lects {
		if i > 0 {
			fmt.Print(" ")
		}
		fmt.Printf("%s:%d", l, lectsPresent[l])
	}
	fmt.Println("]")

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

	fmt.Printf("  lects: %v\n", model.LectIDs)
	fmt.Printf("  pairwise models: %d\n", len(model.PairwiseModels))
	fmt.Printf("  unconditioned classes: %d\n", len(model.UnconditionedClasses))
	fmt.Printf("  conditioned classes:   %d\n", len(model.ConditionedClasses))
	fmt.Println()

	fmt.Println("--- Unconditioned classes involving HAWAIIAN_2:k ---")
	var hawK []regulae.MultiLectCorrespondenceClass
	for _, klass := range model.UnconditionedClasses {
		if klass.Segments["HAWAIIAN_2"] == "k" {
			hawK = append(hawK, klass)
		}
	}
	if len(hawK) == 0 {
		fmt.Println("  (none — reconciliation did not emit any class with Haw:k)")
	}
	sort.Slice(hawK, func(i, j int) bool { return hawK[i].Count > hawK[j].Count })
	for _, klass := range hawK {
		segs := sortedSegStr(klass.Segments)
		fmt.Printf("  count=%5.1f  %s\n", klass.Count, segs)
	}
	fmt.Println()

	fmt.Println("--- Unconditioned classes where Samoan has t (proto *t reflex) ---")
	var samT []regulae.MultiLectCorrespondenceClass
	for _, klass := range model.UnconditionedClasses {
		if klass.Segments["SAMOAN"] == "t" {
			samT = append(samT, klass)
		}
	}
	sort.Slice(samT, func(i, j int) bool { return samT[i].Count > samT[j].Count })
	for _, klass := range samT {
		segs := sortedSegStr(klass.Segments)
		fmt.Printf("  count=%5.1f  %s\n", klass.Count, segs)
	}
	fmt.Println()

	fmt.Println("--- Top 25 unconditioned classes overall ---")
	for i, klass := range model.UnconditionedClasses {
		if i >= 25 {
			break
		}
		segs := sortedSegStr(klass.Segments)
		fmt.Printf("  [%d-way] count=%5.1f  %s\n", len(klass.Segments), klass.Count, segs)
	}
	fmt.Println()

	fmt.Println("--- Conditioned classes (top 15 by count) ---")
	cond := make([]regulae.MultiLectCorrespondenceClass, len(model.ConditionedClasses))
	copy(cond, model.ConditionedClasses)
	sort.Slice(cond, func(i, j int) bool { return cond[i].Count > cond[j].Count })
	for i, klass := range cond {
		if i >= 15 {
			break
		}
		segs := sortedSegStr(klass.Segments)
		ctxStr := formatContexts(klass)
		fmt.Printf("  count=%5.1f  %s   [%s]\n", klass.Count, segs, ctxStr)
	}
}

// sortedSegStr renders a Segments map as "lect:g lect:g ..." sorted by lect.
func sortedSegStr(segs map[string]string) string {
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
		out += l + ":" + segs[l]
	}
	return out
}

// formatContexts renders per-lect contexts for a conditioned class.
func formatContexts(klass regulae.MultiLectCorrespondenceClass) string {
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
			ps = append(ps, "prec=["+joinStrings(fs, ",")+"]")
		}
		if len(ctx.Following) > 0 {
			var fs []string
			for _, c := range ctx.Following {
				fs = append(fs, c.Feature)
			}
			ps = append(ps, "foll=["+joinStrings(fs, ",")+"]")
		}
		if len(ps) > 0 {
			parts = append(parts, lect+": "+joinStrings(ps, " "))
		}
	}
	if len(parts) == 0 {
		return "no constraints"
	}
	return joinStrings(parts, " | ")
}

func joinStrings(ss []string, sep string) string {
	out := ""
	for i, s := range ss {
		if i > 0 {
			out += sep
		}
		out += s
	}
	return out
}
