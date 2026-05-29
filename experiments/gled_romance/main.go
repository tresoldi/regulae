// Command gled_romance loads a 7-way Romance subset from the GLED database
// and runs the multi-lect training pipeline, printing correspondence classes
// and cross-dimensional rules.
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
//	cd experiments/gled_romance && go run .
package main

import (
	"fmt"
	"os"
	"sort"

	regulae "github.com/tresoldi/regulae"
)

const defaultGLEDPath = "/tmp/gled_clone/releases/20221127/gled.tsv"

var romanceDoculects = map[string]bool{
	"LATIN":        true,
	"SPANISH":      true,
	"PORTUGUESE_2": true,
	"FRENCH_2":     true,
	"ITALIAN_2":    true,
	"CATALAN_3":    true,
	"ROMANIAN_2":   true,
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

	fmt.Printf("Loading GLED Romance subset from %s...\n", gledPath)
	corpus, err := regulae.LoadGLED(gledPath, regulae.GLEDLoadOptions{
		Family:    "Indo-European",
		Doculects: romanceDoculects,
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

	fmt.Printf("Loaded %d Romance cognate sets.\n", len(corpus))

	// per-lect coverage
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

	// cognate set sizes
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

	fmt.Println(regulae.FormatMultiLectModel(model, 30, 0))
}
