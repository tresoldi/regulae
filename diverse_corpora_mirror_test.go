package regulae

// Mirror of python/tests/test_diverse_corpora.py
//
// All tests skipped: they depend on Python experiment modules
// (run_experiment.py) in python/experiments/<dataset>/ that import
// the Python regulae package and call Python-specific loaders.
// These cannot be replicated in Go without porting those experiment
// scripts. The Go package has no equivalent loader for the
// real-data TSV corpora used by those experiments.
//
// Skipped tests:
//   TestDiverseCorporaCorpusLoadsAndTrains (7 parametrised variants)
//   TestDiverseCorporaCorpusProducesMultiLectClasses (7 parametrised variants)
//   TestDiverseCorporaMandarinRecoversVoicingTonogenesis
//   TestDiverseCorporaAthabaskanTonesExtractedIntoSegmentTone
//   TestDiverseCorporaContaminatedCognatesFixtureSuppressesNoise
//
// All 16 Python tests in this file are dataset-loader-dependent.
