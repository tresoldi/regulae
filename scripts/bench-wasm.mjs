// Times the WebAssembly build against the native CLI on the same corpora.
//
// Both sides do the same work: parse, train, then render the model with
// alignments and outliers. The native side runs `regulae train --json`, which
// is the adapter's exact workload.
//
// Usage: node scripts/bench-wasm.mjs [runs]
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const repo = dirname(here);
const runs = Number(process.argv[2] ?? 3);
const cli = process.env.REGULAE_CLI ?? join(repo, 'build/c/regulae');

const corpora = [
  ['real_tone_chinese.tsv', 'tsv'],
  ['real_contaminated.tsv', 'tsv'],
  ['real_ppn_hawaiian.tsv', 'tsv'],
  ['real_latin_spanish.tsv', 'tsv'],
  ['real_romance_4lect.tsv', 'tsv'],
];

const createRegulae = (await import(join(repo, 'web/regulae.js'))).default;
const Module = await createRegulae({});
const train = Module.cwrap('regulae_train_json', 'number', ['string', 'string', 'string']);

const best = (fn) => {
  let ms = Infinity;
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    fn();
    ms = Math.min(ms, performance.now() - t0);
  }
  return ms / 1000;
};

console.log('corpus'.padEnd(26), 'native'.padStart(8), 'wasm'.padStart(8), 'ratio'.padStart(8));
console.log('-'.repeat(26), '-'.repeat(8), '-'.repeat(8), '-'.repeat(8));

for (const [name, format] of corpora) {
  const path = join(repo, 'testdata/corpora', name);
  const text = readFileSync(path, 'utf8');

  const native = best(() => execFileSync(cli, ['train', '--json', path], { maxBuffer: 1 << 28 }));
  const wasm = best(() => {
    const ptr = train(text, format, null);
    Module.UTF8ToString(ptr);
    Module.ccall('regulae_free', null, ['number'], [ptr]);
  });

  console.log(
    name.padEnd(26),
    native.toFixed(2).padStart(8),
    wasm.toFixed(2).padStart(8),
    (wasm / native).toFixed(2).concat('x').padStart(8),
  );
}

console.log('\nnative timings include process start; wasm reuses one module');
