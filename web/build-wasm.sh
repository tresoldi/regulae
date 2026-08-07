#!/usr/bin/env bash
#
# Builds regulae for the browser. Requires the Emscripten SDK:
#
#   source ~/emsdk/emsdk_env.sh
#   ./web/build-wasm.sh
#
# The artifacts (regulae.js, regulae.wasm) are committed, so rerun this before
# deploying the page after any change under src/ or include/. This records the
# sources it built from in web/BUILD_INFO; the 'wasm_current' test and the Pages
# workflow both fail if that stamp falls behind, so a stale artifact cannot
# reach the deployed page unnoticed.
#
# Links with -sFILESYSTEM=0: the library is reached only through the
# parse-from-string loaders, and merkmal's built-in models are compiled in.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
build_dir="$repo_dir/build/wasm"

if ! command -v emcc >/dev/null 2>&1; then
    echo "build-wasm: emcc not found; run 'source ~/emsdk/emsdk_env.sh' first" >&2
    exit 2
fi

emcmake cmake -S "$repo_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DREGULAE_BUILD_TESTS=OFF \
    -DREGULAE_BUILD_CLI=OFF \
    -DMERKMAL_REQUIRE_UTF8PROC=OFF > /dev/null

cmake --build "$build_dir" -j"$(nproc)" > /dev/null

emcc \
    -O3 \
    "$script_dir/regulae_wasm.c" \
    "$build_dir/libregulae.a" \
    "$build_dir/_deps/merkmal/libmerkmal.a" \
    -I"$repo_dir/include" \
    -sFILESYSTEM=0 \
    -sMODULARIZE=1 \
    -sEXPORT_NAME=createRegulae \
    -sALLOW_MEMORY_GROWTH=1 \
    -sENVIRONMENT=web,worker,node \
    -sEXPORTED_FUNCTIONS='["_regulae_train_json","_regulae_segment_json","_regulae_version","_regulae_free","_malloc","_free"]' \
    -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString","stringToNewUTF8"]' \
    -o "$script_dir/regulae.js"

"$repo_dir/scripts/wasm-provenance.sh" --write

echo "built:"
ls -lh "$script_dir/regulae.js" "$script_dir/regulae.wasm" | awk '{print "  " $9 "  " $5}'
echo
echo "preview: cd web && python3 -m http.server 8080"
