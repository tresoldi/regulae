# cJSON (vendored)

Upstream: <https://github.com/DaveGamble/cJSON>
Version: 1.7.19
Licence: MIT (see `LICENSE`, kept verbatim)

Files are unmodified. To update, replace `cJSON.c`, `cJSON.h` and `LICENSE`
from upstream and rerun the test suite; nothing here is patched, so there is no
diff to carry forward.

## Why it is here

regulae emits JSON for the CLI's `--json` output and for the WebAssembly
adapter, and reads a small options object back. Emitting is easy to hand-roll;
reading arbitrary input from a browser is not, and a hand-rolled parser exposed
to paste from the network is the kind of thing that is wrong for a year before
anyone notices.

This is regulae's only dependency other than merkmal. It is deliberately
confined: nothing in the training pipeline touches it, and no cJSON type
appears in `include/regulae.h`. If it ever needs removing, the blast radius is
`src/json.c` and `web/regulae_wasm.c`.

## Build notes

`CMakeLists.txt` compiles this file with warnings disabled. regulae itself
builds at zero warnings under `-Wall -Wextra -Wpedantic -Wconversion`, and
third-party source held to that standard would mean carrying local patches,
which is worse than scoping the flags.

`CJSON_HIDE_SYMBOLS` is defined and the library is built with
`C_VISIBILITY_PRESET hidden`, so a **shared** regulae exports no `cJSON_*`
symbols (verified: `nm -D libregulae.so | grep cJSON` is empty). A **static**
`libregulae.a` still carries them as global symbols, as any vendored static
dependency would, so a consumer statically linking both regulae and their own
cJSON can collide. Prefixing the symbols would fix that at the cost of patching
upstream, which is not worth it until someone actually hits it.
