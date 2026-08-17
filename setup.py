from __future__ import annotations

import glob
import os
import subprocess

from setuptools import Extension, setup

# The C core -- regulae's, the vendored cJSON, and merkmal's -- is compiled
# straight into the extension, so a built wheel carries no dependency on an
# installed libregulae or libmerkmal. This file lives at the repository root
# rather than under python/ so an sdist can reach src/, include/ and the sibling
# merkmal checkout; shipped from python/ it would have no C core at all.
#
# Sources are globbed rather than listed, because the same list already lives in
# CMakeLists.txt and web/build-wasm.sh, and the copy that gets forgotten fails
# at link time in whichever build nobody ran locally. Sorted for reproducible
# builds.
HERE = os.path.dirname(os.path.abspath(__file__))

# merkmal is a sibling C checkout, the same default CMakeLists.txt uses. Its C
# is compiled in, so the build needs its sources -- not the merkmal Python
# package.
MERKMAL_DIR = os.environ.get(
    "REGULAE_MERKMAL_SOURCE_DIR", os.path.join(HERE, "..", "merkmal")
)
if not os.path.exists(os.path.join(MERKMAL_DIR, "include", "merkmal.h")):
    raise SystemExit(
        f"regulae: merkmal C sources not found at {MERKMAL_DIR!r}. Set "
        "REGULAE_MERKMAL_SOURCE_DIR to a merkmal checkout."
    )


def rel(path: str) -> str:
    return os.path.relpath(path, HERE)


SOURCES = (
    ["python/src/regulae_module.c", "third_party/cjson/cJSON.c"]
    + sorted(rel(p) for p in glob.glob(os.path.join(HERE, "src", "*.c")))
    + sorted(
        rel(p)
        for pattern in ("src/*.c", "src/generated/*.c")
        for p in glob.glob(os.path.join(MERKMAL_DIR, pattern))
    )
)

# merkmal normalises Unicode through libutf8proc when it is present and an
# IPA-focused fallback otherwise. Detect it the way merkmal's own setup.py does,
# so the two builds agree on what a grapheme normalises to. A distribution build
# should set REGULAE_REQUIRE_UTF8PROC to fail rather than ship the fallback.
REQUIRE_UTF8PROC = os.environ.get("REGULAE_REQUIRE_UTF8PROC", "").lower() in {
    "1",
    "on",
    "true",
    "yes",
}


def utf8proc_options() -> tuple[list[tuple[str, str]], list[str], list[str]]:
    try:
        subprocess.run(
            ["pkg-config", "--exists", "libutf8proc"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        if REQUIRE_UTF8PROC:
            raise SystemExit(
                "regulae: libutf8proc not found via pkg-config, and "
                "REGULAE_REQUIRE_UTF8PROC is set."
            ) from None
        return [("MK_HAVE_UTF8PROC", "0")], [], []
    cflags = subprocess.check_output(
        ["pkg-config", "--cflags", "libutf8proc"], text=True
    ).split()
    libs = subprocess.check_output(
        ["pkg-config", "--libs", "libutf8proc"], text=True
    ).split()
    return [("MK_HAVE_UTF8PROC", "1")], cflags, libs


utf8proc_macros, utf8proc_cflags, utf8proc_ldflags = utf8proc_options()

setup(
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
    ext_modules=[
        Extension(
            "regulae._native",
            sources=SOURCES,
            include_dirs=[
                "include",
                "src",
                "third_party/cjson",
                os.path.join(MERKMAL_DIR, "include"),
                os.path.join(MERKMAL_DIR, "src"),
            ],
            define_macros=[
                ("Py_LIMITED_API", "0x030C0000"),
                *utf8proc_macros,
            ],
            py_limited_api=True,
            extra_compile_args=["-std=c99", *utf8proc_cflags],
            extra_link_args=[*utf8proc_ldflags],
        )
    ],
)
