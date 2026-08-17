#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "regulae.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The native surface the Python package is a thin wrapper over. It is JSON in,
 * JSON out: the C core does the training and renders the model, and everything
 * the package exposes is a dataclass parsed from that string. The boundary is
 * the JSON contract, not the ABI struct layout, so an ABI bump never reaches
 * Python -- which is the whole reason the package stopped reimplementing the
 * engine.
 *
 * Built abi3 (Py_LIMITED_API), a file-static context, m_size = -1: no
 * subinterpreter support, which this does not need. */

static rg_context *default_context = NULL;
static PyObject *rg_py_error = NULL;
static PyObject *rg_py_source_marker = NULL;

/* Built once and kept. Constructing it per call would re-resolve merkmal's
 * whole registry every time. */
static rg_context *shared_context(void) {
    if (default_context == NULL) {
        if (rg_context_new_builtin(&default_context) != RG_OK) {
            return NULL;
        }
    }
    return default_context;
}

/* Turns a non-OK status into the right Python exception. RG_ERR_SOURCE_MARKER
 * is a documented gap in the source data -- CLDF/CLTS markup, not a sound -- so
 * it is its own ValueError subclass, letting a caller skip those without
 * swallowing a grapheme the feature system genuinely lacks. */
static void raise_status(rg_status status, const char *detail) {
    PyObject *exc = (status == RG_ERR_SOURCE_MARKER) ? rg_py_source_marker : rg_py_error;
    if (detail != NULL && detail[0] != '\0') {
        PyErr_SetString(exc, detail);
    } else {
        PyErr_Format(exc, "regulae failed with status %d", (int)status);
    }
}

/* If the failure was an unreadable grapheme, name it and the system. Returns 1
 * and fills `out` when it did, 0 otherwise. */
static int describe_grapheme_error(rg_context *ctx, rg_status status, char *out, size_t out_size) {
    const char *grapheme = NULL;
    const char *system = NULL;
    if (status != RG_ERR_UNKNOWN_GRAPHEME || ctx == NULL) {
        return 0;
    }
    rg_context_last_error(ctx, &grapheme, &system);
    if (grapheme == NULL) {
        return 0;
    }
    snprintf(out, out_size, "grapheme \"%s\" is not in the \"%s\" feature system",
             grapheme, system == NULL ? "" : system);
    return 1;
}

static rg_status parse_corpus(rg_context *ctx, const char *corpus_text, const char *format,
                              rg_corpus **out_corpus) {
    if (format == NULL || format[0] == '\0' || strcmp(format, "wide") == 0) {
        return rg_corpus_parse_wide_tsv(ctx, corpus_text, 0, out_corpus, 0);
    }
    if (strcmp(format, "tsv") == 0) {
        rg_tsv_load_options tsv_options;
        memset(&tsv_options, 0, sizeof(tsv_options));
        tsv_options.confidence_column = "confidence";
        return rg_corpus_parse_tsv(corpus_text, &tsv_options, out_corpus, 0);
    }
    if (strcmp(format, "gled") == 0) {
        return rg_corpus_parse_gled(corpus_text, 0, out_corpus, 0);
    }
    if (strcmp(format, "arcaverborum") == 0) {
        return rg_corpus_parse_arcaverborum(corpus_text, 0, out_corpus, 0);
    }
    return RG_ERR_UNSUPPORTED_OPTION;
}

/* train(corpus_text, format="wide", options_json=None) -> json str
 *
 * The whole model, rendered by the same rg_model_to_json the CLI uses, so the
 * Python model is exactly the CLI's `train --json`. Options are passed as a
 * flat JSON object the C core reads, which keeps every knob available without
 * this module having to know any of them. */
static PyObject *py_train(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *keywords[] = {"corpus_text", "format", "options_json", NULL};
    const char *corpus_text = NULL;
    const char *format = "wide";
    const char *options_json = NULL;
    rg_context *ctx;
    rg_corpus *corpus = NULL;
    rg_multi_model *model = NULL;
    rg_train_options options;
    char detail[256];
    char *json;
    rg_status status;
    PyObject *result;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|zz", keywords,
                                     &corpus_text, &format, &options_json)) {
        return NULL;
    }
    ctx = shared_context();
    if (ctx == NULL) {
        PyErr_SetString(rg_py_error, "could not create the feature context");
        return NULL;
    }
    if (corpus_text[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "the corpus is empty");
        return NULL;
    }

    detail[0] = '\0';
    status = rg_train_options_from_json(options_json, &options, detail, sizeof(detail));
    if (status != RG_OK) {
        raise_status(status, detail);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    status = parse_corpus(ctx, corpus_text, format, &corpus);
    Py_END_ALLOW_THREADS
    if (status != RG_OK) {
        char message[256];
        if (describe_grapheme_error(ctx, status, message, sizeof(message))) {
            raise_status(status, message);
        } else if (status == RG_ERR_UNSUPPORTED_OPTION) {
            PyErr_Format(PyExc_ValueError, "unknown corpus format: %s", format);
        } else {
            raise_status(status, "the corpus could not be read");
        }
        return NULL;
    }
    if (rg_corpus_cognate_count(corpus) == 0) {
        rg_corpus_free(corpus);
        PyErr_SetString(PyExc_ValueError, "no cognate set had two or more lects");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    status = rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus),
                            &options, &model);
    Py_END_ALLOW_THREADS
    if (status != RG_OK) {
        char message[256];
        int described = describe_grapheme_error(ctx, status, message, sizeof(message));
        rg_corpus_free(corpus);
        raise_status(status, described ? message : NULL);
        return NULL;
    }

    json = rg_model_to_json(ctx, model, rg_corpus_cognates(corpus),
                            rg_corpus_cognate_count(corpus), &options, true, true);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
    if (json == NULL) {
        PyErr_SetString(rg_py_error, "could not render the model");
        return NULL;
    }
    result = PyUnicode_FromString(json);
    rg_string_free(json);
    return result;
}

/* segment(word) -> json str: how one written word will be read, so a caller can
 * check its input before a full run. */
static PyObject *py_segment(PyObject *self, PyObject *args) {
    const char *word = NULL;
    rg_context *ctx;
    rg_segment *segments = NULL;
    size_t count = 0;
    char *json;
    rg_status status;
    PyObject *result;

    (void)self;
    if (!PyArg_ParseTuple(args, "s", &word)) {
        return NULL;
    }
    ctx = shared_context();
    if (ctx == NULL) {
        PyErr_SetString(rg_py_error, "could not create the feature context");
        return NULL;
    }
    status = rg_context_segment_word(ctx, word, &segments, &count);
    if (status != RG_OK) {
        char message[256];
        if (describe_grapheme_error(ctx, status, message, sizeof(message))) {
            raise_status(status, message);
        } else {
            raise_status(status, "the word could not be segmented");
        }
        return NULL;
    }
    json = rg_segments_to_json(segments, count);
    rg_segments_free(segments, count);
    if (json == NULL) {
        PyErr_SetString(rg_py_error, "could not render the segmentation");
        return NULL;
    }
    result = PyUnicode_FromString(json);
    rg_string_free(json);
    return result;
}

static PyObject *py_version(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyUnicode_FromString(rg_version_string());
}

static PyObject *py_abi_version(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyLong_FromUnsignedLong((unsigned long)rg_abi_version());
}

static PyMethodDef methods[] = {
    {"train", (PyCFunction)py_train, METH_VARARGS | METH_KEYWORDS,
     "train(corpus_text, format='wide', options_json=None) -> str\n\n"
     "Train a model on a corpus and return it as JSON."},
    {"segment", py_segment, METH_VARARGS,
     "segment(word) -> str\n\nSegment one written word and return it as JSON."},
    {"version", py_version, METH_NOARGS, "Return the library version string."},
    {"abi_version", py_abi_version, METH_NOARGS, "Return the ABI version integer."},
    {NULL, NULL, 0, NULL}
};

static void module_free(void *module) {
    (void)module;
    if (default_context != NULL) {
        rg_context_free(default_context);
        default_context = NULL;
    }
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Native C-backed surface for regulae.",
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    module_free
};

PyMODINIT_FUNC PyInit__native(void) {
    PyObject *module = PyModule_Create(&moduledef);
    if (module == NULL) {
        return NULL;
    }
    rg_py_error = PyErr_NewException("regulae.RegulaeError", NULL, NULL);
    if (rg_py_error == NULL) {
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(rg_py_error);
    if (PyModule_AddObject(module, "RegulaeError", rg_py_error) < 0) {
        Py_DECREF(rg_py_error);
        Py_DECREF(module);
        return NULL;
    }
    rg_py_source_marker = PyErr_NewException("regulae.SourceMarkerError", PyExc_ValueError, NULL);
    if (rg_py_source_marker == NULL) {
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(rg_py_source_marker);
    if (PyModule_AddObject(module, "SourceMarkerError", rg_py_source_marker) < 0) {
        Py_DECREF(rg_py_source_marker);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
