#pragma once
#include <Python.h>

namespace pg::bindings {

// Imports the NumPy C API and adds the fast entries, the routing-fact setters,
// and _build_info to the module. Returns -1 with a Python error set on failure.
// The NumPy headers stay out of this header so no other translation unit gets
// its own unimported copy of the NumPy API table.
int add_fast_entries(PyObject* module);

} // namespace pg::bindings
