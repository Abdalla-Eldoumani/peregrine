# legacy

Frozen v2 code from the 2024 pybind11 build. The kernel has known
tail-handling defects: it crashes or corrupts results at matrix sizes not
divisible by 4. Nothing here is imported or fixed; it is kept for history
and for comparison against the v3 rewrite. See the methodology section in
the root README for how the v2 numbers were re-evaluated.
