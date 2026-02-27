# Upstream: StochasticTree/stochtree

- Repository: https://github.com/StochasticTree/stochtree
- License: MIT
- Tracked commit: 417a40d04b6abdcde5814275b63894ede88e3e4f
- Submodule path: `upstream/stochtree/`

## Modifications for WASM

- No OpenMP (single-threaded). OpenMP is already guarded with `#ifdef _OPENMP` in upstream.
- Compiled only the core C++ library files (container, cutpoint_candidates, data, io, leaf_model, partition_tracker, random_effects, tree). Excluded R/Python binding files (forest.cpp, sampler.cpp, kernel.cpp, serialization.cpp, py_stochtree.cpp).
- Added `csrc/wl_stochtree_api.cpp` as a high-level extern "C" glue layer that internalizes the MCMC training loop (GFR warm-start + MCMC burn-in/sampling). The C++ tree sampler API is exposed through simple fit/predict/serialize functions.
- Albert-Chib data augmentation for probit classification implemented in the C++ glue layer (matching upstream Python-level implementation).

## Update policy

- Track upstream releases/tags as needed
- Rebuild WASM when upstream changes affect the compiled source files
- Local patches in `patches/` directory (currently none needed)

## Dependencies (all header-only, vendored in upstream)

- Eigen (MPL2-only subset)
- Boost.Math (header-only)
- fmt (header-only)
- nlohmann/json (header-only)
- fast_double_parser (header-only)
