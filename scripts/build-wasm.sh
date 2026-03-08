#!/bin/bash
set -euo pipefail

# Build stochtree (BART) as WASM via Emscripten
# Prerequisites: emsdk activated (emcc in PATH)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UPSTREAM_DIR="${PROJECT_DIR}/upstream/stochtree"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

# Verify prerequisites
if ! command -v em++ &> /dev/null; then
  echo "ERROR: em++ not found. Activate emsdk first:"
  echo "  source ~/tools/emsdk/emsdk_env.sh"
  exit 1
fi

if [ ! -f "$UPSTREAM_DIR/CMakeLists.txt" ]; then
  echo "ERROR: stochtree upstream not found at ${UPSTREAM_DIR}"
  echo "  git submodule update --init --recursive"
  exit 1
fi

echo "=== Applying patches ==="
if [ -d "${PROJECT_DIR}/patches" ] && ls "${PROJECT_DIR}/patches"/*.patch &> /dev/null 2>&1; then
  for patch in "${PROJECT_DIR}/patches"/*.patch; do
    echo "Applying: $(basename "$patch")"
    (cd "$UPSTREAM_DIR" && git apply --check "$patch" 2>/dev/null && git apply "$patch") || \
      echo "  (already applied or not applicable)"
  done
else
  echo "  No patches found"
fi

echo "=== Collecting source files ==="
mkdir -p "$OUTPUT_DIR"

# Upstream source files (skip Python/R bindings: forest, sampler, kernel,
# serialization, cpp11, R_*, py_stochtree -- they use cpp11.hpp or pybind11)
UPSTREAM_SOURCES=""
for f in container cutpoint_candidates data io leaf_model \
         partition_tracker random_effects tree; do
  src="${UPSTREAM_DIR}/src/${f}.cpp"
  if [ -f "$src" ]; then
    UPSTREAM_SOURCES="$UPSTREAM_SOURCES $src"
  else
    echo "WARNING: missing upstream source: $src"
  fi
done

# Our C++ glue layer
GLUE_SOURCE="${PROJECT_DIR}/csrc/wl_stochtree_api.cpp"

echo "  Upstream sources: $(echo $UPSTREAM_SOURCES | wc -w | tr -d ' ') files"
echo "  Glue: 1 file"

echo "=== Compiling WASM ==="

# Exported functions from our C glue
EXPORTED_FUNCTIONS='[
  "_wl_st_get_last_error",
  "_wl_st_bart_create",
  "_wl_st_bart_fit",
  "_wl_st_bart_predict",
  "_wl_st_bart_predict_raw",
  "_wl_st_bart_to_json",
  "_wl_st_bart_from_json",
  "_wl_st_free_string",
  "_wl_st_bart_free",
  "_wl_st_bart_num_samples",
  "_wl_st_bart_num_trees",
  "_wl_st_bart_num_features",
  "_wl_st_bart_get_sigma2",
  "_wl_st_bart_get_task",
  "_wl_st_bart_get_nr_class",
  "_malloc",
  "_free"
]'

# Remove newlines for em++
EXPORTED_FUNCTIONS=$(echo "$EXPORTED_FUNCTIONS" | tr -d '\n' | tr -s ' ')

EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF64","HEAPU8","HEAP32","HEAP8","UTF8ToString"]'

# Include paths
INCLUDES="-I $UPSTREAM_DIR/include \
  -I $UPSTREAM_DIR/deps/eigen \
  -I $UPSTREAM_DIR/deps/boost_math/include \
  -I $UPSTREAM_DIR/deps/fmt/include \
  -I $UPSTREAM_DIR/deps/fast_double_parser/include"

em++ \
  $GLUE_SOURCE \
  $UPSTREAM_SOURCES \
  -std=c++17 \
  -O2 \
  -DNDEBUG \
  -DEIGEN_DONT_PARALLELIZE \
  -DEIGEN_MPL2_ONLY \
  -fexceptions \
  $INCLUDES \
  -o "${OUTPUT_DIR}/stochtree.js" \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s EXPORT_NAME=createStochtree \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=67108864 \
  -s ENVIRONMENT='web,node' \
  -s FORCE_FILESYSTEM=0

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: StochasticTree/stochtree
upstream_commit: $(cd "$UPSTREAM_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(em++ --version | head -1)
build_flags: -O2 -std=c++17 SINGLE_FILE=1 no-openmp
wasm_embedded: true
EOF

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/stochtree.js"
cat "${OUTPUT_DIR}/BUILD_INFO"
