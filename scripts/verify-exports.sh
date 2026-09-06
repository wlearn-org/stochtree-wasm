#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASM_FILE="${1:-${SCRIPT_DIR}/../wasm/stochtree.js}"

node - "$WASM_FILE" <<'NODE'
const assert = require('node:assert/strict')
const path = require('node:path')
const createModule = require(path.resolve(process.argv[2]))
const expected = [
  '_wl_st_get_last_error', '_wl_st_bart_create', '_wl_st_bart_fit',
  '_wl_st_bart_predict', '_wl_st_bart_predict_raw', '_wl_st_bart_to_json',
  '_wl_st_bart_from_json', '_wl_st_free_string', '_wl_st_bart_free',
  '_wl_st_bart_num_samples', '_wl_st_bart_num_trees', '_wl_st_bart_num_features',
  '_wl_st_bart_get_sigma2', '_wl_st_bart_get_task', '_wl_st_bart_get_nr_class',
  '_malloc', '_free'
]
createModule().then(module => {
  for (const name of expected) {
    assert.equal(typeof module[name], 'function', `Missing WASM export: ${name}`)
  }
  console.log(`All ${expected.length} exports verified.`)
}).catch(error => {
  console.error(error.message)
  process.exitCode = 1
})
NODE
