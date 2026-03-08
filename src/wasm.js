// WASM loader -- loads the stochtree WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

async function loadStochtree(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    // SINGLE_FILE=1: .wasm is embedded in the .js file, no locateFile needed
    const createStochtree = require('../wasm/stochtree.cjs')
    wasmModule = await createStochtree(options)
    return wasmModule
  })()

  return loading
}

function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadStochtree() first')
  return wasmModule
}

module.exports = { loadStochtree, getWasm }
