import { loadStochtree, getWasm } from './wasm.js'
import {
  normalizeY,
  encodeBundle, decodeBundle,
  register,
  DisposedError, NotFittedError
} from '@wlearn/core'

// FinalizationRegistry safety net -- warns if dispose() was never called
const leakRegistry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ref, freeFn }) => {
    if (ref[0]) {
      console.warn('@wlearn/stochtree: BARTModel was not disposed -- calling free() automatically.')
      freeFn(ref[0])
    }
  })
  : null

// Internal sentinel for load path
const LOAD_SENTINEL = Symbol('load')

function getLastError() {
  return getWasm().ccall('wl_st_get_last_error', 'string', [], [])
}

export class BARTModel {
  #handle = null
  #freed = false
  #handleRef = null
  #params = {}
  #fitted = false
  #nrClass = 0
  #classes = null
  #isRegressor = true
  #numFeatures = 0

  constructor(sentinel, arg1, arg2) {
    if (sentinel === LOAD_SENTINEL) {
      // Load path: arg1 = handle, arg2 = { params, nrClass, classes, ... }
      this.#handle = arg1
      this.#params = arg2.params || {}
      this.#nrClass = arg2.nrClass || 0
      this.#classes = arg2.classes ? new Int32Array(arg2.classes) : null
      this.#isRegressor = !arg2.nrClass || arg2.nrClass === 0
      this.#numFeatures = arg2.numFeatures || 0
      this.#fitted = true
      this.#freed = false
      this.#handleRef = [this.#handle]
      if (leakRegistry) {
        leakRegistry.register(this, {
          ref: this.#handleRef,
          freeFn: (h) => { try { getWasm()._wl_st_bart_free(h) } catch {} }
        }, this)
      }
    } else {
      // Normal construction from create()
      this.#params = sentinel || {}
      this.#freed = false
    }
  }

  static async create(params = {}) {
    await loadStochtree()
    return new BARTModel(params)
  }

  // --- Estimator interface ---

  fit(X, y) {
    this.#ensureFitted(false)

    // Dispose previous model if refitting
    if (this.#handle) {
      getWasm()._wl_st_bart_free(this.#handle)
      this.#handle = null
      if (this.#handleRef) this.#handleRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yNorm = normalizeY(y)
    if (yNorm.length !== rows) {
      throw new Error(`y length (${yNorm.length}) does not match X rows (${rows})`)
    }

    // Determine task
    const isRegressor = this.#params.objective === 'regression' || this.#detectRegression(yNorm)
    this.#isRegressor = isRegressor

    if (!isRegressor) {
      const unique = new Set()
      for (let i = 0; i < yNorm.length; i++) {
        const v = yNorm[i]
        if (v !== Math.floor(v)) throw new Error(`Classifier labels must be integers, got ${v} at index ${i}`)
        unique.add(v)
      }
      this.#classes = new Int32Array([...unique].sort((a, b) => a - b))
      this.#nrClass = this.#classes.length
    } else {
      this.#classes = null
      this.#nrClass = 0
    }

    // Extract params
    const numTrees = this.#params.numTrees || 200
    const numGfr = this.#params.numGfr ?? 10
    const numBurnin = this.#params.numBurnin ?? 200
    const numSamples = this.#params.numSamples || 100
    const alpha = this.#params.alpha || 0.95
    const beta = this.#params.beta || 2.0
    const minSamplesLeaf = this.#params.minSamplesLeaf || 5
    const maxDepth = this.#params.maxDepth ?? -1
    const leafScale = this.#params.leafScale ?? -1.0
    const cutpointGrid = this.#params.cutpointGrid || 100
    const seed = this.#params.seed ?? 42

    // Create BART handle
    const handle = wasm._wl_st_bart_create(
      numTrees, numGfr, numBurnin, numSamples,
      alpha, beta, minSamplesLeaf, maxDepth,
      leafScale, cutpointGrid, seed
    )
    if (!handle) throw new Error(`BARTModel create failed: ${getLastError()}`)

    // Allocate WASM heap
    const task = isRegressor ? 0 : 1
    const yF64 = new Float64Array(yNorm)
    const xPtr = wasm._malloc(xData.byteLength)
    const yPtr = wasm._malloc(yF64.byteLength)
    wasm.HEAPF64.set(xData, xPtr / 8)
    wasm.HEAPF64.set(yF64, yPtr / 8)

    const ret = wasm._wl_st_bart_fit(handle, xPtr, rows, cols, yPtr, task)
    wasm._free(xPtr)
    wasm._free(yPtr)

    if (ret !== 0) {
      wasm._wl_st_bart_free(handle)
      throw new Error(`BARTModel fit failed: ${getLastError()}`)
    }

    this.#handle = handle
    this.#numFeatures = cols
    this.#fitted = true
    this.#handleRef = [handle]
    if (leakRegistry) {
      leakRegistry.register(this, {
        ref: this.#handleRef,
        freeFn: (h) => { try { getWasm()._wl_st_bart_free(h) } catch {} }
      }, this)
    }

    return this
  }

  predict(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const xPtr = wasm._malloc(xData.byteLength)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const outPtr = wasm._malloc(rows * 8)

    const ret = wasm._wl_st_bart_predict(this.#handle, xPtr, rows, cols, outPtr)
    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(outPtr)
      throw new Error(`Predict failed: ${getLastError()}`)
    }

    const raw = new Float64Array(rows)
    for (let i = 0; i < rows; i++) raw[i] = wasm.HEAPF64[outPtr / 8 + i]

    wasm._free(xPtr)
    wasm._free(outPtr)

    if (this.#isRegressor) {
      return raw
    }

    // Classification: convert probabilities to class labels
    const result = new Float64Array(rows)
    for (let i = 0; i < rows; i++) {
      // raw[i] is P(class 1) from probit
      result[i] = raw[i] >= 0.5 ? this.#classes[1] : this.#classes[0]
    }
    return result
  }

  predictProba(X) {
    this.#ensureFitted()
    if (this.#isRegressor) {
      throw new Error('predictProba is not available for regression')
    }

    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const xPtr = wasm._malloc(xData.byteLength)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const outPtr = wasm._malloc(rows * 8)

    const ret = wasm._wl_st_bart_predict(this.#handle, xPtr, rows, cols, outPtr)
    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(outPtr)
      throw new Error(`PredictProba failed: ${getLastError()}`)
    }

    // Binary: expand to [P(0), P(1)] per sample
    const result = new Float64Array(rows * 2)
    for (let i = 0; i < rows; i++) {
      const p1 = wasm.HEAPF64[outPtr / 8 + i]
      result[i * 2] = 1.0 - p1
      result[i * 2 + 1] = p1
    }

    wasm._free(xPtr)
    wasm._free(outPtr)
    return result
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = normalizeY(y)

    if (this.#isRegressor) {
      let ssRes = 0, ssTot = 0, yMean = 0
      for (let i = 0; i < yArr.length; i++) yMean += yArr[i]
      yMean /= yArr.length
      for (let i = 0; i < yArr.length; i++) {
        ssRes += (yArr[i] - preds[i]) ** 2
        ssTot += (yArr[i] - yMean) ** 2
      }
      return ssTot === 0 ? 0 : 1 - ssRes / ssTot
    }

    // Accuracy
    let correct = 0
    for (let i = 0; i < preds.length; i++) {
      if (preds[i] === yArr[i]) correct++
    }
    return correct / preds.length
  }

  // --- BART-specific: posterior access ---

  predictPosterior(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const ns = wasm._wl_st_bart_num_samples(this.#handle)

    const xPtr = wasm._malloc(xData.byteLength)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const outPtr = wasm._malloc(rows * ns * 8)
    const nsPtr = wasm._malloc(4)

    const ret = wasm._wl_st_bart_predict_raw(this.#handle, xPtr, rows, cols, outPtr, nsPtr)
    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(outPtr)
      wasm._free(nsPtr)
      throw new Error(`PredictPosterior failed: ${getLastError()}`)
    }

    const actualNs = wasm.HEAP32[nsPtr / 4]
    const result = new Float64Array(rows * actualNs)
    for (let i = 0; i < rows * actualNs; i++) {
      result[i] = wasm.HEAPF64[outPtr / 8 + i]
    }

    wasm._free(xPtr)
    wasm._free(outPtr)
    wasm._free(nsPtr)
    return { predictions: result, nSamples: actualNs, nRows: rows }
  }

  getSigma2Samples() {
    this.#ensureFitted()
    const wasm = getWasm()
    const ns = wasm._wl_st_bart_num_samples(this.#handle)
    if (ns === 0) return new Float64Array(0)

    const outPtr = wasm._malloc(ns * 8)
    const count = wasm._wl_st_bart_get_sigma2(this.#handle, outPtr)

    const result = new Float64Array(count)
    for (let i = 0; i < count; i++) result[i] = wasm.HEAPF64[outPtr / 8 + i]

    wasm._free(outPtr)
    return result
  }

  // --- Model I/O ---

  save() {
    this.#ensureFitted()
    const wasm = getWasm()

    const jsonPtr = wasm._wl_st_bart_to_json(this.#handle)
    if (!jsonPtr) throw new Error(`Save failed: ${getLastError()}`)
    const jsonStr = wasm.UTF8ToString(jsonPtr)
    wasm._wl_st_free_string(jsonPtr)

    const jsonBytes = new TextEncoder().encode(jsonStr)
    const typeId = this.#isRegressor
      ? 'wlearn.stochtree.regressor@1'
      : 'wlearn.stochtree.classifier@1'

    return encodeBundle(
      {
        typeId,
        params: this.getParams(),
        metadata: {
          nrClass: this.#nrClass,
          classes: this.#classes ? Array.from(this.#classes) : [],
          numFeatures: this.#numFeatures
        }
      },
      [{ id: 'model', data: jsonBytes }]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return BARTModel._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs) {
    await loadStochtree()
    const wasm = getWasm()

    const entry = toc.find(e => e.id === 'model')
    if (!entry) throw new Error('Bundle missing "model" artifact')
    const raw = blobs.subarray(entry.offset, entry.offset + entry.length)
    const jsonStr = new TextDecoder().decode(raw)

    // Pass JSON to C++ for deserialization
    const jsonBytes = new TextEncoder().encode(jsonStr + '\0')
    const jsonInPtr = wasm._malloc(jsonBytes.length)
    wasm.HEAPU8.set(jsonBytes, jsonInPtr)
    const handle = wasm._wl_st_bart_from_json(jsonInPtr)
    wasm._free(jsonInPtr)

    if (!handle) throw new Error(`Load failed: ${getLastError()}`)

    const meta = manifest.metadata || {}
    return new BARTModel(LOAD_SENTINEL, handle, {
      params: manifest.params || {},
      nrClass: meta.nrClass || 0,
      classes: meta.classes || null,
      numFeatures: meta.numFeatures || 0
    })
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    if (this.#handle) {
      try { getWasm()._wl_st_bart_free(this.#handle) } catch {}
    }

    if (this.#handleRef) this.#handleRef[0] = null
    if (leakRegistry) leakRegistry.unregister(this)

    this.#handle = null
    this.#fitted = false
  }

  // --- Params ---

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    return this
  }

  static defaultSearchSpace() {
    return {
      numTrees: { type: 'int_uniform', low: 50, high: 500 },
      numGfr: { type: 'int_uniform', low: 0, high: 20 },
      numBurnin: { type: 'int_uniform', low: 100, high: 500 },
      numSamples: { type: 'int_uniform', low: 50, high: 200 },
      alpha: { type: 'uniform', low: 0.5, high: 0.99 },
      beta: { type: 'uniform', low: 0.5, high: 3.0 },
      minSamplesLeaf: { type: 'int_uniform', low: 1, high: 10 },
      maxDepth: { type: 'int_uniform', low: 3, high: 20 }
    }
  }

  // --- Inspection ---

  get nrClass() { return this.#nrClass }
  get classes() { return this.#classes ? Int32Array.from(this.#classes) : new Int32Array(0) }
  get isFitted() { return this.#fitted && !this.#freed }
  get numFeatures() { return this.#numFeatures }

  get capabilities() {
    return {
      classifier: !this.#isRegressor,
      regressor: this.#isRegressor,
      predictProba: !this.#isRegressor,
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false,
      posteriorSamples: true
    }
  }

  get probaDim() {
    if (!this.isFitted) return 0
    if (this.#isRegressor) return 0
    return 2 // binary only for v1
  }

  // --- Private helpers ---

  #normalizeX(X) {
    // Fast path: typed matrix { data, rows, cols }
    if (X && typeof X === 'object' && !Array.isArray(X) && X.data) {
      const { data, rows, cols } = X
      if (data instanceof Float64Array) return { data, rows, cols }
      return { data: new Float64Array(data), rows, cols }
    }

    // Slow path: number[][]
    if (Array.isArray(X) && Array.isArray(X[0])) {
      const rows = X.length
      const cols = X[0].length
      const data = new Float64Array(rows * cols)
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          data[i * cols + j] = X[i][j]
        }
      }
      return { data, rows, cols }
    }

    throw new Error('X must be number[][] or { data: TypedArray, rows, cols }')
  }

  #ensureFitted(requireFit = true) {
    if (this.#freed) throw new DisposedError('BARTModel has been disposed.')
    if (requireFit && !this.#fitted) throw new NotFittedError('BARTModel is not fitted. Call fit() first.')
  }

  #detectRegression(y) {
    for (let i = 0; i < y.length; i++) {
      if (y[i] !== Math.floor(y[i])) return true
    }
    return false
  }
}

// --- Register loaders with @wlearn/core ---

register('wlearn.stochtree.classifier@1', async (m, t, b) => BARTModel._fromBundle(m, t, b))
register('wlearn.stochtree.regressor@1', async (m, t, b) => BARTModel._fromBundle(m, t, b))
