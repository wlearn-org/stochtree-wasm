let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    if (err.stack) {
      const lines = err.stack.split('\n').slice(1, 3)
      for (const line of lines) console.log(`        ${line.trim()}`)
    }
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

// Deterministic data generation (LCG PRNG)
function lcg(seed) {
  let s = seed
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    return s / 0x7fffffff
  }
}

function makeRegressionData(n, seed = 7) {
  const rng = lcg(seed)
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const x1 = rng() * 10
    const x2 = rng() * 10
    const noise = rng() * 0.5
    X.push([x1, x2])
    y.push(2 * x1 + 3 * x2 + noise)
  }
  return { X, y }
}

function makeBinaryData(n, seed = 13) {
  const rng = lcg(seed)
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const x1 = rng() * 10
    const x2 = rng() * 10
    X.push([x1, x2])
    y.push(x1 + x2 > 10 ? 1 : 0)
  }
  return { X, y }
}

// Common BART params (small for fast tests)
const fastParams = {
  numTrees: 30, numGfr: 5, numBurnin: 20, numSamples: 20, seed: 42
}

async function main() {
// ============================================================
// WASM loading
// ============================================================
console.log('\n=== WASM Loading ===')

const { loadStochtree } = require('../src/wasm.js')
const wasm = await loadStochtree()

await test('WASM module loads', async () => {
  assert(wasm, 'wasm module is null')
  assert(typeof wasm.ccall === 'function', 'ccall not available')
})

await test('get_last_error returns string', async () => {
  const err = wasm.ccall('wl_st_get_last_error', 'string', [], [])
  assert(typeof err === 'string', `expected string, got ${typeof err}`)
})

// ============================================================
// BARTModel basics
// ============================================================
console.log('\n=== BARTModel ===')

const { BARTModel } = require('../src/model.js')

await test('create() returns model', async () => {
  const model = await BARTModel.create()
  assert(model, 'model is null')
  assert(!model.isFitted, 'should not be fitted yet')
  model.dispose()
})

await test('unfitted model throws on predict', async () => {
  const model = await BARTModel.create()
  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict before fit should throw')
  model.dispose()
})

// ============================================================
// Regression
// ============================================================
console.log('\n=== Regression ===')

await test('regression fit and predict', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(200)

  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  assert(model.capabilities.regressor, 'should be regressor')
  assert(model.numFeatures === 2, `expected 2 features, got ${model.numFeatures}`)

  const preds = model.predict(X)
  assert(preds instanceof Float64Array, 'predictions should be Float64Array')
  assert(preds.length === 200, `expected 200 predictions, got ${preds.length}`)

  model.dispose()
})

await test('regression score (R2)', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(200)
  model.fit(X, y)

  const r2 = model.score(X, y)
  assert(typeof r2 === 'number', 'score should be a number')
  assert(r2 > 0.8, `R-squared ${r2.toFixed(4)} too low`)

  model.dispose()
})

await test('regression with typed matrix fast path', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(100)

  const data = new Float64Array(100 * 2)
  for (let i = 0; i < 100; i++) {
    data[i * 2] = X[i][0]
    data[i * 2 + 1] = X[i][1]
  }

  model.fit({ data, rows: 100, cols: 2 }, y)
  const preds = model.predict({ data, rows: 100, cols: 2 })
  assert(preds.length === 100, `expected 100 predictions, got ${preds.length}`)

  const r2 = model.score({ data, rows: 100, cols: 2 }, y)
  assert(r2 > 0.7, `R2 ${r2.toFixed(4)} too low with typed matrix`)

  model.dispose()
})

// ============================================================
// Binary classification
// ============================================================
console.log('\n=== Binary Classification ===')

await test('binary classification fit and predict', async () => {
  const model = await BARTModel.create({
    ...fastParams, objective: 'classification'
  })
  const { X, y } = makeBinaryData(200)

  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  assert(model.capabilities.classifier, 'should be classifier')
  assert(model.nrClass === 2, `expected 2 classes, got ${model.nrClass}`)

  const classes = model.classes
  assert(classes.length === 2, `expected 2 classes`)
  assert(classes[0] === 0 && classes[1] === 1, 'expected classes [0, 1]')

  const preds = model.predict(X)
  assert(preds instanceof Float64Array, 'predictions should be Float64Array')
  assert(preds.length === 200, `expected 200 predictions, got ${preds.length}`)

  // All predictions should be valid class labels
  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] === 0 || preds[i] === 1,
      `invalid prediction at ${i}: ${preds[i]}`)
  }

  model.dispose()
})

await test('binary classification accuracy', async () => {
  const model = await BARTModel.create({
    ...fastParams, objective: 'classification'
  })
  const { X, y } = makeBinaryData(200)
  model.fit(X, y)

  const acc = model.score(X, y)
  assert(acc > 0.7, `accuracy ${acc.toFixed(4)} too low for separable data`)

  model.dispose()
})

await test('auto-detect classification from integer labels', async () => {
  // Don't pass objective: should auto-detect from integer labels
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeBinaryData(200)
  model.fit(X, y)

  assert(model.capabilities.classifier, 'should auto-detect as classifier')
  assert(model.nrClass === 2, `expected 2 classes, got ${model.nrClass}`)

  model.dispose()
})

// ============================================================
// Probability estimates
// ============================================================
console.log('\n=== Probability Estimates ===')

await test('predictProba returns valid probabilities', async () => {
  const model = await BARTModel.create({
    ...fastParams, objective: 'classification'
  })
  const { X, y } = makeBinaryData(100)
  model.fit(X, y)

  const probs = model.predictProba(X)
  assert(probs.length === 200, `expected 200 probabilities (100*2), got ${probs.length}`)

  for (let r = 0; r < 100; r++) {
    const p0 = probs[r * 2]
    const p1 = probs[r * 2 + 1]
    assert(p0 >= 0 && p0 <= 1, `P(0) out of [0,1]: ${p0}`)
    assert(p1 >= 0 && p1 <= 1, `P(1) out of [0,1]: ${p1}`)
    assertClose(p0 + p1, 1.0, 1e-6, `row ${r} probabilities sum to ${p0 + p1}`)
  }

  model.dispose()
})

await test('predictProba throws for regression', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(50)
  model.fit(X, y)

  let threw = false
  try { model.predictProba(X) } catch { threw = true }
  assert(threw, 'predictProba should throw for regression')

  model.dispose()
})

// ============================================================
// Posterior access
// ============================================================
console.log('\n=== Posterior Access ===')

await test('predictPosterior returns per-sample predictions', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(100)
  model.fit(X, y)

  const post = model.predictPosterior(X.slice(0, 5))
  assert(post.nSamples === fastParams.numSamples,
    `expected ${fastParams.numSamples} samples, got ${post.nSamples}`)
  assert(post.nRows === 5, `expected 5 rows, got ${post.nRows}`)
  assert(post.predictions instanceof Float64Array, 'predictions should be Float64Array')
  assert(post.predictions.length === 5 * post.nSamples,
    `expected ${5 * post.nSamples} values, got ${post.predictions.length}`)

  // Average of posterior should be close to mean prediction
  const meanPred = model.predict(X.slice(0, 5))
  for (let i = 0; i < 5; i++) {
    let sum = 0
    for (let s = 0; s < post.nSamples; s++) {
      sum += post.predictions[i * post.nSamples + s]
    }
    const avg = sum / post.nSamples
    assertClose(avg, meanPred[i], 1e-6,
      `row ${i}: posterior mean ${avg} != predict ${meanPred[i]}`)
  }

  model.dispose()
})

await test('getSigma2Samples returns variance posterior', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(100)
  model.fit(X, y)

  const sigma2 = model.getSigma2Samples()
  assert(sigma2 instanceof Float64Array, 'sigma2 should be Float64Array')
  assert(sigma2.length === fastParams.numSamples,
    `expected ${fastParams.numSamples} samples, got ${sigma2.length}`)

  // All values should be positive
  for (let i = 0; i < sigma2.length; i++) {
    assert(sigma2[i] > 0, `sigma2[${i}] = ${sigma2[i]} should be positive`)
  }

  model.dispose()
})

// ============================================================
// Save / Load
// ============================================================
console.log('\n=== Save / Load ===')

const { decodeBundle, load: coreLoad } = require('@wlearn/core')

await test('save produces WLRN bundle', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(100)
  model.fit(X, y)

  const buf = model.save()
  assert(buf instanceof Uint8Array, 'save should return Uint8Array')
  assert(buf.length > 0, 'saved model should not be empty')

  // Verify WLRN magic
  assert(buf[0] === 0x57, 'bad magic[0]')
  assert(buf[1] === 0x4c, 'bad magic[1]')
  assert(buf[2] === 0x52, 'bad magic[2]')
  assert(buf[3] === 0x4e, 'bad magic[3]')

  const { manifest, toc } = decodeBundle(buf)
  assert(manifest.typeId === 'wlearn.stochtree.regressor@1',
    `expected regressor typeId, got ${manifest.typeId}`)
  assert(toc.length === 1, `expected 1 TOC entry, got ${toc.length}`)
  assert(toc[0].id === 'model', `expected TOC entry "model", got ${toc[0].id}`)

  model.dispose()
})

await test('save classifier uses classifier typeId', async () => {
  const model = await BARTModel.create({
    ...fastParams, objective: 'classification'
  })
  const { X, y } = makeBinaryData(100)
  model.fit(X, y)

  const buf = model.save()
  const { manifest } = decodeBundle(buf)
  assert(manifest.typeId === 'wlearn.stochtree.classifier@1',
    `expected classifier typeId, got ${manifest.typeId}`)

  model.dispose()
})

await test('save and load regression round-trip', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(100)
  model.fit(X, y)

  const preds1 = model.predict(X)
  const buf = model.save()

  const model2 = await BARTModel.load(buf)
  assert(model2.isFitted, 'loaded model should be fitted')

  const preds2 = model2.predict(X)
  assert(preds1.length === preds2.length, 'prediction length mismatch')
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i],
      `prediction ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  model.dispose()
  model2.dispose()
})

await test('save and load classification round-trip', async () => {
  const model = await BARTModel.create({
    ...fastParams, objective: 'classification'
  })
  const { X, y } = makeBinaryData(100)
  model.fit(X, y)

  const preds1 = model.predict(X)
  const probs1 = model.predictProba(X)
  const buf = model.save()

  const model2 = await BARTModel.load(buf)
  assert(model2.nrClass === 2, `loaded nrClass = ${model2.nrClass}`)
  assert(model2.capabilities.classifier, 'loaded model should be classifier')

  const preds2 = model2.predict(X)
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i],
      `prediction ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  const probs2 = model2.predictProba(X)
  for (let i = 0; i < probs1.length; i++) {
    assertClose(probs1[i], probs2[i], 1e-10,
      `proba ${i}: ${probs1[i]} !== ${probs2[i]}`)
  }

  model.dispose()
  model2.dispose()
})

// ============================================================
// Registry dispatch
// ============================================================
console.log('\n=== Registry Dispatch ===')

await test('core.load() dispatches to BART loader', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(60)
  model.fit(X, y)

  const preds1 = model.predict(X)
  const buf = model.save()

  const model2 = await coreLoad(buf)
  assert(model2.isFitted, 'registry-loaded model should be fitted')

  const preds2 = model2.predict(X)
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i],
      `core.load prediction ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  model.dispose()
  model2.dispose()
})

// ============================================================
// Params
// ============================================================
console.log('\n=== Params ===')

await test('getParams / setParams', async () => {
  const model = await BARTModel.create({ numTrees: 100, alpha: 0.9 })

  const params = model.getParams()
  assert(params.numTrees === 100, `expected 100, got ${params.numTrees}`)
  assert(params.alpha === 0.9, `expected 0.9, got ${params.alpha}`)

  model.setParams({ numTrees: 50 })
  const params2 = model.getParams()
  assert(params2.numTrees === 50, `expected 50 after setParams, got ${params2.numTrees}`)

  model.dispose()
})

await test('defaultSearchSpace returns object', async () => {
  const space = BARTModel.defaultSearchSpace()
  assert(space, 'search space is null')
  assert(space.numTrees, 'missing numTrees in search space')
  assert(space.alpha, 'missing alpha in search space')
  assert(space.numSamples, 'missing numSamples in search space')
})

// ============================================================
// Resource management
// ============================================================
console.log('\n=== Resource Management ===')

await test('dispose is idempotent', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(50)
  model.fit(X, y)
  model.dispose()
  model.dispose() // should not throw
})

await test('throws after dispose', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(50)
  model.fit(X, y)
  model.dispose()

  let threw = false
  try { model.predict(X) } catch { threw = true }
  assert(threw, 'predict after dispose should throw')
})

await test('throws before fit', async () => {
  const model = await BARTModel.create()

  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict before fit should throw')

  model.dispose()
})

await test('refit does not leak', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(50)
  model.fit(X, y)

  const r2a = model.score(X, y)
  model.fit(X, y) // refit

  const preds = model.predict(X)
  assert(preds.length === 50, 'should predict after refit')

  const r2b = model.score(X, y)
  assert(r2b > 0.5, `R2 after refit ${r2b.toFixed(4)} too low`)

  model.dispose()
})

// ============================================================
// Capabilities
// ============================================================
console.log('\n=== Capabilities ===')

await test('capabilities reflect task type', async () => {
  const cls = await BARTModel.create({
    ...fastParams, objective: 'classification'
  })
  const { X: Xc, y: yc } = makeBinaryData(80)
  cls.fit(Xc, yc)
  assert(cls.capabilities.classifier === true, 'should be classifier')
  assert(cls.capabilities.regressor === false, 'should not be regressor')
  assert(cls.capabilities.predictProba === true, 'should support predictProba')
  assert(cls.capabilities.posteriorSamples === true, 'should support posteriorSamples')
  cls.dispose()

  const reg = await BARTModel.create(fastParams)
  const { X: Xr, y: yr } = makeRegressionData(80)
  reg.fit(Xr, yr)
  assert(reg.capabilities.regressor === true, 'should be regressor')
  assert(reg.capabilities.classifier === false, 'should not be classifier')
  assert(reg.capabilities.predictProba === false, 'should not support predictProba')
  assert(reg.capabilities.posteriorSamples === true, 'should support posteriorSamples')
  reg.dispose()
})

// ============================================================
// Input coercion
// ============================================================
console.log('\n=== Input Coercion ===')

await test('typed matrix fast path', async () => {
  const model = await BARTModel.create(fastParams)
  const { X, y } = makeRegressionData(60)

  const data = new Float64Array(60 * 2)
  for (let i = 0; i < 60; i++) {
    data[i * 2] = X[i][0]
    data[i * 2 + 1] = X[i][1]
  }

  model.fit({ data, rows: 60, cols: 2 }, y)
  const preds = model.predict({ data, rows: 60, cols: 2 })
  assert(preds.length === 60, `expected 60 predictions, got ${preds.length}`)

  model.dispose()
})

// ============================================================
// Summary
// ============================================================
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`)
process.exit(failed > 0 ? 1 : 0)
}

main()
