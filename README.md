# @wlearn/stochtree

WASM port of [stochtree](https://github.com/StochasticTree/stochtree) -- Bayesian Additive Regression Trees (BART) for browser and Node.js.

BART fits an ensemble of decision trees via MCMC sampling, providing full posterior inference with uncertainty quantification.

## Install

```
npm install @wlearn/stochtree
```

Requires `@wlearn/core` and `@wlearn/types`.

## Usage

### Regression

```js
const { BARTModel } = require('@wlearn/stochtree')

const model = await BARTModel.create({
  numTrees: 200,
  numSamples: 100,
  seed: 42
})

model.fit(X, y)
const predictions = model.predict(X_test)
const r2 = model.score(X_test, y_test)
```

### Classification

```js
const model = await BARTModel.create({
  objective: 'classification',
  numTrees: 200,
  numSamples: 100,
  seed: 42
})

model.fit(X, y)  // y: integer labels {0, 1}
const labels = model.predict(X_test)
const probs = model.predictProba(X_test)  // [P(0), P(1)] per sample
```

### Posterior access

```js
// Per-sample posterior predictions (nRows x nSamples)
const { predictions, nSamples, nRows } = model.predictPosterior(X_test)

// Posterior variance samples (regression)
const sigma2 = model.getSigma2Samples()
```

### Save / Load

```js
const bundle = model.save()  // Uint8Array (WLRN format)
const loaded = await BARTModel.load(bundle)

// Or via @wlearn/core registry
const { load } = require('@wlearn/core')
const model2 = await load(bundle)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `numTrees` | 200 | Trees per forest |
| `numGfr` | 10 | GFR warm-start iterations |
| `numBurnin` | 200 | MCMC burn-in iterations |
| `numSamples` | 100 | Posterior samples to keep |
| `alpha` | 0.95 | Tree prior split probability |
| `beta` | 2.0 | Tree prior depth penalty |
| `minSamplesLeaf` | 5 | Min samples per leaf |
| `maxDepth` | -1 | Max tree depth (-1 = unlimited) |
| `cutpointGrid` | 100 | Cutpoint grid size |
| `seed` | 42 | RNG seed |
| `objective` | auto | `'regression'` or `'classification'` (auto-detected from labels) |

## API

- `BARTModel.create(params)` -- async, returns unfitted model
- `model.fit(X, y)` -- train (runs full MCMC loop)
- `model.predict(X)` -- averaged posterior predictions
- `model.predictProba(X)` -- class probabilities (classification only)
- `model.score(X, y)` -- R-squared (regression) or accuracy (classification)
- `model.predictPosterior(X)` -- per-sample posterior predictions
- `model.getSigma2Samples()` -- posterior variance samples
- `model.save()` / `BARTModel.load(bytes)` -- serialization
- `model.dispose()` -- free WASM resources
- `model.getParams()` / `model.setParams(p)` -- parameter management
- `BARTModel.defaultSearchSpace()` -- AutoML search space

## Data format

X can be `number[][]` (array of rows) or `{ data: Float64Array, rows, cols }` (typed matrix, zero-copy fast path).

## Building from source

Requires [Emscripten](https://emscripten.org/) (emsdk).

```bash
git clone --recurse-submodules https://github.com/wlearn-org/stochtree-wasm
cd stochtree-wasm
bash scripts/build-wasm.sh
npm test
```

## License

MIT. Upstream stochtree is MIT licensed.
