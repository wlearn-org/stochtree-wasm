const { loadStochtree, getWasm } = require('./wasm.js')
const { BARTModel } = require('./model.js')

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await BARTModel.create(params)
  model.fit(X, y)
  return model
}

// Convenience: load WLRN bundle and predict, auto-disposes model
async function predict(bundleBytes, X) {
  const model = await BARTModel.load(bundleBytes)
  const result = model.predict(X)
  model.dispose()
  return result
}

module.exports = { loadStochtree, getWasm, BARTModel, train, predict }
