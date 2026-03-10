const { loadStochtree, getWasm } = require('./wasm.js')
const { BARTModel: BARTModelImpl } = require('./model.js')
const { createModelClass } = require('@wlearn/core')

const BARTModel = createModelClass(BARTModelImpl, BARTModelImpl, { name: 'BARTModel', load: loadStochtree })

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await BARTModel.create(params)
  await model.fit(X, y)
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
