# Changelog

## 0.2.0

- Wrap BARTModel with `createModelClass` for unified task detection
- Add `task` parameter: `'classification'` or `'regression'`, auto-detected from labels if omitted

## 0.1.0

Initial release.

- WASM port of stochtree (BART) via Emscripten
- BARTModel class with full wlearn estimator contract
- Regression with R-squared scoring
- Binary classification via Albert-Chib probit BART
- Posterior access: predictPosterior(), getSigma2Samples()
- Save/load via WLRN bundle format
- Registry loaders: wlearn.stochtree.regressor@1, wlearn.stochtree.classifier@1
- 27 tests passing
