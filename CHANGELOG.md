# Changelog

All notable changes to Orion Engine are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the major version is 0, breaking changes bump the
minor version.

## [Unreleased]

## [0.1.0] - 2026-09-29

A complete rewrite: a Keras-style, mini-batch training library for Node.js and the browser. It is not source-compatible
with 0.0.x; [docs/migration.md](https://github.com/Zzza38/orion-engine/blob/main/docs/migration.md) maps every old API to its replacement, and 0.0.x model files still
load.

### Added

- `Sequential` model with `compile`, `fit`, `fitAsync`, `evaluate`, `predict`, `trainOnBatch`, `summary`, `clone`,
  `getWeights` / `setWeights` and `toArtifact` / `fromArtifact`. The input size can be inferred from the first call.
- The `initialEpoch` fit option resumes training across several `fit` calls without restarting epoch numbering or
  learning-rate schedules.
- One `seed` drives weight initialization, shuffling and dropout, for bit-identical, reproducible training runs.
- Layers: `Dense` (initializers, optional bias, L1/L2 kernel regularization), `Dropout`, `BatchNormalization`
  (single-sample batches are kept out of its moving statistics) and `ActivationLayer`, with the factories `dense`,
  `dropout`, `batchNormalization` and `activation`. Custom layers can extend `BaseLayer` and be registered for loading
  with `registerLayer`.
- 15 activations: `linear`, `sigmoid`, `tanh`, `relu`, `relu6`, `leakyRelu`, `elu`, `selu`, `gelu`, `swish`, `mish`,
  `softplus`, `softsign`, `hardSigmoid` and `softmax`.
- Losses `meanSquaredError`, `meanAbsoluteError`, `huber`, `binaryCrossentropy`, `categoricalCrossentropy` and
  `sparseCategoricalCrossentropy`, with the aliases `mse`, `mae`, `bce`, `cce` and `scce`. Sigmoid and softmax outputs
  train with fused, numerically stable cross-entropy gradients, exact also for soft and multi-hot targets.
- Optimizers `sgd` (momentum, Nesterov, weight decay), `adam` (AMSGrad), `adamw`, `rmsprop` (momentum, centered) and
  `adagrad`, all with `clipNorm` / `clipValue` gradient clipping.
- Metrics `accuracy` (binary with a `bce` loss, including multi-label outputs; otherwise binary, categorical or sparse,
  chosen from the shapes), `binaryAccuracy`, `categoricalAccuracy`, `sparseCategoricalAccuracy`, `meanSquaredError`,
  `meanAbsoluteError` and `rootMeanSquaredError`.
- Initializers `glorotUniform`, `glorotNormal`, `heUniform`, `heNormal`, `lecunUniform`, `lecunNormal`, `zeros`, `ones`,
  `constant`, `randomUniform` and `randomNormal`.
- Callbacks `earlyStopping`, `learningRateScheduler`, `reduceLROnPlateau` and `progressLogger`, custom callbacks with
  batch and epoch hooks, and a `History` of every epoch's logs, including validation metrics (`validationSplit`,
  `validationData`).
- Learning-rate schedules `constantSchedule`, `stepDecay`, `exponentialDecay`, `cosineDecay`, `piecewiseConstant` and
  `linearWarmup`.
- `fitAsync`, which keeps pages responsive, awaits async callbacks and can be cancelled with an `AbortSignal`.
- Data utilities: `StandardScaler` (constant features transform to 0, as in scikit-learn), `MinMaxScaler` (both
  serializable), `oneHot`, `argmax`, `trainTestSplit` and `shuffleTogether`.
- Model files: a compact, checksummed binary format (float32 or float64 weights) and a lossless JSON format, specified
  in [docs/format.md](https://github.com/Zzza38/orion-engine/blob/main/docs/format.md). `serializeModel` / `deserializeModel` work everywhere; `saveModel`, `loadModel`,
  `saveModelSync` and `loadModelSync` in the new `@zzza38/orion-engine/node` entry point handle files. Low-level codecs
  (`encodeBinary`, `decodeJson`, `detectFormat`, …) are exported too.
- Import of 0.0.x `.onn` text models (`decodeLegacyOnn`, and automatically in `deserializeModel` / `loadModel`).
- Typed errors with actionable messages: `OrionError`, `ValidationError`, `ShapeError`, `TrainingError` and
  `SerializationError` (all accept `{ cause }`). Training stops with a `TrainingError` when the loss becomes `NaN` or
  infinite; `deserializeModel` reports files that describe an impossible model (for example an unregistered layer type)
  as a `SerializationError` with the original error as its `cause`.
- `Matrix` (row-major, `Float64Array`-backed) with the matrix operations the layers use, and a seedable `Random`.
- `VERSION` export.
- An [interactive playground](https://zzza38.github.io/orion-engine/) that trains networks in the browser.
- Documentation ([guide](https://github.com/Zzza38/orion-engine/blob/main/docs/guide.md), [API reference](https://github.com/Zzza38/orion-engine/blob/main/docs/api.md), [migration guide](https://github.com/Zzza38/orion-engine/blob/main/docs/migration.md)),
  runnable [examples](https://github.com/Zzza38/orion-engine/tree/main/examples) (`pnpm examples`) and benchmarks (`pnpm bench`).

### Changed

- The package is ESM-only, with two entry points: `@zzza38/orion-engine` (browser-safe) and
  `@zzza38/orion-engine/node` (everything, plus file helpers). Node.js 22 or newer is required.
- Training uses mini-batches (32 samples by default) and Adam by default, instead of per-sample gradient descent.
  On typical layer sizes, training is about 5× and single-sample prediction about 4× faster than in 0.0.2.
- The input is no longer a layer: pass `inputSize` instead of a first `addLayer` call.
- Dense layers default to a linear activation (`addLayer` defaulted to `relu`).
- Kernels are initialized Glorot-uniform and biases to zero, from the model's seeded generator, instead of with
  `Math.random()`.
- The `crossEntropy` loss is replaced by `bce` (sigmoid outputs) and `cce` / `scce` (softmax outputs).
- Invalid inputs, shapes and weights throw typed errors instead of logging warnings and zero-filling; there is no
  `strict` flag any more.
- Models are saved in the new binary or JSON formats. The 0.0.x text format is import-only.

### Removed

- The `NeuralNetwork` class (`addLayer`, `runNetwork`, `train`, `backpropagate`, `calculateLoss`,
  `loadWeightsAndBiases`, `fixWeights`, `layers`). Use `Sequential`.
- The static `Activation`, `ActivationDerivative` and `Loss` classes. Use `getActivation` and `getLoss`.
- `loadNetwork`, `loadNetworkFromFile`, `writeNetwork` and `writeNetworkToFile`. Use `deserializeModel`, `loadModel`,
  `serializeModel` and `saveModel`.
- The types `NeuralNetworkActivationFunction`, `NeuralNetworkLayer`, `NeuralNetworkLayerType`, `NeuralNetworkLossType`,
  `NeuralNetworkModel` and `NeuralNetworkNeuron`.

### Fixed

- Softmax layers backpropagate through the full Jacobian; 0.0.x used only its diagonal with losses other than
  cross-entropy.
- Diverging training is detected instead of silently producing `NaN` weights.

## [0.0.2] - 2026-06-21

First release on npm as `@zzza38/orion-engine`: a `NeuralNetwork` class with per-layer activations, per-sample
backpropagation with the `mse`, `mae` and `crossEntropy` losses, and saving and loading networks as `.onn` text files.

[Unreleased]: https://github.com/Zzza38/orion-engine/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Zzza38/orion-engine/releases/tag/v0.1.0
[0.0.2]: https://www.npmjs.com/package/@zzza38/orion-engine/v/0.0.2
