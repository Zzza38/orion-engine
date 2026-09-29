# Migrating from 0.0.x to 0.1

Orion Engine 0.1 is a rewrite. The per-neuron `NeuralNetwork` class is replaced by a `Sequential` model of layers that
trains on mini-batches with modern optimizers, and the text `.onn` format by a binary and a JSON format. Old model
files still load. This page maps every 0.0.x API to its replacement.

- [At a glance](#at-a-glance)
- [Before and after](#before-and-after)
- [Building a network](#building-a-network)
- [Training](#training)
- [Losses](#losses)
- [Running a network](#running-a-network)
- [Saving and loading](#saving-and-loading)
- [Reading and writing weights](#reading-and-writing-weights)
- [Removed APIs](#removed-apis)
- [Behaviour changes](#behaviour-changes)

## At a glance

| 0.0.x | 0.1 |
|---|---|
| `new NeuralNetwork(strict?)` | `new Sequential({ inputSize, layers, seed? })` (always strict) |
| `addLayer(size, activation)`, where the **first** call is the input layer | `inputSize` option, then `dense(units, activation)` per layer |
| `addLayer(size)` (activation defaulted to `"relu"`) | `dense(units)` defaults to `"linear"`: always name the activation |
| `runNetwork(input)` | `predict(input)` (also accepts many rows at once) |
| `train(inputs, targets, epochs, learningRate, loss)` | `compile({ loss, optimizer })` once, then `fit(x, y, { epochs, batchSize })` |
| `backpropagate(input, target, learningRate, loss)` | `trainOnBatch([input], [target])` |
| `calculateLoss(input, target, loss)` | `evaluate([input], [target]).loss` |
| Loss `"crossEntropy"` with a `sigmoid` output | `"bce"` |
| Loss `"crossEntropy"` with a `softmax` output | `"cce"` (one-hot targets) or `"scce"` (class indices) |
| Loss `"mse"`, `"mae"` | `"mse"`, `"mae"` (unchanged) |
| `writeNetworkToFile(network, path)` | `await saveModel(model, path)` or `saveModelSync` from `@zzza38/orion-engine/node` |
| `loadNetworkFromFile(path)` | `await loadModel(path)` or `loadModelSync` from `@zzza38/orion-engine/node` |
| `writeNetwork(network)` (a string) | `serializeModel(model)` (bytes) or `serializeModel(model, { format: "json" })` (a string) |
| `loadNetwork(text)` | `deserializeModel(text)` |
| `network.layers[i].neurons[j].weights` / `.bias` | `model.getWeights()`: `kernel` [inputs, units] and `bias` [1, units] per layer |
| `loadWeightsAndBiases(layer, weights, biases)` | `model.setWeights(entries)` |
| `Activation.use(name, value)`, `ActivationDerivative` | `getActivation(name).forward(matrix)` / `.backward(…)` |
| `Loss.mse(pred, target)`, `Loss.gradient(…)` | `getLoss("mse").compute(prediction, target)` / `.gradient(…)` |

## Before and after

The XOR example from the 0.0.x README:

```ts
// Before (0.0.x)
import { loadNetworkFromFile, NeuralNetwork, writeNetworkToFile } from "@zzza38/orion-engine";

const network = new NeuralNetwork();
network.addLayer(2, "linear"); // the input layer
network.addLayer(4, "relu");
network.addLayer(1, "sigmoid");

network.train([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]], 10000, 0.3, "crossEntropy");
console.log(network.runNetwork([1, 0]));

writeNetworkToFile(network, "model.onn");
const loaded = loadNetworkFromFile("model.onn");
```

The same in 0.1:

```ts
import { dense, loadModel, saveModel, Sequential } from "@zzza38/orion-engine/node";

const model = new Sequential({ inputSize: 2, seed: 3, layers: [dense(4, "relu"), dense(1, "sigmoid")] });
model.compile({ loss: "bce", optimizer: { name: "adam", learningRate: 0.05 } });

model.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0], { epochs: 500, batchSize: 4 });
console.log(model.predict([1, 0])); // [0.999…]

await saveModel(model, "model.onn");
const loaded = await loadModel("model.onn");
```

The file helpers moved to the `@zzza38/orion-engine/node` entry point, which re-exports everything else too. The main
entry point, `@zzza38/orion-engine`, now works in browsers as well as Node.js. The package is ESM-only and requires
Node.js 22 or newer.

## Building a network

In 0.0.x the first `addLayer` call described the input: its size was the number of inputs and its activation was
ignored. In 0.1 the input is not a layer. Pass its size as `inputSize` (or let `fit`/`predict` infer it), and list only
the layers that compute something:

```ts
import { dense, Sequential } from "@zzza38/orion-engine";

// 0.0.x: addLayer(3, "linear"); addLayer(16, "relu"); addLayer(16, "relu"); addLayer(2, "softmax");
const model = new Sequential({ inputSize: 3, layers: [dense(16, "relu"), dense(16, "relu")] });
model.add(dense(2, "softmax")); // layers can also be appended one at a time, like addLayer
```

Every 0.0.x activation exists in 0.1 under the same name (`linear`, `sigmoid`, `tanh`, `relu`, `leakyRelu`, `elu`,
`softmax`, `swish`), and there are more (`gelu`, `selu`, `mish`, …). Note the different default: `addLayer(size)` used
`"relu"`, `dense(units)` is linear.

**Initialization** differs too. 0.0.x drew kernels uniformly from ±√(6 / fanIn) and biases from ±0.1 with
`Math.random()`; 0.1 uses Glorot-uniform kernels and zero biases, drawn from the model's seeded generator. To mimic the
old scheme:

```ts
import { dense } from "@zzza38/orion-engine";

dense(16, {
    activation: "relu",
    kernelInitializer: "heUniform", // U(±√(6 / fanIn)), as in 0.0.x
    biasInitializer: { name: "randomUniform", minval: -0.1, maxval: 0.1 },
});
```

## Training

`train(inputs, targets, epochs, learningRate, loss)` did three things that 0.1 separates:

1. `compile` chooses the loss and the optimizer (and optional metrics) once.
2. `fit` trains, with options for batch size, validation data, callbacks and logging.
3. `fit` returns a `History` with the loss of every epoch, instead of the final epoch's average loss.

0.0.x updated the weights after every sample with plain gradient descent. The closest 0.1 equivalent is SGD with a
batch size of 1:

```ts
import { dense, Sequential } from "@zzza38/orion-engine";

const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
const targets = [[0], [1], [1], [0]]; // 0.0.x-style targets (one array per sample) still work

const model = new Sequential({ inputSize: 2, seed: 3, layers: [dense(4, "tanh"), dense(1, "sigmoid")] });
model.compile({ loss: "bce", optimizer: { name: "sgd", learningRate: 0.3 } });
const history = model.fit(inputs, targets, { epochs: 2000, batchSize: 1 });
console.log(history.last("loss")); // what train() used to return
```

You will usually do better with the defaults: Adam (`optimizer: "adam"`, or a config with a `learningRate`) and a
batch size of 16 to 64, which is also several times faster.

- `backpropagate(input, target, learningRate, loss)` is `trainOnBatch([input], [target])`; set the learning rate with
  `compile`, or change `model.optimizer.learningRate`.
- `calculateLoss(input, target, loss)` is `model.evaluate([input], [target]).loss`.
- Targets no longer need to be wrapped per sample: for one output, `[0, 1, 1, 0]` works, and class indices can be
  passed directly with the `"scce"` loss.

## Losses

| 0.0.x | 0.1 | Notes |
|---|---|---|
| `"mse"` | `"mse"` (`meanSquaredError`) | Same values and gradients |
| `"mae"` | `"mae"` (`meanAbsoluteError`) | Same values and gradients |
| `"crossEntropy"` + `sigmoid` output | `"bce"` (`binaryCrossentropy`) | Same gradient. The reported loss now includes the `−(1 − y)·ln(1 − p)` term, so values differ |
| `"crossEntropy"` + `softmax` output | `"cce"` (`categoricalCrossentropy`) with one-hot targets, or `"scce"` with class indices | Loss and gradient are summed over the classes instead of averaged, so they are k times larger for k classes; divide an old SGD learning rate by k to match |
| `"crossEntropy"` + another output | `"bce"` with a `sigmoid` output | Cross-entropy needs probabilities: end binary and multi-label models with `sigmoid` |

An unknown loss name, including `"crossEntropy"`, throws a `ValidationError` that lists the valid names.

## Running a network

`runNetwork(input)` becomes `predict(input)`. One sample in gives one output array back, as before; rows of samples
give rows of outputs, computed in one batched pass:

```ts
import { dense, Sequential } from "@zzza38/orion-engine";

const model = new Sequential({ inputSize: 2, layers: [dense(3, "relu"), dense(1, "sigmoid")] });
const one = model.predict([1, 0]); // number[]: one value per output unit
const many = model.predict([[1, 0], [0, 1]]); // number[][]: one row per sample
console.log(one, many);
```

`predict` needs no `compile`. Dropout and batch normalization, which 0.0.x did not have, automatically switch to
inference behaviour.

## Saving and loading

0.1 writes two new formats, both described in [format.md](format.md):

- **Binary** (the default): compact and checksummed, with float32 weights (or float64 with `precision: "float64"`).
- **JSON** (`format: "json"`, or a `.json` path with `saveModel`): human-readable and bit-exact.

**Old `.onn` files load unchanged.** `loadModel`, `loadModelSync` and `deserializeModel` recognize the 0.0.x text format
and convert it: each layer after the input becomes a `dense` layer with the same activation and weights, so the loaded
model computes exactly what the old one did.

```ts
import { deserializeModel } from "@zzza38/orion-engine";

// The contents of a 0.0.x .onn file: an XOR network (2 inputs, 3 tanh units, 1 sigmoid output).
const text = "2:linear:3:tanh:1:sigmoid\n-2.212:-2.238:-0.051|3.532:-4.58:-1.408|-4.545:3.465:-1.361|-4.654:8.302:8.303:3.817";
const model = deserializeModel(text);
console.log(model.predict([[0, 0], [0, 1], [1, 0], [1, 1]])); // ≈ [[0], [1], [1], [0]]
```

To upgrade a file on disk, load it and save it again:

```ts
import { writeFileSync } from "node:fs";
import { loadModel, saveModel } from "@zzza38/orion-engine/node";

writeFileSync("old-model.onn", "2:relu:2:swish\n0.71:-0.2:0.19|-1.82:0.95:0.97"); // stands in for your 0.0.x file
const model = await loadModel("old-model.onn");
await saveModel(model, "old-model.onn", { precision: "float64" }); // now the binary format, same weights
```

Converted models have no training configuration (0.0.x files did not store one), so call `compile` before training
them further. 0.0.x cannot read the new formats. If you still have a live 0.0.x `NeuralNetwork` object (for example
with the old version installed under an npm alias), `deserializeModel(writeNetwork(network))` converts it in memory.

## Reading and writing weights

0.0.x stored weights per neuron: `layers[l].neurons[j].weights[i]` connected input `i` to neuron `j`. 0.1 stores one
kernel matrix per layer with shape `[inputs, units]`, so that weight is `kernel[i][j]`: the old per-neuron arrays are
the kernel's **columns**. Biases are a `[1, units]` matrix. Weights are named after their layer (`dense_1/kernel`,
`dense_1/bias`, …) and stored row-major in `data`:

```ts
import { dense, Sequential } from "@zzza38/orion-engine";

const model = new Sequential({ inputSize: 2, seed: 1, layers: [dense(3, "tanh"), dense(1, "sigmoid")] });

// 0.0.x-style weights for the first layer: one array of input weights per neuron, plus biases.
const neuronWeights = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]; // 3 neurons × 2 inputs
const biases = [0.01, 0.02, 0.03];

const kernel = new Array<number>(2 * 3);
neuronWeights.forEach((weights, j) => weights.forEach((w, i) => (kernel[i * 3 + j] = w))); // transpose into [2, 3]

const entries = model.getWeights().map((entry) => {
    if (entry.name === "dense_1/kernel") return { ...entry, data: kernel };
    if (entry.name === "dense_1/bias") return { ...entry, data: biases };
    return entry;
});
model.setWeights(entries); // replaces loadWeightsAndBiases(1, neuronWeights, biases)
```

`setWeights` validates every entry's name, shape and values and changes nothing if any is wrong, where 0.0.x logged a
warning (or threw in strict mode).

## Removed APIs

| 0.0.x export | Replacement |
|---|---|
| `NeuralNetwork` | `Sequential` |
| `Activation` (static `sigmoid`, `relu`, …, `use`) | `getActivation(name)`, which returns an object with batched `forward` / `backward` over matrices |
| `ActivationDerivative` | `getActivation(name).backward(z, a, gradOutput)` |
| `Loss` (static `mse`, `mae`, `crossEntropy`, `gradient`) | `getLoss(name)`, with `compute` and `gradient` over matrices |
| `loadNetwork`, `loadNetworkFromFile`, `writeNetwork`, `writeNetworkToFile` | `deserializeModel`, `loadModel`, `serializeModel`, `saveModel` |
| Types `NeuralNetworkActivationFunction`, `NeuralNetworkLossType`, `NeuralNetworkLayer`, `NeuralNetworkLayerType`, `NeuralNetworkModel`, `NeuralNetworkNeuron` | `ActivationName`, `LossName` / `LossAlias`, `Layer`, `LayerConfig`, `ModelArtifact`, `WeightEntry` |
| The `strict` constructor flag | Always strict |

## Behaviour changes

- **Errors instead of warnings.** Wrong input sizes, non-numeric values and bad weights throw typed errors
  (`ShapeError`, `ValidationError`, …) with an explanation. 0.0.x logged a warning and zero-filled missing inputs unless
  `strict` was set. `addLayer` returned `false` for an invalid size; the new factories throw.
- **Seeded randomness.** Initialization, shuffling and dropout use the model's seed, so runs are reproducible. 0.0.x
  used `Math.random()`.
- **Exact softmax gradients.** With a `softmax` output and a loss other than cross-entropy, 0.0.x used only the diagonal
  of the softmax Jacobian; 0.1 backpropagates through the full Jacobian.
- **Divergence is detected.** If the loss becomes `NaN` or infinite, training stops with a `TrainingError` that suggests
  a fix, instead of silently producing `NaN` weights. (The 0.0.x XOR example [above](#before-and-after) often ends
  that way: it saves `NaN` weights, and loading the file then fails.)
