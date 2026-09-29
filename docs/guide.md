# Orion Engine guide

This guide explains how Orion Engine works and how to get good results with it. For a first taste, start with the
[quick start in the README](../README.md#quick-start); for exact signatures and defaults, see the
[API reference](api.md).

- [Concepts](#concepts)
  - [Samples, features and matrices](#samples-features-and-matrices)
  - [Layers and the Sequential model](#layers-and-the-sequential-model)
  - [Compiling: loss, optimizer, metrics](#compiling-loss-optimizer-metrics)
  - [Training with fit](#training-with-fit)
  - [Logs and History](#logs-and-history)
  - [Callbacks](#callbacks)
  - [Evaluating and predicting](#evaluating-and-predicting)
- [Preparing data](#preparing-data)
- [Training tips](#training-tips)
- [Training in the browser: fitAsync](#training-in-the-browser-fitasync)
- [Saving models](#saving-models)
- [Troubleshooting](#troubleshooting)

## Concepts

### Samples, features and matrices

A dataset is a table: one **row per sample**, one **column per feature**. Orion accepts it in three shapes, together
called `MatrixLike`:

| You pass | Meaning |
|---|---|
| `number[][]` | Rows of samples: `[[0.1, 2.3], [0.4, 1.9], …]` |
| `number[]` | As input (`x`): **one** sample. As a target (`y`): one value per sample |
| `Matrix` | A dense, row-major matrix backed by a `Float64Array` |

The flat-array rule matters for single-feature data: `[0.1, 0.2, 0.3]` as `x` is one sample with three features. Write
`[[0.1], [0.2], [0.3]]` for three samples with one feature each.

`Matrix` is what the layers compute with. Converting arrays is cheap, but if you call `fit` or `predict` many times on
the same large dataset, convert it once with `Matrix.from(rows)` and pass the matrix:

```ts
import { Matrix } from "@zzza38/orion-engine";

const m = Matrix.from([[1, 2], [3, 4], [5, 6]]);
console.log(m.shape, m.get(2, 1)); // [3, 2] 6
console.log(m.row(1)); // Float64Array [3, 4] (a live view, not a copy)
console.log(m.toArray()); // [[1, 2], [3, 4], [5, 6]]
```

Internally a batch is a `Matrix` of shape `[batchSize, features]`, a `Dense` kernel is `[inputs, units]` and a bias is
`[1, units]`, so a forward pass is `activation(x · kernel + bias)`.

### Layers and the Sequential model

A `Sequential` model is a stack of layers, each feeding the next. Layers are created with factory functions:

```ts
import { activation, batchNormalization, dense, dropout, Sequential } from "@zzza38/orion-engine";

const model = new Sequential({
    inputSize: 10, // number of input features
    seed: 1, // makes initialization, shuffling and dropout reproducible
    name: "classifier", // shown by summary() and stored in saved models
    layers: [dense(64, "relu"), dropout(0.2), dense(32), batchNormalization(), activation("relu")],
});
model.add(dense(3, "softmax")); // add() appends a layer and returns the model, so calls can be chained
console.log(model.summary());
```

- **There is no input layer.** The first layer's input size is `inputSize`. If you leave `inputSize` out, the model
  infers it from the first `fit`, `predict`, `evaluate` or `trainOnBatch` call (or call `build(inputSize)` yourself).
  Weights are created once the input size is known.
- **Layer names** default to `<type>_<n>` (`dense_1`, `dropout_1`, `batch_normalization_1`, …). Pass `{ name }` to
  choose your own; names must be unique within a model. Weights are named after their layer: `dense_1/kernel`,
  `dense_1/bias`, `batch_normalization_1/movingMean`, …
- **Inspecting a model:** `summary()` prints a table of layers, output sizes and parameter counts; `layers`,
  `inputSize`, `outputSize`, `parameterCount`, `built` and `compiled` are properties; `getWeights()` and `setWeights()`
  copy weights out and in; `clone()` makes an independent deep copy.

`dense` takes the activation as its second argument, or an options object for everything else:

```ts
import { dense } from "@zzza38/orion-engine";

dense(8, "relu");
dense(8, { name: "hidden", activation: { name: "leakyRelu", alpha: 0.2 } });
dense(64, { activation: "relu", kernelInitializer: "heNormal", kernelRegularizer: { l2: 1e-4 } });
dense(64, { useBias: false }); // no bias (e.g. right before batch normalization)
```

### Compiling: loss, optimizer, metrics

`compile` chooses what to minimize, how, and what else to report:

```ts
model.compile({
    loss: "bce", // required
    optimizer: { name: "adam", learningRate: 0.01 }, // default: "adam" with its defaults (learning rate 0.001)
    metrics: ["accuracy"], // optional, reported in the logs
});
```

The loss and the optimizer accept a name, a config object with options, or an instance; metrics take a name or a
`Metric` object. Match the loss to the last layer:

| Task | Last layer | Loss | Targets (`y`) |
|---|---|---|---|
| Regression | `dense(n)` (linear) | `"mse"`, `"mae"` or `{ name: "huber", delta }` | Numbers, one column per output |
| Binary classification | `dense(1, "sigmoid")` | `"bce"` | `0` or `1` per sample (a flat array works) |
| Multi-label | `dense(k, "sigmoid")` | `"bce"` | `k` columns of 0/1 |
| Multi-class, integer labels | `dense(k, "softmax")` | `"scce"` | One class index per sample: `[2, 0, 1, …]` |
| Multi-class, one-hot | `dense(k, "softmax")` | `"cce"` | `k` columns, e.g. from `oneHot(labels, k)` |

`sigmoid` + `bce` and `softmax` + `cce`/`scce` are trained with a fused, numerically stable gradient, so prefer these
pairs over, say, `softmax` + `mse`.

Metrics are reported under their **canonical names**: `"mae"` becomes `meanAbsoluteError`, `"rmse"`
`rootMeanSquaredError`. `"accuracy"` picks the right variant:

- with a `"bce"` loss it is always **binary** accuracy (each output thresholded at 0.5), so it also works for
  multi-label models with several sigmoid units;
- otherwise it follows the shapes: binary for one output column, **sparse categorical** for several outputs and one
  target column (class indices), **categorical** for one-hot targets.

Compiling again replaces the loss, optimizer and metrics. An optimizer name or config creates a fresh optimizer; an
`Optimizer` instance is used as-is and keeps its state. The compiled optimizer is available as `model.optimizer`; its
`learningRate` can be changed at any time. `compile` and `add` cannot be called while the model is training (from a
callback, say): stop training first.

### Training with fit

`fit(x, y, options)` trains for a number of **epochs** (passes over the data). Each epoch shuffles the samples (unless
`shuffle: false`) and splits them into **batches** of `batchSize` (default 32; the last one may be smaller). For every
batch the model:

1. runs a forward pass in **training mode** (dropout active, batch normalization using the batch's statistics);
2. computes the loss (the batch mean) plus any weight regularization;
3. backpropagates to get every parameter's gradient;
4. lets the optimizer update the weights.

```ts
const history = model.fit(x, y, {
    epochs: 100, // default 1
    batchSize: 4, // default 32
    shuffle: true, // default true: reshuffle every epoch
    verbose: 20, // log every 20th epoch (true: every epoch; false or 0: silent, the default)
});
```

Validation data is evaluated in inference mode at the end of every epoch:

- `validationSplit: 0.2` holds out the **last** 20% of the samples, taken before any shuffling (as in Keras). If your
  data is ordered (all of class 0, then all of class 1, …), shuffle it first with `shuffleTogether` or split it with
  `trainTestSplit`.
- `validationData: [xVal, yVal]` uses an explicit set and takes precedence over `validationSplit`.

**Resuming training.** Calling `fit` again continues from the current weights, but epoch numbering starts at 0 again,
which restarts any learning-rate schedule. Pass `initialEpoch` to continue the numbering instead: callbacks, schedules,
the progress log and the returned `History` then count from there.

```ts
import { cosineDecay, learningRateScheduler } from "@zzza38/orion-engine";

const schedule = learningRateScheduler(cosineDecay({ initial: 0.05, epochs: 200 }));
model.fit(x, y, { epochs: 100, batchSize: 4, callbacks: [schedule] }); // epochs 0-99
// … inspect, save a checkpoint, change the data …
const rest = model.fit(x, y, { epochs: 100, batchSize: 4, initialEpoch: 100, callbacks: [schedule] }); // epochs 100-199
console.log(rest.epochs[0], rest.last("learningRate")); // 100, close to 0
```

For full control you can run single steps with `trainOnBatch(xBatch, yBatch)`, which returns the batch's loss and
metrics.

### Logs and History

Every epoch produces **logs**: a flat `Record<string, number>` with

- `loss`, plus one entry per metric (e.g. `accuracy`): means over the epoch's batches, weighted by batch size;
- `valLoss`, `valAccuracy`, … when validating;
- `learningRate` (the rate used for the epoch) and `durationMs`.

`fit` returns a `History` of those logs:

```ts
const history = model.fit(x, y, { epochs: 50, batchSize: 4 });

history.length; // 50 epochs recorded
history.last("loss"); // the final training loss
history.best("accuracy"); // { epoch, value }: max for accuracy-like keys, min for everything else
history.history.loss; // number[]: the loss of every epoch
history.toJSON(); // { epochs, history }, a plain snapshot
```

Training metrics are measured during training (in training mode, before each batch's update), so they lag behind the
model and are noisy with dropout or batch normalization. Validation metrics are computed afterwards in inference mode
and are the better guide.

### Callbacks

A callback is any object with some of these hooks, called in the order the callbacks are listed:

| Hook | Called | Arguments |
|---|---|---|
| `onTrainBegin` | Once, before the first epoch | `ctx` |
| `onEpochBegin` | Before each epoch | `epoch`, `ctx` |
| `onBatchEnd` | After each batch's update | `batch`, `logs` (the batch's `loss`, metrics and `size`), `ctx` |
| `onEpochEnd` | After each epoch (and validation) | `epoch`, `logs`, `ctx` |
| `onTrainEnd` | Once, at the end | `logs` (of the last epoch), `ctx` |

Epoch and batch numbers are 0-based (epochs start at `initialEpoch` when you set it). The context `ctx` gives access
to the `model`, the `optimizer` (whose `learningRate` you may change), `initialEpoch` and `epochs` (one past the last
epoch number), `batchSize`, `samples`, `stepsPerEpoch`, `hasValidation`, the `history` so far, and `stopTraining()`,
which ends training after the current epoch.

```ts
import type { Callback } from "@zzza38/orion-engine";

// Stop as soon as the training loss drops below a threshold.
const stopAtLoss = (threshold: number): Callback => ({
    onEpochEnd(epoch, logs, ctx) {
        if (logs.loss < threshold) {
            console.log(`Reached loss ${logs.loss.toFixed(4)} after ${epoch + 1} epochs`);
            ctx.stopTraining();
        }
    },
});

model.fit(x, y, { epochs: 1000, batchSize: 4, callbacks: [stopAtLoss(0.01)] });
```

For the common case there is a shorthand: `fit(x, y, { onEpochEnd: (epoch, logs) => … })`.

The built-in callbacks:

- `earlyStopping({ monitor, patience, minDelta, mode, restoreBestWeights })` stops training once `monitor` (default
  `"valLoss"`) has not improved by more than `minDelta` for `patience` epochs. The default `patience` is 0, which stops
  at the first epoch without improvement, so you will usually want 5 to 20. With `restoreBestWeights: true` the model
  ends with the weights of its best epoch (restored when training ends normally, not when it ends with an error such
  as an aborted `fitAsync`). After `fit`, the callback's `stoppedEpoch`, `bestEpoch` and `bestValue` describe the run.
- `learningRateScheduler(schedule)` sets the learning rate at the start of each epoch from `schedule(epoch, currentRate)`.
  Pass one of the schedule factories (`cosineDecay`, `stepDecay`, `exponentialDecay`, `piecewiseConstant`,
  `linearWarmup`, `constantSchedule`) or your own function.
- `reduceLROnPlateau({ monitor, factor, patience, minDelta, cooldown, minLearningRate })` multiplies the learning rate by
  `factor` (default 0.1) after `patience` (default 10) epochs without improvement.
- `progressLogger({ every, log })` prints a line every `every` epochs. `verbose` installs one for you.

`earlyStopping` and `reduceLROnPlateau` watch `"valLoss"` by default; without validation data they fall back to
`"loss"` and print a one-time warning. `mode: "auto"` (the default) maximizes keys that look like accuracy and minimizes
everything else.

```ts
import { cosineDecay, earlyStopping, learningRateScheduler, linearWarmup } from "@zzza38/orion-engine";

model.fit(x, y, {
    epochs: 200,
    batchSize: 4,
    validationData: [x, y],
    callbacks: [
        earlyStopping({ monitor: "valLoss", patience: 15, restoreBestWeights: true }),
        // 5 epochs of warm-up from 0 to 0.01, then cosine decay over 195 epochs.
        learningRateScheduler(linearWarmup(cosineDecay({ initial: 0.01, epochs: 195 }), { epochs: 5 })),
    ],
});
```

Hooks may return a Promise. `fitAsync` awaits it; the synchronous `fit` cannot, and throws a `ValidationError` that
points you to `fitAsync`.

### Evaluating and predicting

`evaluate(x, y)` returns the loss (including regularization) and every metric over a dataset, in inference mode:

```ts
const { loss, accuracy } = model.evaluate(xTest, yTest);
```

`predict` runs the model in inference mode (dropout off, batch normalization using its moving averages) and mirrors
its input:

```ts
import { Matrix } from "@zzza38/orion-engine";

model.predict([1, 0]); // one sample in: number[] out, e.g. [0.98]
model.predict([[1, 0], [0, 0]]); // rows in: number[][] out, e.g. [[0.98], [0.01]]
model.predict(Matrix.from([[1, 0]])); // Matrix in: a new Matrix out
```

Large inputs are processed in chunks of `batchSize` (default 256, set with `predict(x, { batchSize })`). For
classification, `argmax(model.predict(rows))` turns probabilities into class indices.

## Preparing data

**Scale your inputs.** Networks train best when every feature is roughly centred with unit spread. Fit a scaler on the
training data only, then apply the same transform everywhere else, including at prediction time:

```ts
import { MinMaxScaler, StandardScaler } from "@zzza38/orion-engine";

const xTrain = [[150, 0.2], [180, 0.5], [165, 0.9]];
const xNew = [[170, 0.4]];

const scaler = new StandardScaler(); // (x − mean) / std, per feature
const xTrainScaled = scaler.fitTransform(xTrain);
const xNewScaled = scaler.transform(xNew);

const minMax = new MinMaxScaler({ featureRange: [-1, 1] }); // linear map of [min, max] to [-1, 1]
minMax.fitTransform(xTrain);

// Scalers serialize to JSON, so you can ship them next to the model.
const restored = StandardScaler.fromJSON(JSON.parse(JSON.stringify(scaler)));
console.log(restored.transform(xNew), xNewScaled); // the same
console.log(scaler.inverseTransform(xTrainScaled)); // back to the original units
```

Scale regression **targets** too when they are large (house prices in dollars, say): fit a second scaler on `y` (as
one-column rows) and `inverseTransform` the predictions.

**Encode labels.** Class labels must be integers `0 … k-1`. Map strings to indices yourself, then either pass the
indices with `"scce"` or one-hot encode them for `"cce"`:

```ts
import { argmax, oneHot } from "@zzza38/orion-engine";

const species = ["setosa", "virginica", "versicolor", "setosa"];
const names = [...new Set(species)].sort(); // ["setosa", "versicolor", "virginica"]
const labels = species.map((s) => names.indexOf(s)); // [0, 2, 1, 0]

oneHot(labels); // [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
oneHot(labels, 4); // with an explicit number of classes
argmax([0.1, 0.7, 0.2]); // 1: back from probabilities to a class index
```

**Split your data.** Keep a test set that training never sees. `trainTestSplit` shuffles (with its own `seed`) and
keeps inputs and targets paired; it returns the same kinds it was given (arrays or matrices):

```ts
import { shuffleTogether, trainTestSplit } from "@zzza38/orion-engine";

const x = Array.from({ length: 10 }, (_, i) => [i, i * i]);
const y = x.map(([a]) => a % 2);

const { xTrain, xTest, yTrain, yTest } = trainTestSplit(x, y, { testSize: 0.2, seed: 42 }); // 8 train, 2 test
const [xShuffled, yShuffled] = shuffleTogether(x, y, 42); // same permutation for both
```

`testSize` is a fraction in (0, 1) or an absolute count. There is no stratification: with small, imbalanced datasets,
check that every class made it into both sets.

**Clean missing values.** Inputs and targets must be finite numbers. `NaN` or `Infinity` anywhere raises a
`ValidationError` naming the row and column; impute or drop those samples first.

## Training tips

**Learning rate.** The single most important setting. Adam's default of 0.001 is safe; 0.01 is often faster for small
networks. If the loss explodes or turns `NaN`, lower it by 10×. If it creeps down very slowly, raise it. Schedules
(`cosineDecay`, `stepDecay`, `reduceLROnPlateau`) let you start fast and finish precisely.

**Batch size.** Smaller batches (8 to 32) mean more updates per epoch and some helpful noise; larger batches (64 to 256)
process more samples per second and give smoother gradients but may need a higher learning rate or more epochs. The
default, 32, is a good start.

**Architecture.** Start small (one or two hidden layers of 16 to 64 units) and grow only if the model underfits (both
training and validation loss stay high). `relu` is the default choice for hidden layers; `tanh` works well in small
networks; `gelu`, `swish` and `mish` are smooth alternatives. The output layer's activation follows the task (see the
[table above](#compiling-loss-optimizer-metrics)).

**Initializers.** `Dense` uses `glorotUniform` for kernels and `zeros` for biases, which suits `tanh` and `sigmoid`.
For deep ReLU networks, `heNormal` or `heUniform` usually trains faster:

```ts
import { dense } from "@zzza38/orion-engine";

dense(128, { activation: "relu", kernelInitializer: "heNormal" });
dense(8, { activation: "tanh", biasInitializer: { name: "constant", value: 0.1 } });
```

**Batch normalization.** Put `batchNormalization()` between a linear `Dense` layer and its activation. The Dense bias
is then redundant (batch normalization re-centres every feature), so drop it:

```ts
import { activation, batchNormalization, dense } from "@zzza38/orion-engine";

const block = [dense(64, { useBias: false, kernelInitializer: "heNormal" }), batchNormalization(), activation("relu")];
```

During training it normalizes with the current batch's mean and variance and updates moving averages (momentum 0.99);
at inference it uses the moving averages, which are saved with the model. Very small batches (under about 8) make the
batch statistics noisy; a batch of a single sample (a trailing partial batch, say) is normalized but kept out of the
moving averages. If validation results lag behind training for many epochs, try `batchNormalization({ momentum: 0.9 })`
so the moving averages catch up faster.

**Dropout.** `dropout(rate)` zeroes a random fraction of its inputs during training and scales the rest by
`1 / (1 − rate)`, so nothing changes at inference. Rates of 0.1 to 0.5 after wide hidden layers fight overfitting.
Expect a noisier, higher training loss: compare validation numbers instead.

**Regularization.** When the training loss keeps falling but the validation loss rises, the model is overfitting.
Besides dropout and early stopping you can penalize large weights:

```ts
import { dense } from "@zzza38/orion-engine";

dense(64, { activation: "relu", kernelRegularizer: { l2: 1e-4 } }); // adds l2·Σw² to the loss (l1·Σ|w| with l1)
model.compile({ loss: "bce", optimizer: { name: "adamw", learningRate: 0.001, weightDecay: 0.01 } });
```

`kernelRegularizer` is part of the loss (it appears in `loss`, `valLoss` and `evaluate`). AdamW's `weightDecay` shrinks
the weights directly at every step; SGD's `weightDecay` adds `λ·w` to the gradient. Neither touches biases or batch
normalization parameters.

**Gradient clipping.** If training occasionally blows up (large inputs, recurrent spikes), clip the gradients:

```ts
model.compile({ loss: "mse", optimizer: { name: "sgd", learningRate: 0.01, momentum: 0.9, clipNorm: 1 } });
```

`clipNorm` rescales each parameter's gradient to an L2 norm of at most the given value; `clipValue` clamps each
element to `[-clipValue, clipValue]`. Both can be combined (norm first).

**Reproducibility.** Pass a `seed` to the model (and to `trainTestSplit`) and the same code produces bit-identical
results on the same JavaScript engine. Without one, a random seed is used; `model.seed` tells you which.

## Training in the browser: fitAsync

`fit` is synchronous: while it runs, nothing else on the thread does. That is fine in scripts, but in a page it freezes
rendering and input. `fitAsync` runs the same training loop and returns a Promise of the same `History`, with three
differences:

- It **yields to the event loop** whenever `yieldEvery` milliseconds (default 16, about one animation frame) of work
  have passed, using `scheduler.yield()` where available and a zero-delay timeout otherwise.
- It **awaits callbacks that return Promises**, so a callback can draw a chart, save a checkpoint or wait for the next
  animation frame before training continues.
- It can be **cancelled** with an `AbortSignal`. The Promise then rejects with the signal's reason at the next batch;
  the model keeps the weights it had reached.

```ts
const controller = new AbortController();
// stopButton.onclick = () => controller.abort();

const done = model.fitAsync(x, y, {
    epochs: 100_000,
    batchSize: 4,
    signal: AbortSignal.any([controller.signal, AbortSignal.timeout(500)]), // Stop button or 0.5 s
    onEpochEnd: async (epoch, logs) => {
        if (epoch % 1000 === 0) console.log(`epoch ${epoch}: loss ${logs.loss.toFixed(4)}`); // e.g. update a chart
    },
});

try {
    await done;
} catch (error) {
    if (!(error instanceof Error && (error.name === "AbortError" || error.name === "TimeoutError"))) throw error;
    console.log("Stopped early; the model keeps what it learned.");
}
```

A model can run only one `fit`/`fitAsync` at a time; starting another throws a `ValidationError` ("already training").
The library has no DOM dependencies, so you can also move training into a Web Worker and post the logs back to the
page. [`examples/async-training.ts`](../examples/async-training.ts) and
[`examples/browser/index.html`](../examples/browser/index.html) show complete programs.

## Saving models

A saved model contains the architecture, the weights (including batch normalization's moving averages) and, when the
model was compiled, the loss, the optimizer's settings and the metrics. It does not contain the optimizer's internal
state (Adam's moment estimates), so continued training starts the optimizer afresh. The saved settings include the
**current** learning rate: a model saved at the end of a decaying schedule comes back with a tiny rate, so recompile it
with the rate you want (or train it with a schedule and `initialEpoch`) before training further.

| Where | Save | Load |
|---|---|---|
| Node.js files | `saveModel(model, path, options?)`, `saveModelSync` | `loadModel(path, options?)`, `loadModelSync` |
| Anywhere, in memory | `serializeModel(model, options?)` | `deserializeModel(data, options?)` |

The file helpers come from `@zzza38/orion-engine/node`. Choose the format with `format: "binary"` (the default, a
compact checksummed container) or `"json"`; `saveModel` picks JSON for paths ending in `.json`. Binary weights are
stored as float32 by default (`precision: "float64"` keeps them bit-exact); JSON is always exact. Loading detects the
format, including legacy 0.0.x `.onn` files. [format.md](format.md) specifies the formats, and the README has
[examples](../README.md#saving-and-loading). Save scalers with `JSON.stringify(scaler)` next to the model.

## Troubleshooting

Orion throws typed errors, all subclasses of `OrionError`:

| Error | Thrown when |
|---|---|
| `ValidationError` | An argument or option is invalid, or the model is in the wrong state (not compiled, no layers, already training) |
| `ShapeError` | Shapes do not line up: wrong number of features, targets, or weights |
| `TrainingError` | The loss became `NaN` or infinite during training |
| `SerializationError` | Saved model data is corrupt, truncated or not a model |

Messages name the problem and, where there is one, the fix. The most common ones:

**`TrainingError: Training diverged: the loss became NaN at epoch 3/50, batch 2/4`.** The weights blew up. In order of
likelihood: the learning rate is too high (the message suggests a value 10× smaller), the inputs are not scaled
(features in the thousands), or the gradients occasionally spike (set `clipNorm`). The model's weights are left as
they were when the error was thrown, so rebuild it before retrying.

**`ShapeError: Expected input with 2 features (inputSize), got 3`.** The rows of `x` have a different length from the
model's `inputSize`. If the model has one input and you passed a flat array, the message adds a hint: a flat array is
one sample, so write `[[0.1], [0.2]]`.

**`ShapeError: fit: expected y with 3 columns (outputSize of the last layer), got 1`.** The targets do not match the
last layer. For integer class labels with a softmax output, use `loss: "scce"` or one-hot encode them with
`oneHot(labels, 3)`.

**`ValidationError: fit: y[1] = 3 is not a class index in [0, 3)`.** With `"scce"`, labels must be integers from 0 to
the number of output units minus 1.

**`ValidationError: Non-numeric value at [1, 1]: NaN`** or **`… contains Infinity at row 4, column 0`.** Clean or impute
the data first.

**`ValidationError: fit: the model is not compiled`.** Call `compile` before `fit`, `evaluate` or `trainOnBatch`
(`predict` works without it).

**`ValidationError: A callback's onEpochEnd() returned a Promise, but fit() is synchronous`.** Use `await
model.fitAsync(…)` for async callbacks.

**`earlyStopping: "valLoss" is not available because fit() has no validation data`** (a warning). Pass
`validationSplit` or `validationData`, or monitor `"loss"` explicitly.

**`SerializationError: Unrecognized model format: this looks like a binary model that was converted to a string`.**
Binary models must stay bytes: read files without an encoding, and `fetch(…).arrayBuffer()` rather than `.text()`.

**`SerializationError: deserializeModel: Unknown layer type "…"`.** The file is intact but uses a custom layer: call
`registerLayer` before loading it. For errors like this one, the original error is available as `error.cause`.

**`ValidationError: compile() cannot be called while the model is training`.** A callback tried to recompile (or `add`
a layer) during `fit`. Call `ctx.stopTraining()`, then change the model and call `fit` again.

**The loss does not go down, or accuracy is stuck at chance level.** Work through this list:

1. Check the data: are `x` and `y` paired correctly, and are the labels what you think? Try overfitting 10 samples; a
   healthy model reaches near-zero training loss on them.
2. Scale the inputs with `StandardScaler`.
3. Check that the output activation and the loss match the task ([table](#compiling-loss-optimizer-metrics)); a
   `relu` or linear output with `"bce"`, for example, will not train properly.
4. Try another learning rate: 10× higher and 10× lower.
5. Train longer, or make the network wider or deeper.
6. With deep ReLU networks, use `kernelInitializer: "heNormal"` or add batch normalization.

**Validation accuracy is higher than training accuracy.** Normal with dropout or batch normalization: training metrics
are measured in training mode, validation metrics in inference mode.
