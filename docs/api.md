# API reference

Everything exported by `@zzza38/orion-engine` (browser-safe) and `@zzza38/orion-engine/node` (the same exports plus
file helpers), grouped by topic. For explanations and advice, read the [guide](guide.md).

```ts
import { dense, Sequential } from "@zzza38/orion-engine"; // Node.js and browsers
import { loadModel, saveModel } from "@zzza38/orion-engine/node"; // Node.js: everything above, plus files
```

**Conventions.** Options are plain objects; unknown keys and out-of-range values throw a `ValidationError` that lists
the valid choices. "Default" columns give the value used when an option is omitted. Examples on this page that do not
build their own model use a compiled 2-input model called `model` and XOR data `x` (`number[][]`) and `y`
(`number[]`), as in the [quick start](../README.md#quick-start).

**Contents**

- [Model](#model): [`Sequential`](#sequential)
- [Layers](#layers): [`dense`](#dense--dense), [`dropout`](#dropout--dropout),
  [`batchNormalization`](#batchnormalization--batchnormalization), [`activation`](#activation--activationlayer),
  [custom layers](#custom-layers)
- [Activations](#activations), [losses](#losses), [metrics](#metrics), [initializers](#initializers)
- [Optimizers](#optimizers), [learning-rate schedules](#learning-rate-schedules)
- [Callbacks and History](#callbacks-and-history)
- [Data utilities](#data-utilities)
- [Saving and loading](#saving-and-loading), [model formats](#model-formats-low-level)
- [Matrix and math](#matrix-and-math), [`Random`](#random), [errors](#errors), [`VERSION`](#version)

---

## Model

### `Sequential`

A linear stack of layers with `compile` / `fit` / `evaluate` / `predict`.

```ts
import { dense, Sequential } from "@zzza38/orion-engine";

const model = new Sequential({ inputSize: 4, seed: 1, layers: [dense(16, "relu"), dense(3, "softmax")] });
```

#### `new Sequential(options?: SequentialOptions)`

| Option | Type | Default | Description |
|---|---|---|---|
| `inputSize` | `number` | inferred | Number of input features. When omitted, it is set by the first `fit`, `predict`, `evaluate` or `trainOnBatch` call, or by `build(inputSize)` |
| `layers` | `Layer[]` | `[]` | Initial layers; more can be added with `add` |
| `seed` | integer | random | Seeds weight initialization, shuffling and dropout |
| `name` | `string` | `"sequential"` | Shown by `summary()`, stored in saved models |

#### Properties

| Property | Type | Description |
|---|---|---|
| `name` | `string` | Model name (read-only) |
| `seed` | `number` | Seed of the model's generator (the random one when none was given) |
| `layers` | `readonly Layer[]` | The layers, in order |
| `inputSize` | `number \| undefined` | Number of input features, once known |
| `outputSize` | `number \| undefined` | Output size of the last layer, once built |
| `built` | `boolean` | True once the input size is known and every layer has weights |
| `compiled` | `boolean` | True after `compile` |
| `loss` | `Loss \| undefined` | The compiled loss |
| `optimizer` | `Optimizer \| undefined` | The compiled optimizer; its `learningRate` may be changed at any time |
| `metrics` | `readonly Metric[]` | The compiled metrics |
| `parameterCount` | `number` | Total number of scalar weights, trainable and not |

#### Structure

- **`add(layer: Layer): this`** appends a layer (chainable). Unnamed layers are named `<type>_<n>`; names must be unique
  within the model. If the input size is known the layer is built immediately. Throws while the model is training.
- **`build(inputSize: number): this`** fixes the input size and creates every layer's weights. Building again with the
  same size is a no-op; a different size throws a `ShapeError`.
- **`parameters(): Parameter[]`** returns every parameter (trainable and not) in layer order.
- **`summary(): string`** returns a table of layers, output sizes, activations and parameter counts.
- **`clone(): Sequential`** returns a deep copy with the same architecture, name, seed and weights, compiled the same
  way (with a fresh optimizer).

```ts
import { dense, dropout, Sequential } from "@zzza38/orion-engine";

const net = new Sequential({ inputSize: 3, seed: 1 });
net.add(dense(4, "relu")).add(dropout(0.1)).add(dense(1));
console.log(net.layers.map((layer) => layer.name)); // ["dense_1", "dropout_1", "dense_2"]
console.log(net.parameterCount); // 21
```

#### `compile(options: CompileOptions): this`

| Option | Type | Default | Description |
|---|---|---|---|
| `loss` | `LossIdentifier` | required | A name or alias (`"mse"`, `"bce"`, `"scce"`, …), a config (`{ name: "huber", delta: 2 }`) or a `Loss` |
| `optimizer` | `OptimizerIdentifier` | `"adam"` | A name, a config (`{ name: "sgd", learningRate: 0.1, momentum: 0.9 }`) or an `Optimizer` |
| `metrics` | `MetricIdentifier[]` | `[]` | Names, aliases or `Metric` objects; logged under their canonical names |

Compiling again replaces the loss, optimizer and metrics. An optimizer name or config creates a fresh optimizer; an
`Optimizer` instance is used as-is, keeping its state. `compile` throws while the model is training (for example when
called from a callback).

#### `fit(x: MatrixLike, y: MatrixLike, options?: FitOptions): History`

Trains for a fixed number of epochs and returns the per-epoch [`History`](#history). `x` is `number[][]` (one row per
sample), a `Matrix`, or one sample as `number[]`. `y` has one row per sample; a flat `number[]` is one value per sample
(binary targets, or class indices for `"scce"`).

| Option | Type | Default | Description |
|---|---|---|---|
| `epochs` | positive integer | `1` | Passes over the training data in this call |
| `initialEpoch` | integer ≥ 0 | `0` | Number of the first epoch, to resume training across `fit` calls: epoch numbers seen by callbacks, schedules, the progress log and the `History` start here |
| `batchSize` | positive integer | `32` | Samples per gradient step; the last batch of an epoch may be smaller |
| `shuffle` | `boolean` | `true` | Reshuffle the training samples every epoch (with the model's seeded generator) |
| `validationSplit` | number in (0, 1) | none | Fraction of samples to hold out for validation: the **last** ones, before shuffling |
| `validationData` | `[x, y]` | none | Explicit validation set; takes precedence over `validationSplit` |
| `callbacks` | `Callback[]` | `[]` | Called in order; see [callbacks](#callbacks-and-history) |
| `onEpochEnd` | `(epoch, logs) => void` | none | Shorthand for a callback with only `onEpochEnd` |
| `verbose` | `boolean \| number` | `false` | Log progress: `true` every epoch, `N` every N epochs; `0` or `false` is silent |

Epoch logs contain `loss`, each metric, `valLoss` / `val<Metric>` when validating, `learningRate` and `durationMs`.
Throws a `TrainingError` if the loss becomes `NaN` or infinite, and a `ValidationError` if a callback returns a Promise
(use `fitAsync`), if `x` has no samples, or if the model is already training.

```ts
const history = model.fit(x, y, { epochs: 100, batchSize: 4, validationData: [x, y], verbose: 25 });
console.log(history.last("valAccuracy"));

// Continue for another 100 epochs: numbering (and any schedule) picks up at epoch 100.
const more = model.fit(x, y, { epochs: 100, batchSize: 4, initialEpoch: 100 });
console.log(more.epochs[0]); // 100
```

#### `fitAsync(x: MatrixLike, y: MatrixLike, options?: FitAsyncOptions): Promise<History>`

The same training loop as `fit`, but it periodically yields to the event loop, awaits callbacks that return Promises,
and can be cancelled. `FitAsyncOptions` has every `FitOptions` option (its `onEpochEnd` may return a Promise) plus:

| Option | Type | Default | Description |
|---|---|---|---|
| `signal` | `AbortSignal` | none | Aborts training: the Promise rejects with `signal.reason` at the next batch |
| `yieldEvery` | number ≥ 0 | `16` | Yield to the event loop whenever this many milliseconds of work have passed |

```ts
const history = await model.fitAsync(x, y, {
    epochs: 200,
    batchSize: 4,
    signal: AbortSignal.timeout(5000),
    onEpochEnd: async (epoch, logs) => {
        if (epoch % 50 === 0) console.log(epoch, logs.loss);
    },
});
```

#### `trainOnBatch(x: MatrixLike, y: MatrixLike): Logs`

Runs one gradient step on one (non-empty) batch and returns its `loss` and metrics (computed before the update).

#### `evaluate(x: MatrixLike, y: MatrixLike, options?: BatchOptions): Logs`

Returns the loss (including weight regularization) and every metric over a dataset, in inference mode. `batchSize`
(default `32`) only affects memory use.

```ts
const { loss, accuracy } = model.evaluate(x, y);
```

#### `predict(input, options?: BatchOptions)`

Runs the model in inference mode (dropout off, batch normalization using its moving averages). The output mirrors the
input:

| Input | Output |
|---|---|
| `number[]` (one sample) | `number[]` |
| `number[][]` (rows) | `number[][]` |
| `Matrix` | a new `Matrix` |

`batchSize` (default `256`) sets how many rows go through the network at once. `predict` does not need a compiled
model.

```ts
import { Matrix } from "@zzza38/orion-engine";

model.predict([1, 0]); // [0.99…]
model.predict([[1, 0], [1, 1]]); // [[0.99…], [0.00…]]
model.predict(Matrix.from(x), { batchSize: 2 }); // Matrix [4, 1]
```

#### Weights and artifacts

- **`getWeights(): WeightEntry[]`** returns copies of every parameter as `{ name, shape: [rows, cols], data }`, in
  `parameters()` order (including batch normalization's moving statistics).
- **`setWeights(entries: readonly WeightEntry[]): void`** replaces every parameter. Each of the model's weights must
  appear exactly once (matched by name, in any order) with the right shape and finite values; nothing changes if any
  entry is invalid. The model must be built.
- **`toArtifact(metadata?): ModelArtifact`** describes the model (architecture, weights, and the training config when
  compiled) as a [`ModelArtifact`](#modelartifact), the in-memory form of a saved model. The name is stored as
  `metadata.name`.
- **`Sequential.fromArtifact(artifact: ModelArtifact, options?: { seed?: number }): Sequential`** rebuilds a model from an
  artifact, compiled (with a fresh optimizer) if the artifact has a training config.

```ts
import { Sequential } from "@zzza38/orion-engine";

const copy = Sequential.fromArtifact(model.toArtifact({ trainedOn: "xor" }), { seed: 1 });
copy.setWeights(model.getWeights());
```

---

## Layers

Every layer accepts a `name` (see `LayerOptions`); leave it out to get `<type>_<n>`. Names must not contain `/`,
because weights are named `"<layer name>/<parameter>"`. Layers are stateful objects: create a new one for each model.

| Type | Factory | Class | Parameters |
|---|---|---|---|
| `"dense"` | `dense(units, activationOrOptions?)` | `Dense` | `kernel` [inputs, units], `bias` [1, units] |
| `"dropout"` | `dropout(rate, options?)` | `Dropout` | none |
| `"batchNormalization"` | `batchNormalization(options?)` | `BatchNormalization` | `gamma`, `beta`, `movingMean`, `movingVariance` (all [1, features]) |
| `"activation"` | `activation(id, options?)` | `ActivationLayer` | none |

`LAYER_TYPES` lists the built-in type names.

### `dense()` / `Dense`

A fully connected layer computing `activation(x · kernel + bias)`.

`dense(units: number, activationOrOptions?: ActivationIdentifier | Omit<DenseOptions, "units">): Dense`. The second
argument is either an activation (a name, a config such as `{ name: "leakyRelu", alpha: 0.2 }`, or an `Activation`)
or the remaining options. `new Dense(options: DenseOptions)` takes all options, including `units`.

| Option | Type | Default | Description |
|---|---|---|---|
| `units` | positive integer | required | Number of outputs |
| `activation` | `ActivationIdentifier` | `"linear"` | Applied to `x · kernel + bias` |
| `useBias` | `boolean` | `true` | Add a bias vector |
| `kernelInitializer` | `InitializerIdentifier` | `"glorotUniform"` | How to initialize the kernel |
| `biasInitializer` | `InitializerIdentifier` | `"zeros"` | How to initialize the bias |
| `kernelRegularizer` | `{ l1?, l2? } \| null` | none | Adds `l1·Σ\|w\| + l2·Σw²` to the loss (both default 0) |
| `name` | `string` | auto | Layer name |

A `Dense` layer exposes `units`, `activation`, `useBias`, `kernelInitializer`, `biasInitializer`, `kernelRegularizer`,
`kernel` and `bias` (the `Parameter`s, after build) and `regularizationLoss()`.

```ts
import { dense } from "@zzza38/orion-engine";

dense(8, "relu");
dense(1, { activation: "sigmoid", name: "output" });
dense(64, { activation: "relu", kernelInitializer: "heNormal", kernelRegularizer: { l2: 1e-4 } });
```

An object whose only key is `name` is read as an activation config when the name is an activation (`{ name: "relu" }`)
and as the layer name otherwise (`{ name: "output" }`).

### `dropout()` / `Dropout`

`dropout(rate: number, options?: LayerOptions): Dropout`; `new Dropout({ rate, name? })`.

Inverted dropout: during training, each input is zeroed with probability `rate` (in [0, 1)) and the others are scaled
by `1 / (1 − rate)`; at inference it passes inputs through unchanged. Masks come from the model's seeded generator.

### `batchNormalization()` / `BatchNormalization`

`batchNormalization(options?: BatchNormalizationOptions): BatchNormalization`; `new BatchNormalization(options?)`.

Normalizes each feature, then applies a learned scale and offset: `y = gamma · (x − mean) / sqrt(variance + epsilon) +
beta`. Training uses the batch's mean and (biased) variance and updates the moving averages; inference uses the
moving averages. A training batch of a single sample (such as a trailing partial batch) is normalized but left out of
the moving averages, since it has no variance. The moving averages are non-trainable parameters and are saved with the
model.

| Option | Type | Default | Description |
|---|---|---|---|
| `momentum` | number in [0, 1) | `0.99` | `moving = momentum · moving + (1 − momentum) · batch` |
| `epsilon` | number > 0 | `1e-3` | Added to the variance |
| `center` | `boolean` | `true` | Learn the offset `beta` |
| `scale` | `boolean` | `true` | Learn the scale `gamma` |
| `name` | `string` | auto | Layer name |

### `activation()` / `ActivationLayer`

`activation(id: ActivationIdentifier, options?: { name?: string }): ActivationLayer`;
`new ActivationLayer({ activation, name? })`.

Applies an activation with no parameters, typically after batch normalization:

```ts
import { activation, batchNormalization, dense, Sequential } from "@zzza38/orion-engine";

const net = new Sequential({ inputSize: 10, layers: [dense(32), batchNormalization(), activation("relu"), dense(1)] });
```

### Custom layers

#### `BaseLayer`

The abstract base class of the built-in layers. Extend it to write your own; the model then skips input gradients for
the first layer and handles naming and building for you. A subclass implements:

| Member | Description |
|---|---|
| `readonly type: string` | The type name, matching `getConfig().type` and the name passed to `registerLayer` |
| `constructor(...)` | Calls `super(name)`; pass `""` to let the model assign a name |
| `protected onBuild(inputSize: number, rng: Random): void` | Creates parameters and buffers, once the input size is known |
| `forward(input: Matrix, training: boolean): Matrix` | The forward pass over a batch `[batch, inputSize]`; cache what `propagate` needs |
| `propagate(gradOutput: Matrix, inputGradient: boolean): Matrix \| null` | Writes every parameter's `grad` for the last forward batch and returns dL/dInput, or `null` when `inputGradient` is false |
| `getConfig(): LayerConfig` | A JSON-safe `{ type, name, … }` from which the layer can be rebuilt |
| `parameters(): Parameter[]` | Optional; default `[]` |
| `get outputSize(): number` | Optional; defaults to the input size |

`BaseLayer` provides `name`, `built`, `inputSize`, `build(inputSize, rng)`, `backward(gradOutput)` (which calls
`propagate(gradOutput, true)`), and the protected helpers `checkInput(input)` (throws unless built and the feature
count matches) and `label` (a `Dense layer "dense_1"`-style name for error messages).

Parameters are usually `LayerParameter`s: `new LayerParameter(owner, key, rows, cols, { trainable?, regularize? })`
allocates a `value` and a `grad` matrix and is named `"<owner name>/<key>"` (it follows renames of the owner). Set
`trainable: false` for statistics that the optimizer must not touch and `regularize: false` to exempt a parameter from
weight decay.

```ts
import type { LayerConfig, Random } from "@zzza38/orion-engine";
import { BaseLayer, dense, LayerParameter, Matrix, registerLayer, Sequential } from "@zzza38/orion-engine";

/** Learns one multiplier per feature: y = x · w. */
class FeatureScale extends BaseLayer {
    readonly type = "featureScale";
    private weight: LayerParameter | null = null;
    private lastInput: Matrix | null = null;

    constructor(name = "") {
        super(name);
    }

    protected onBuild(inputSize: number, _rng: Random): void {
        this.weight = new LayerParameter(this, "weight", 1, inputSize, { regularize: false });
        this.weight.value.fill(1);
    }

    forward(input: Matrix, _training: boolean): Matrix {
        this.checkInput(input);
        this.lastInput = input;
        const w = (this.weight as LayerParameter).value.data;
        return input.map((v, _row, col) => v * w[col]);
    }

    propagate(gradOutput: Matrix, inputGradient: boolean): Matrix | null {
        const param = this.weight as LayerParameter;
        const input = this.lastInput as Matrix;
        param.grad.fill(0);
        for (let r = 0; r < input.rows; r++) {
            for (let c = 0; c < input.cols; c++) param.grad.data[c] += gradOutput.get(r, c) * input.get(r, c);
        }
        return inputGradient ? gradOutput.map((g, _row, col) => g * param.value.data[col]) : null;
    }

    override parameters(): LayerParameter[] {
        return this.weight === null ? [] : [this.weight];
    }

    getConfig(): LayerConfig {
        return { type: this.type, name: this.name };
    }
}
registerLayer("featureScale", (config) => new FeatureScale(config.name));

const net = new Sequential({ inputSize: 2, seed: 1, layers: [new FeatureScale(), dense(1)] });
net.compile({ loss: "mse" });
net.fit([[1, 2], [3, 4]], [1, 2], { epochs: 5 });
```

You can also implement the `Layer` interface directly (`type`, `name`, `built`, `outputSize`, `build`, `forward`,
`backward`, `parameters`, `getConfig`) without extending `BaseLayer`.

#### `registerLayer(type: string, fromConfig: (config: LayerConfig) => Layer): void`

Registers a custom layer type so `layerFromConfig`, and therefore `loadModel` / `deserializeModel`, can rebuild it.
Built-in types cannot be replaced.

#### `layerFromConfig(config: LayerConfig): Layer`

Rebuilds an unbuilt layer (without weights) from `getConfig()` output. Throws a `ValidationError` for unknown types.

---

## Activations

Pass an activation as a name, a config object, or an `Activation` instance (`ActivationIdentifier`).

| Name | Formula / notes | Parameters |
|---|---|---|
| `linear` | `z` | |
| `sigmoid` | `1 / (1 + e^−z)` | |
| `tanh` | `tanh(z)` | |
| `relu` | `max(0, z)` | |
| `relu6` | `min(max(0, z), 6)` | |
| `leakyRelu` | `z` if `z > 0`, else `alpha · z` | `alpha` (default `0.01`) |
| `elu` | `z` if `z > 0`, else `alpha · (e^z − 1)` | `alpha` (default `1`) |
| `selu` | Scaled ELU with the constants of Klambauer et al. (2017) | |
| `gelu` | Tanh approximation of the Gaussian error linear unit | |
| `swish` | `z · sigmoid(z)` | |
| `mish` | `z · tanh(softplus(z))` | |
| `softplus` | `ln(1 + e^z)` | |
| `softsign` | `z / (1 + \|z\|)` | |
| `hardSigmoid` | `clamp(0.2 · z + 0.5, 0, 1)` | |
| `softmax` | Row-wise `e^z / Σ e^z` | |

- **`ACTIVATION_NAMES: readonly ActivationName[]`**: every name above.
- **`getActivation(id: ActivationIdentifier): Activation`** resolves an identifier; instances are returned as-is.

```ts
import { getActivation, Matrix } from "@zzza38/orion-engine";

const leaky = getActivation({ name: "leakyRelu", alpha: 0.2 });
console.log(leaky.forward(Matrix.from([[-1, 2]])).toArray()); // [[-0.2, 2]]
```

The `Activation` interface: `name`, `forward(z, out?)`, `backward(z, a, gradOutput, out?)` (returns dL/dz) and
`getConfig()`.

## Losses

| Name | Alias | Per-sample loss | Reduction |
|---|---|---|---|
| `meanSquaredError` | `mse` | `(p − y)²` | Mean over all elements |
| `meanAbsoluteError` | `mae` | `\|p − y\|` | Mean over all elements |
| `huber` | | `½e²` if `\|e\| ≤ delta`, else `delta · (\|e\| − ½delta)`; `delta` default `1` | Mean over all elements |
| `binaryCrossentropy` | `bce` | `−y·ln p − (1 − y)·ln(1 − p)` | Mean over all elements |
| `categoricalCrossentropy` | `cce` | `−Σ y·ln p` over classes | Mean over samples |
| `sparseCategoricalCrossentropy` | `scce` | `−ln p[label]`; targets are class indices | Mean over samples |

Cross-entropies clamp probabilities to [1e-7, 1 − 1e-7]. With a `sigmoid` output and `bce`, or a `softmax` output and
`cce` / `scce`, training uses a fused closed-form gradient with respect to the pre-activation (`(p − y) / batch` for
one-hot targets), which is faster and numerically stabler. It is exact for any targets, including soft labels and
multi-hot rows.

- **`LOSS_NAMES: readonly LossName[]`**: the canonical names (without aliases).
- **`getLoss(id: LossIdentifier): Loss`** resolves a name, alias, config (`{ name: "huber", delta: 2 }`) or instance.

The `Loss` interface: `name`, `compute(prediction, target)` (the batch-mean loss), `gradient(prediction, target, out?)`,
optional `fusedGradient(activation, prediction, target, out?)` and `getConfig()`.

## Metrics

| Name | Alias | Description |
|---|---|---|
| `accuracy` | | With a `binaryCrossentropy` loss, always `binaryAccuracy` (so multi-label models work). Otherwise chosen from the shapes: `binaryAccuracy` for one output column, `sparseCategoricalAccuracy` for several outputs and one target column, `categoricalAccuracy` otherwise |
| `binaryAccuracy` | | Fraction of elements on the same side of 0.5 as the target |
| `categoricalAccuracy` | | Fraction of rows where `argmax(prediction) = argmax(target)` |
| `sparseCategoricalAccuracy` | | Fraction of rows where `argmax(prediction)` equals the class-index target |
| `meanSquaredError` | `mse` | |
| `meanAbsoluteError` | `mae` | |
| `rootMeanSquaredError` | `rmse` | Epoch values are exact: the model averages the squared error and takes the root at the end |

Metrics appear in logs under their canonical name (`meanAbsoluteError`, not `mae`).

- **`METRIC_NAMES: readonly MetricName[]`**: the canonical names.
- **`getMetric(id: MetricIdentifier): Metric`** resolves a name, alias or `Metric` object `{ name, compute(prediction,
  target) }`. Its `"accuracy"` always chooses from the shapes; the `bce` rule above is applied by `compile`.

## Initializers

| Name | Distribution | Parameters |
|---|---|---|
| `zeros`, `ones` | Constant 0 / 1 | |
| `constant` | Constant | `value` (default `0`) |
| `randomUniform` | U(minval, maxval) | `minval` (`-0.05`), `maxval` (`0.05`) |
| `randomNormal` | N(mean, stddev²) | `mean` (`0`), `stddev` (`0.05`) |
| `glorotUniform`, `glorotNormal` | Variance `1 / fanAvg`, where `fanAvg = (fanIn + fanOut) / 2` | |
| `heUniform`, `heNormal` | Variance `2 / fanIn` | |
| `lecunUniform`, `lecunNormal` | Variance `1 / fanIn` | |

Uniform variants sample U(−limit, limit) with `limit = sqrt(3 · variance)`; normal variants sample a normal truncated at
±2σ, rescaled so the result has the requested variance (as in Keras).

- **`INITIALIZER_NAMES: readonly InitializerName[]`**.
- **`getInitializer(id: InitializerIdentifier): Initializer`** resolves a name, a config
  (`{ name: "constant", value: 0.1 }`) or an instance with `initialize(target, fanIn, fanOut, rng)` and `getConfig()`.

---

## Optimizers

Pass an optimizer to `compile` as a name, a config `{ name, ...options }`, or an instance. Every optimizer accepts:

| Option | Default | Description |
|---|---|---|
| `learningRate` | per optimizer (below) | Step size; finite and ≥ 0 |
| `clipNorm` | none | Rescale each parameter's gradient to an L2 norm of at most this value (> 0) |
| `clipValue` | none | Clamp every gradient element to `[-clipValue, clipValue]` (> 0); applied after `clipNorm` |

| Name | Class | `learningRate` | Other options (defaults) |
|---|---|---|---|
| `sgd` | `SGD` | `0.01` | `momentum` (`0`), `nesterov` (`false`), `weightDecay` (`0`, coupled L2 added to the gradient) |
| `adam` | `Adam` | `0.001` | `beta1` (`0.9`), `beta2` (`0.999`), `epsilon` (`1e-7`), `amsgrad` (`false`) |
| `adamw` | `AdamW` | `0.001` | As Adam, plus `weightDecay` (`0.01`, decoupled) |
| `rmsprop` | `RMSprop` | `0.001` | `rho` (`0.9`), `momentum` (`0`), `epsilon` (`1e-7`), `centered` (`false`) |
| `adagrad` | `Adagrad` | `0.01` | `initialAccumulatorValue` (`0.1`), `epsilon` (`1e-7`) |

Weight decay never applies to parameters with `regularize: false` (biases, batch normalization). Parameters with
`trainable: false` are skipped entirely.

```ts
import { Adam, getOptimizer } from "@zzza38/orion-engine";

model.compile({ loss: "bce", optimizer: { name: "sgd", learningRate: 0.1, momentum: 0.9, nesterov: true } });
model.compile({ loss: "bce", optimizer: new Adam({ learningRate: 0.01, clipNorm: 1 }) });
console.log(getOptimizer("rmsprop").getConfig()); // { name: "rmsprop", learningRate: 0.001, rho: 0.9, … }
```

- **`OPTIMIZER_NAMES: readonly OptimizerName[]`**: `"sgd"`, `"adam"`, `"adamw"`, `"rmsprop"`, `"adagrad"`.
- **`getOptimizer(id: OptimizerIdentifier): Optimizer`**: a name (case-insensitive) or config creates a new optimizer
  with fresh state; an instance is returned as-is.

The `Optimizer` interface (implemented by every class via `BaseOptimizer`):

| Member | Description |
|---|---|
| `name` | The optimizer name |
| `learningRate` | Current learning rate; settable (callbacks and schedules change it between steps) |
| `iterations` | Number of `step` calls since construction or `reset()` |
| `step(params)` | Applies one update to every trainable parameter from its `grad` |
| `reset()` | Clears per-parameter state (moments, accumulators) and `iterations` |
| `getConfig()` | A JSON-safe config; `getOptimizer(config)` builds an equivalent optimizer with fresh state |

`BaseOptimizer` is the abstract base class of the built-ins: subclasses implement `update(param, grad, lr, t)` for one
parameter and `hyperparameters()`, and can keep per-parameter state with `slots(param, count, initial?)`.

## Learning-rate schedules

A `LearningRateSchedule` is a function `(epoch: number) => number` from a 0-based epoch to a learning rate. Use one with
`learningRateScheduler(schedule)`, or call it yourself.

| Factory | Formula | Options |
|---|---|---|
| `constantSchedule(learningRate)` | `learningRate` | |
| `stepDecay(options)` | `initial · factor^⌊epoch / every⌋` | `initial`, `factor` in (0, 1], `every` (positive integer) |
| `exponentialDecay(options)` | `initial · rate^(epoch / every)` | `initial`, `rate` in (0, 1], `every` (default `1`) |
| `cosineDecay(options)` | `minimum + (initial − minimum) · ½(1 + cos(π · min(epoch, epochs) / epochs))` | `initial`, `epochs`, `minimum` (default `0`) |
| `piecewiseConstant(options)` | `values[k]` for the k-th interval between `boundaries` | `boundaries` (increasing epochs), `values` (one more than `boundaries`) |
| `linearWarmup(schedule, options)` | From `from` to `schedule(0)` over `epochs` epochs, then `schedule(epoch − epochs)` | `epochs`, `from` (default `0`) |

```ts
import { cosineDecay, linearWarmup, piecewiseConstant, stepDecay } from "@zzza38/orion-engine";

stepDecay({ initial: 0.1, factor: 0.5, every: 10 })(25); // 0.025
piecewiseConstant({ boundaries: [10, 20], values: [0.1, 0.01, 0.001] })(15); // 0.01
const warm = linearWarmup(cosineDecay({ initial: 1e-3, epochs: 100 }), { epochs: 5 }); // 5 warm-up epochs, then 100
console.log(warm(0), warm(5), warm(105)); // 0 0.001 0
```

---

## Callbacks and History

### `Callback`

An object with any of these optional hooks (each may return a Promise, which `fitAsync` awaits and `fit` rejects):

| Hook | Arguments |
|---|---|
| `onTrainBegin(ctx)` | |
| `onEpochBegin(epoch, ctx)` | 0-based epoch |
| `onBatchEnd(batch, logs, ctx)` | 0-based batch; `logs` has the batch's `loss`, metrics (before the update) and `size` |
| `onEpochEnd(epoch, logs, ctx)` | The epoch's logs |
| `onTrainEnd(logs, ctx)` | The last epoch's logs (empty if no epoch ran) |

`ctx` is a `CallbackContext`:

| Member | Description |
|---|---|
| `model`, `optimizer` | The model being trained and its optimizer (whose `learningRate` may be changed) |
| `initialEpoch` | Number of the first epoch of this `fit` call (the `initialEpoch` option, default `0`) |
| `epochs` | One past the last epoch number: `initialEpoch + epochs` of the `fit` call |
| `batchSize`, `samples`, `stepsPerEpoch` | Batch size, training samples (after any validation split) and batches per epoch |
| `hasValidation` | True when validation metrics are computed |
| `history` | The `History` recorded so far |
| `stopRequested`, `stopTraining()` | `stopTraining()` ends training after the current epoch |

Callbacks must not call `compile` or `add` on the model while it trains; both throw.

`Logs` is `Record<string, number>`. `MonitorMode` is `"min" | "max" | "auto"`; `"auto"` maximizes keys containing
`acc` or `auc` and minimizes everything else.

### `History`

Returned by `fit` / `fitAsync`.

| Member | Description |
|---|---|
| `epochs: number[]` | 0-based indices of the completed epochs |
| `history: Record<string, number[]>` | One array per log key, aligned with `epochs` (`NaN` where a key was missing) |
| `length` | Number of recorded epochs |
| `last(key)` | The latest value of `key`, or `undefined` |
| `best(key, mode = "auto")` | `{ epoch, value }` of the best value (first on ties), or `undefined` |
| `append(epoch, logs)` | Records one epoch (used by the model) |
| `toJSON()` | A plain `{ epochs, history }` copy |

### `earlyStopping(options?: EarlyStoppingOptions): EarlyStoppingCallback`

Stops training once `monitor` has not improved for `patience` consecutive epochs. State resets at the start of every
`fit`, so the same callback can be reused.

| Option | Default | Description |
|---|---|---|
| `monitor` | `"valLoss"` | Log key to watch; without validation data `"valX"` falls back to `"x"` with a warning |
| `patience` | `0` | Epochs without improvement before stopping (0 stops at the first one) |
| `minDelta` | `0` | Minimum change that counts as an improvement |
| `mode` | `"auto"` | `"min"`, `"max"` or `"auto"` |
| `restoreBestWeights` | `false` | When training ends, restore the weights from the best epoch |

The returned object also exposes `stoppedEpoch` (or `null` if training ran to completion), `bestEpoch` and `bestValue`.
Best weights are restored in `onTrainEnd`, so they are not restored when training ends with an error (an aborted
`fitAsync`, a `TrainingError`).

### `learningRateScheduler(schedule: (epoch: number, learningRate: number) => number): Callback`

Sets the optimizer's learning rate at the start of every epoch. The function receives the epoch number (which starts
at `initialEpoch`) and the current rate, and must return a finite number ≥ 0.

```ts
import { cosineDecay, learningRateScheduler } from "@zzza38/orion-engine";

model.fit(x, y, { epochs: 50, batchSize: 4, callbacks: [learningRateScheduler(cosineDecay({ initial: 0.05, epochs: 50 }))] });
model.fit(x, y, { epochs: 50, batchSize: 4, callbacks: [learningRateScheduler((epoch, lr) => (epoch < 10 ? lr : lr * 0.95))] });
```

### `reduceLROnPlateau(options?: ReduceLROnPlateauOptions): Callback`

Multiplies the learning rate by `factor` when `monitor` has not improved for `patience` epochs.

| Option | Default | Description |
|---|---|---|
| `monitor` | `"valLoss"` | As for `earlyStopping` |
| `factor` | `0.1` | Multiplier in (0, 1) |
| `patience` | `10` | Epochs without improvement before reducing |
| `minDelta` | `1e-4` | Minimum change that counts as an improvement |
| `mode` | `"auto"` | `"min"`, `"max"` or `"auto"` |
| `cooldown` | `0` | Epochs to wait after a reduction before counting again |
| `minLearningRate` | `0` | Lower bound for the learning rate |

### `progressLogger(options?: ProgressLoggerOptions): Callback`

Logs a line such as `Epoch 10/100 - loss: 0.2143 - accuracy: 0.9375 - 4ms` every `every` epochs (default `1`; the
last epoch, and an epoch where training stops early, are always logged) through `log` (default `console.log`).
`fit(x, y, { verbose })` installs one.

### `formatEpoch(epoch: number, epochs: number, logs: Logs): string`

Formats one progress line as `progressLogger` does (`epoch` is 1-based here).

---

## Data utilities

### `oneHot(labels: ArrayLike<number>, numClasses?: number): number[][]`

One-hot encodes integer class labels. `numClasses` defaults to the largest label + 1.

### `argmax(values)`

`argmax(values: number[]): number` returns the index of the largest value (the first on ties);
`argmax(rows: number[][] | Matrix): number[]` returns it for every row.

```ts
import { argmax, oneHot } from "@zzza38/orion-engine";

oneHot([0, 2, 1]); // [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
argmax([0.1, 0.7, 0.2]); // 1
argmax([[0.9, 0.1], [0.2, 0.8]]); // [0, 1]
```

### `trainTestSplit(x, y, options?: TrainTestSplitOptions): TrainTestSplit`

Splits paired inputs and targets into `{ xTrain, xTest, yTrain, yTest }`. Arrays come back as arrays (rows are copied)
and matrices as matrices.

| Option | Default | Description |
|---|---|---|
| `testSize` | `0.2` | A fraction in (0, 1) (the test set gets `ceil(testSize · n)` samples) or an absolute count |
| `shuffle` | `true` | Shuffle before splitting; without it the test set is the last samples |
| `seed` | random | Seed for the shuffle |

### `shuffleTogether(x, y, rng?: Random | number): [x, y]`

Shuffles two datasets with the same permutation and returns new arrays (or matrices). `rng` is a `Random`, a seed, or
nothing for a random shuffle.

### `StandardScaler`

Standardizes each feature: `(x − mean) / std`, using the population standard deviation. Constant features (variance
zero, or within rounding error of it) get `std` 1 and transform to 0, as in scikit-learn.

### `MinMaxScaler`

`new MinMaxScaler({ featureRange?: [min, max] })` rescales each feature linearly so the fitted data spans
`featureRange` (default `[0, 1]`). Constant features map to the range minimum.

Both scalers have:

| Member | Description |
|---|---|
| `fit(x): this` | Learns per-feature statistics from rows (or a `Matrix`) |
| `transform(x)` | Scales `x`; returns the same kind it was given (`Matrix`, rows, or one sample) |
| `fitTransform(x)` | `fit(x)` then `transform(x)` |
| `inverseTransform(x)` | Undoes `transform` |
| `fitted` | True after `fit` |
| `toJSON()` / `static fromJSON(json)` | Save and restore (`StandardScalerJSON`, `MinMaxScalerJSON`) |
| `mean`, `std` (Standard) / `dataMin`, `dataMax`, `featureRange` (MinMax) | The learned statistics (copies) |

```ts
import { MinMaxScaler, StandardScaler } from "@zzza38/orion-engine";

const scaler = new MinMaxScaler({ featureRange: [-1, 1] });
scaler.fitTransform([[0, 10], [5, 20], [10, 30]]); // [[-1, -1], [0, 0], [1, 1]]
scaler.transform([5, 30]); // [0, 1]

const standard = StandardScaler.fromJSON({ type: "standardScaler", mean: [2], std: [0.5] });
standard.transform([[3]]); // [[2]]
```

Types: `Samples` (a `Matrix` or an array with one entry per sample) and `SamplesOf<T>` (what the split and shuffle
functions return for input type `T`).

---

## Saving and loading

A saved model holds the architecture, all weights (including batch normalization's moving statistics), the model name
and your metadata, and, when compiled, the loss, optimizer config and metrics. Optimizer state (such as Adam's moments)
is not saved; a loaded model starts with a fresh optimizer. The optimizer config records the **current** learning rate:
a model saved at the end of a decaying schedule reloads with that small rate, so recompile it (or train with a
schedule) before training it further.

### `serializeModel(model: Sequential, options?: SerializeOptions): Uint8Array | string`

Encodes a built model. Returns a `Uint8Array` for `format: "binary"` and a `string` for `format: "json"` (the return
type follows the option).

| Option | Default | Description |
|---|---|---|
| `format` | `"binary"` | `"binary"` (compact, checksummed) or `"json"` (human-readable, bit-exact) |
| `precision` | `"float32"` | Binary only: `"float32"` halves the size (~7 significant digits); `"float64"` is bit-exact |
| `pretty` | `false` | JSON only: indent the document |
| `metadata` | none | Extra JSON-safe metadata to store (the model name is stored as `metadata.name`) |

### `deserializeModel(data: string | Uint8Array | ArrayBuffer, options?: DeserializeOptions): Sequential`

Decodes a model saved by `serializeModel` / `saveModel` (binary or JSON) or a legacy 0.0.x `.onn` text model; the
format is detected from the content. `options.seed` seeds the loaded model's shuffling and dropout (default random).
Throws a `SerializationError` for invalid data, including well-formed files that describe an impossible model (an
unregistered layer type, weights of the wrong shape); the underlying error is in its `cause`.

```ts
import { deserializeModel, serializeModel } from "@zzza38/orion-engine";

const text = serializeModel(model, { format: "json", pretty: true, metadata: { version: 3 } });
const restored = deserializeModel(text, { seed: 1 });
console.log(restored.predict([1, 0]), restored.compiled); // same prediction, true
```

### Node.js: `@zzza38/orion-engine/node`

- **`saveModel(model: Sequential, path: string, options?: SaveModelOptions): Promise<void>`** writes a model file,
  creating parent directories. `SaveModelOptions` are `SerializeOptions`; when `format` is omitted it follows the path:
  names ending in `.json` (such as `model.onn.json`) are written as JSON, anything else as binary.
- **`saveModelSync(model, path, options?): void`**: the synchronous version.
- **`loadModel(path: string, options?: LoadModelOptions): Promise<Sequential>`** reads a model file of any supported
  format. `LoadModelOptions` are `DeserializeOptions`.
- **`loadModelSync(path, options?): Sequential`**: the synchronous version.

```ts
import { loadModel, saveModel } from "@zzza38/orion-engine/node";

await saveModel(model, "out/xor.onn", { precision: "float64", metadata: { dataset: "xor" } });
const restored = await loadModel("out/xor.onn");
```

---

## Model formats (low level)

Most code only needs the functions above. These lower-level pieces read and write the [documented file
formats](format.md) without building a model.

### `ModelArtifact`

The format-independent, in-memory description of a saved model (`Sequential.toArtifact()` creates one):

| Key | Type | Description |
|---|---|---|
| `format` | `"orion-engine"` | Format marker |
| `formatVersion` | `1` | Schema version |
| `inputSize` | `number` | Number of input features |
| `layers` | `LayerConfig[]` | Each layer's `getConfig()`: `{ type, name, … }` |
| `weights` | `WeightEntry[]` | `{ name, shape: [rows, cols], data }` in `parameters()` order; `data` is row-major |
| `training` | `TrainingConfig` (optional) | `{ loss: LossConfig, optimizer: OptimizerConfig, metrics: MetricName[] }` |
| `metadata` | JSON object (optional) | Free-form |

`JsonValue` is the type of JSON-representable values used in configs and metadata.

### Codecs

| Function | Description |
|---|---|
| `detectFormat(data: string \| Uint8Array \| ArrayBuffer): ArtifactFormat` | `"binary"`, `"json"` or `"legacy"`, from the first bytes |
| `decodeArtifact(data): ModelArtifact` | Decodes any supported format (detected automatically) |
| `encodeArtifact(artifact, options: EncodeArtifactOptions): Uint8Array \| string` | `{ format: "binary", precision? }` or `{ format: "json", pretty? }` |
| `encodeBinary(artifact, options?: BinaryEncodeOptions): Uint8Array` | The binary container; `precision` is `"float32"` (default) or `"float64"` (`BinaryPrecision`) |
| `decodeBinary(bytes: Uint8Array \| ArrayBuffer): ModelArtifact` | Verifies the CRC-32 and the layout |
| `encodeJson(artifact, options?: JsonEncodeOptions): string` | JSON text; `pretty` (default `false`) indents it |
| `decodeJson(text: string): ModelArtifact` | Parses and validates JSON text |
| `decodeLegacyOnn(text: string): ModelArtifact` | Converts a 0.0.x `.onn` text model |
| `validateArtifact(value: unknown): ModelArtifact` | Checks the structure and returns the value, typed |
| `crc32(bytes: Uint8Array): number` | The CRC-32 (zlib/PNG) of some bytes |

Decoders return weight data as `Float64Array`s and throw a `SerializationError` naming the offending field.

```ts
import { decodeLegacyOnn, encodeBinary, encodeJson } from "@zzza38/orion-engine";

const artifact = decodeLegacyOnn("2:relu:2:swish\n0.71:-0.2:0.19|-1.82:0.95:0.97");
const bytes = encodeBinary(artifact, { precision: "float64" }); // upgrade to the binary format
console.log(bytes.byteLength, encodeJson(artifact).length);
```

---

## Matrix and math

### `Matrix`

A dense, row-major 2-D matrix backed by a `Float64Array`. Rows are samples; columns are features.

| Member | Description |
|---|---|
| `new Matrix(rows, cols, data?)` | Zero-filled, or wrapping `data` (a `Float64Array` is used as-is, other arrays are copied) |
| `Matrix.zeros(rows, cols)`, `Matrix.filled(rows, cols, value)` | Constant matrices |
| `Matrix.fromArray(rows: number[][])` | From rows; throws on ragged rows or `NaN` |
| `Matrix.fromVector(values)` | A `[1, n]` row |
| `Matrix.from(value: MatrixLike)` | Any `MatrixLike`; matrices are returned as-is, a flat array becomes one row |
| `rows`, `cols`, `data`, `size`, `shape` | Dimensions and the underlying `Float64Array` |
| `get(row, col)`, `set(row, col, value)` | Element access |
| `row(index)` | A live `Float64Array` view of one row |
| `toArray(): number[][]`, `clone()`, `fill(value)`, `copyFrom(other)` | Conversion and copying |
| `map(fn, out?)` | Element-wise `fn(value, row, col)` into a new (or the given) matrix |
| `hasShape(rows, cols)`, `toString()` | Helpers |

`MatrixLike` is `Matrix | number[][] | number[]`.

### Functions

All take an optional `out` matrix of the result's shape to write into (a new matrix is allocated otherwise) and throw
a `ShapeError` when shapes do not match. The element-wise functions may write into one of their inputs; the others
(`matmul*`, `transpose`, `sumRows`, `gatherRows`) throw a `ValidationError` if `out` shares memory with an input.
`NaN` and `Infinity` propagate as in plain arithmetic.

| Function | Result |
|---|---|
| `add(a, b, out?)`, `subtract(a, b, out?)`, `multiply(a, b, out?)` | Element-wise `a + b`, `a − b`, `a ⊙ b` |
| `scale(a, scalar, out?)` | `a · scalar` |
| `addRowVector(a, v, out?)` | Adds the row vector `v` (a `[1, cols]` matrix or `Float64Array`) to every row |
| `matmul(a, b, out?)` | `a · b` |
| `matmulTransposeA(a, b, out?)` | `aᵀ · b` |
| `matmulTransposeB(a, b, out?)` | `a · bᵀ` |
| `transpose(a, out?)` | `aᵀ` |
| `sumRows(a, out?)` | Column sums, `[1, cols]` |
| `gatherRows(a, indices, out?)` | The given rows, in order |
| `sliceRows(a, start, end)` | A copy of rows `[start, end)` |
| `argmaxRows(a)` | `Int32Array` of each row's largest index |

```ts
import { addRowVector, Matrix, matmul } from "@zzza38/orion-engine";

const inputs = Matrix.from([[1, 2], [3, 4]]);
const kernel = Matrix.from([[0.5], [-1]]);
console.log(addRowVector(matmul(inputs, kernel), Matrix.from([0.1])).toArray()); // [[-1.4], [-2.4]]
```

## `Random`

A seedable pseudo-random generator (xoshiro128\*\*); every random choice in the library goes through one.

| Member | Description |
|---|---|
| `new Random(seed?)` | A 32-bit integer seed; omit it for a random one |
| `seed` | The seed |
| `next()` | Uniform in [0, 1) |
| `nextUint32()` | A raw 32-bit unsigned integer |
| `uniform(min = 0, max = 1)` | Uniform in [min, max) |
| `int(maxExclusive)` | Integer in [0, maxExclusive) |
| `normal(mean = 0, stddev = 1)` | Normal sample |
| `truncatedNormal(mean = 0, stddev = 1)` | Normal sample within 2 standard deviations |
| `shuffle(array)` | Fisher-Yates shuffle, in place; returns the array |
| `fork()` | An independent generator derived from this one |

```ts
import { Random } from "@zzza38/orion-engine";

const rng = new Random(123);
const noise = Array.from({ length: 3 }, () => rng.normal(0, 0.1));
console.log(noise.length, rng.shuffle([1, 2, 3, 4]));
```

## Errors

Every error thrown by the library is an `OrionError` (which extends `Error`; its `name` is the class name). The
constructors take `(message, options?: { cause?: unknown })`, like `Error`.

| Class | Thrown when |
|---|---|
| `ValidationError` | An argument, option or config is invalid, or the model is in the wrong state |
| `ShapeError` | Matrix, input, target or weight shapes do not line up |
| `TrainingError` | Training cannot continue, e.g. the loss became `NaN` or infinite |
| `SerializationError` | Model data cannot be decoded |

```ts
import { ShapeError } from "@zzza38/orion-engine";

try {
    model.predict([1, 2, 3]);
} catch (error) {
    if (error instanceof ShapeError) console.log(error.message); // Expected input with 2 features (inputSize), got 3
}
```

## `VERSION`

The library version as a string (`"0.1.0"`).

---

## Type index

Every exported type, and where it is described:

| Types | Section |
|---|---|
| `SequentialOptions`, `CompileOptions`, `FitOptions`, `FitAsyncOptions`, `BatchOptions` | [Sequential](#sequential) |
| `Layer`, `LayerConfig`, `LayerOptions`, `LayerFromConfig`, `Parameter`, `DenseOptions`, `RegularizerOptions`, `DropoutOptions`, `BatchNormalizationOptions`, `ActivationLayerOptions` | [Layers](#layers) |
| `Activation`, `ActivationConfig`, `ActivationIdentifier`, `ActivationName` | [Activations](#activations) |
| `Loss`, `LossConfig`, `LossIdentifier`, `LossName`, `LossAlias` | [Losses](#losses) |
| `Metric`, `MetricIdentifier`, `MetricName` | [Metrics](#metrics) |
| `Initializer`, `InitializerConfig`, `InitializerIdentifier`, `InitializerName` | [Initializers](#initializers) |
| `Optimizer`, `OptimizerConfig`, `OptimizerIdentifier`, `OptimizerName`, `OptimizerOptions`, `SGDOptions`, `AdamOptions`, `AdamWOptions`, `RMSpropOptions`, `AdagradOptions` | [Optimizers](#optimizers) |
| `LearningRateSchedule`, `StepDecayOptions`, `ExponentialDecayOptions`, `CosineDecayOptions`, `LinearWarmupOptions`, `PiecewiseConstantOptions` | [Schedules](#learning-rate-schedules) |
| `Callback`, `CallbackContext`, `Logs`, `MonitorMode`, `EarlyStoppingOptions`, `EarlyStoppingCallback`, `ReduceLROnPlateauOptions`, `ProgressLoggerOptions` | [Callbacks](#callbacks-and-history) |
| `Samples`, `SamplesOf`, `TrainTestSplit`, `TrainTestSplitOptions`, `StandardScalerJSON`, `MinMaxScalerJSON`, `MinMaxScalerOptions` | [Data utilities](#data-utilities) |
| `SerializeOptions`, `DeserializeOptions`, `SaveModelOptions`, `LoadModelOptions` | [Saving and loading](#saving-and-loading) |
| `ModelArtifact`, `WeightEntry`, `TrainingConfig`, `JsonValue`, `ArtifactFormat`, `EncodeArtifactOptions`, `BinaryEncodeOptions`, `BinaryPrecision`, `JsonEncodeOptions` | [Model formats](#model-formats-low-level) |
| `MatrixLike` | [Matrix](#matrix) |
