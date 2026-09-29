/**
 * Shared contracts for every Orion Engine module.
 *
 * Conventions:
 *  - Batches are Matrix [batchSize, features]. Row = sample.
 *  - Dense kernels are Matrix [inputSize, units]; biases are Matrix [1, units].
 *    Forward pass: Y = X · W + b.
 *  - Loss values are the mean over the batch. Loss gradients are dL/dPred of that mean,
 *    so they are already divided by the batch size.
 *  - All randomness goes through a `Random` instance.
 */
import type { Matrix } from "./matrix.js";
import type { Random } from "./random.js";

/**
 * JSON-compatible value, used for configs that round-trip through serialization.
 * Object members may be `undefined` (dropped by JSON.stringify) so optional config fields nest cleanly.
 */
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue | undefined };

// ---------------------------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------------------------

export type ActivationName =
    | "linear"
    | "sigmoid"
    | "tanh"
    | "relu"
    | "relu6"
    | "leakyRelu"
    | "elu"
    | "selu"
    | "gelu"
    | "swish"
    | "mish"
    | "softplus"
    | "softsign"
    | "hardSigmoid"
    | "softmax";

/** Serializable activation description. Extra numeric params (e.g. `alpha`) are activation-specific. */
export interface ActivationConfig {
    name: ActivationName;
    [param: string]: JsonValue | undefined;
}

export interface Activation {
    readonly name: ActivationName;
    /** a = f(z). `z` is [batch, units]. */
    forward(z: Matrix, out?: Matrix): Matrix;
    /**
     * dL/dz given the pre-activation `z`, the activation output `a` (= forward(z)), and dL/da.
     * Handles non element-wise activations (softmax) with the full Jacobian-vector product per row.
     */
    backward(z: Matrix, a: Matrix, gradOutput: Matrix, out?: Matrix): Matrix;
    getConfig(): ActivationConfig;
}

export type ActivationIdentifier = ActivationName | ActivationConfig | Activation;

// ---------------------------------------------------------------------------------------------
// Losses
// ---------------------------------------------------------------------------------------------

export type LossName =
    | "meanSquaredError"
    | "meanAbsoluteError"
    | "huber"
    | "binaryCrossentropy"
    | "categoricalCrossentropy"
    | "sparseCategoricalCrossentropy";

export interface LossConfig {
    name: LossName;
    [param: string]: JsonValue | undefined;
}

export interface Loss {
    readonly name: LossName;
    /** Mean loss over the batch. `target` is [batch, units], or [batch, 1] class indices for sparse losses. */
    compute(prediction: Matrix, target: Matrix): number;
    /** dL/dPrediction of the batch-mean loss (already divided by batch size). Shape matches `prediction`. */
    gradient(prediction: Matrix, target: Matrix, out?: Matrix): Matrix;
    /**
     * Optional closed-form dL/dz for (output activation, loss) pairs that simplify, e.g.
     * sigmoid + binaryCrossentropy → (p - y) / (batch·units), or softmax + categoricalCrossentropy
     * → (p - y) / batch for one-hot targets.
     * Returns null when the pair has no fused form. Models should prefer this when available:
     * it is faster and numerically stabler.
     */
    fusedGradient?(activation: ActivationName, prediction: Matrix, target: Matrix, out?: Matrix): Matrix | null;
    getConfig(): LossConfig;
}

/** Short aliases accepted anywhere a loss is accepted: "mse", "mae", "bce", "cce", "scce". */
export type LossAlias = "mse" | "mae" | "bce" | "cce" | "scce";
export type LossIdentifier = LossName | LossAlias | LossConfig | Loss;

// ---------------------------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------------------------

export type MetricName =
    | "accuracy"
    | "binaryAccuracy"
    | "categoricalAccuracy"
    | "sparseCategoricalAccuracy"
    | "meanSquaredError"
    | "meanAbsoluteError"
    | "rootMeanSquaredError";

/**
 * Metrics must be batch-decomposable: the model reports the batch-size-weighted mean of
 * `compute` across batches (rootMeanSquaredError is the exception: models average its square).
 */
export interface Metric {
    readonly name: MetricName;
    compute(prediction: Matrix, target: Matrix): number;
}

export type MetricIdentifier = MetricName | "mse" | "mae" | "rmse" | Metric;

// ---------------------------------------------------------------------------------------------
// Initializers
// ---------------------------------------------------------------------------------------------

export type InitializerName =
    | "zeros"
    | "ones"
    | "constant"
    | "randomUniform"
    | "randomNormal"
    | "glorotUniform"
    | "glorotNormal"
    | "heUniform"
    | "heNormal"
    | "lecunUniform"
    | "lecunNormal";

export interface InitializerConfig {
    name: InitializerName;
    [param: string]: JsonValue | undefined;
}

export interface Initializer {
    readonly name: InitializerName;
    /** Fills `target` in place. fanIn/fanOut are the layer's input/output sizes. */
    initialize(target: Matrix, fanIn: number, fanOut: number, rng: Random): void;
    getConfig(): InitializerConfig;
}

export type InitializerIdentifier = InitializerName | InitializerConfig | Initializer;

// ---------------------------------------------------------------------------------------------
// Parameters & optimizers
// ---------------------------------------------------------------------------------------------

/** A trainable tensor and its gradient buffer (same shape). Optimizers key their state by object identity. */
export interface Parameter {
    /** Unique within a model, e.g. "dense_1/kernel". */
    readonly name: string;
    readonly value: Matrix;
    readonly grad: Matrix;
    /** When false, optimizers skip this parameter. */
    trainable: boolean;
    /** When false, weight decay / regularization is not applied (biases). */
    readonly regularize: boolean;
}

export type OptimizerName = "sgd" | "adam" | "adamw" | "rmsprop" | "adagrad";

export interface OptimizerConfig {
    name: OptimizerName;
    learningRate?: number;
    /** Clip each parameter's gradient to this L2 norm before the update. */
    clipNorm?: number;
    /** Clip every gradient element to [-clipValue, clipValue] before the update. */
    clipValue?: number;
    [param: string]: JsonValue | undefined;
}

export interface Optimizer {
    readonly name: OptimizerName;
    /** Current learning rate. Mutable so schedules/callbacks can adjust it between steps. */
    learningRate: number;
    /** Number of `step` calls since construction or the last `reset`. */
    readonly iterations: number;
    /** Applies one update to every trainable parameter using its `grad`. Does not zero gradients. */
    step(params: readonly Parameter[]): void;
    /** Clears all internal state (moments, iteration count). */
    reset(): void;
    getConfig(): OptimizerConfig;
}

export type OptimizerIdentifier = OptimizerName | OptimizerConfig | Optimizer;

// ---------------------------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------------------------

export interface LayerConfig {
    type: string;
    name: string;
    [param: string]: JsonValue | undefined;
}

export interface Layer {
    readonly type: string;
    name: string;
    readonly built: boolean;
    /** Output feature count. Only valid after `build`. */
    readonly outputSize: number;
    /** Allocates parameters for the given input feature count. */
    build(inputSize: number, rng: Random): void;
    /** Forward pass over a batch. `training` toggles train-only behaviour (dropout). Caches what backward needs. */
    forward(input: Matrix, training: boolean): Matrix;
    /** Given dL/dOutput for the last forward batch, writes parameter grads and returns dL/dInput. */
    backward(gradOutput: Matrix): Matrix;
    parameters(): Parameter[];
    getConfig(): LayerConfig;
}

// ---------------------------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------------------------

export interface WeightEntry {
    /** Matches `Parameter.name`, e.g. "dense_1/kernel". */
    name: string;
    shape: [number, number];
    /** Row-major values. */
    data: Float64Array | Float32Array | number[];
}

export interface TrainingConfig {
    loss: LossConfig;
    optimizer: OptimizerConfig;
    metrics: MetricName[];
}

/** Format-agnostic, in-memory description of a saved model. JSON and binary encodings both decode to this. */
export interface ModelArtifact {
    format: "orion-engine";
    formatVersion: 1;
    inputSize: number;
    layers: LayerConfig[];
    /** In `model.parameters()` order. */
    weights: WeightEntry[];
    training?: TrainingConfig;
    metadata?: { [key: string]: JsonValue };
}
