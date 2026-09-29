/**
 * Orion Engine: a zero-dependency neural-network library for Node.js and the browser.
 *
 * @example
 * import { Sequential, dense } from "@zzza38/orion-engine";
 *
 * const model = new Sequential({ inputSize: 2, seed: 42, layers: [dense(8, "tanh"), dense(1, "sigmoid")] });
 * model.compile({ loss: "bce", optimizer: { name: "adam", learningRate: 0.05 }, metrics: ["accuracy"] });
 * model.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0], { epochs: 300, batchSize: 4 });
 * model.predict([1, 0]); // ≈ [1]
 *
 * This entry point is browser-safe. File helpers (saveModel / loadModel) live in
 * "@zzza38/orion-engine/node".
 *
 * @packageDocumentation
 */

// ---- Building blocks: registries, optimizers, schedules --------------------------------------
export { ACTIVATION_NAMES, getActivation } from "./activations.js";
export type {
    Callback,
    CallbackContext,
    EarlyStoppingCallback,
    EarlyStoppingOptions,
    Logs,
    MonitorMode,
    ProgressLoggerOptions,
    ReduceLROnPlateauOptions,
} from "./callbacks.js";
// ---- Callbacks ------------------------------------------------------------------------------
export {
    earlyStopping,
    formatEpoch,
    History,
    learningRateScheduler,
    progressLogger,
    reduceLROnPlateau,
} from "./callbacks.js";
export { OrionError, SerializationError, ShapeError, TrainingError, ValidationError } from "./core/errors.js";
export type { MatrixLike } from "./core/matrix.js";
// ---- Math, randomness, errors ----------------------------------------------------------------
export {
    add,
    addRowVector,
    argmaxRows,
    gatherRows,
    Matrix,
    matmul,
    matmulTransposeA,
    matmulTransposeB,
    multiply,
    scale,
    sliceRows,
    subtract,
    sumRows,
    transpose,
} from "./core/matrix.js";
export { Random } from "./core/random.js";
// ---- Contracts --------------------------------------------------------------------------------
export type {
    Activation,
    ActivationConfig,
    ActivationIdentifier,
    ActivationName,
    Initializer,
    InitializerConfig,
    InitializerIdentifier,
    InitializerName,
    JsonValue,
    Layer,
    LayerConfig,
    Loss,
    LossAlias,
    LossConfig,
    LossIdentifier,
    LossName,
    Metric,
    MetricIdentifier,
    MetricName,
    ModelArtifact,
    Optimizer,
    OptimizerConfig,
    OptimizerIdentifier,
    OptimizerName,
    Parameter,
    TrainingConfig,
    WeightEntry,
} from "./core/types.js";
export type {
    MinMaxScalerJSON,
    MinMaxScalerOptions,
    Samples,
    SamplesOf,
    StandardScalerJSON,
    TrainTestSplit,
    TrainTestSplitOptions,
} from "./data.js";
// ---- Data utilities -------------------------------------------------------------------------
export { argmax, MinMaxScaler, oneHot, StandardScaler, shuffleTogether, trainTestSplit } from "./data.js";
export { getInitializer, INITIALIZER_NAMES } from "./initializers.js";
export type {
    ArtifactFormat,
    BinaryEncodeOptions,
    BinaryPrecision,
    EncodeArtifactOptions,
    JsonEncodeOptions,
} from "./io/index.js";
export {
    crc32,
    decodeArtifact,
    decodeBinary,
    decodeJson,
    decodeLegacyOnn,
    detectFormat,
    encodeArtifact,
    encodeBinary,
    encodeJson,
    validateArtifact,
} from "./io/index.js";
export type {
    ActivationLayerOptions,
    BatchNormalizationOptions,
    DenseOptions,
    DropoutOptions,
    LayerFromConfig,
    LayerOptions,
    RegularizerOptions,
} from "./layers/index.js";
// ---- Layers ---------------------------------------------------------------------------------
export {
    ActivationLayer,
    activation,
    BaseLayer,
    BatchNormalization,
    batchNormalization,
    Dense,
    Dropout,
    dense,
    dropout,
    LAYER_TYPES,
    LayerParameter,
    layerFromConfig,
    registerLayer,
} from "./layers/index.js";
export { getLoss, LOSS_NAMES } from "./losses.js";
export { getMetric, METRIC_NAMES } from "./metrics.js";
export type { BatchOptions, CompileOptions, FitAsyncOptions, FitOptions, SequentialOptions } from "./model.js";
// ---- Model ----------------------------------------------------------------------------------
export { Sequential } from "./model.js";
export type {
    AdagradOptions,
    AdamOptions,
    AdamWOptions,
    OptimizerOptions,
    RMSpropOptions,
    SGDOptions,
} from "./optimizers.js";
export { Adagrad, Adam, AdamW, BaseOptimizer, getOptimizer, OPTIMIZER_NAMES, RMSprop, SGD } from "./optimizers.js";
export type {
    CosineDecayOptions,
    ExponentialDecayOptions,
    LearningRateSchedule,
    LinearWarmupOptions,
    PiecewiseConstantOptions,
    StepDecayOptions,
} from "./schedules.js";
export {
    constantSchedule,
    cosineDecay,
    exponentialDecay,
    linearWarmup,
    piecewiseConstant,
    stepDecay,
} from "./schedules.js";
export type { DeserializeOptions, SerializeOptions } from "./serialization.js";
// ---- Saving & loading -----------------------------------------------------------------------
export { deserializeModel, serializeModel } from "./serialization.js";
export { VERSION } from "./version.js";
