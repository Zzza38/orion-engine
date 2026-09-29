/**
 * Playground configuration: the single source of truth for everything a user can set.
 * Pure data + validation, no DOM, so it can be unit-tested and round-tripped through the URL.
 */

export type DatasetId = "circle" | "xor" | "spiral" | "blobs" | "moons" | "plane" | "wave";
export type TaskKind = "binary" | "multiclass" | "regression";
export type FeatureId = "x" | "y" | "x2" | "y2" | "xy" | "sinx" | "siny";
export type ActivationId =
    | "tanh"
    | "relu"
    | "leakyRelu"
    | "elu"
    | "selu"
    | "gelu"
    | "swish"
    | "mish"
    | "sigmoid"
    | "softplus"
    | "linear";
export type OptimizerId = "sgd" | "adam" | "adamw" | "rmsprop" | "adagrad";
export type SpeedId = "1" | "3" | "10" | "max";

export interface LayerSpec {
    units: number;
    activation: ActivationId;
    /** Dropout rate applied after this layer's activation; 0 disables it. */
    dropout: number;
}

export interface PlaygroundConfig {
    dataset: DatasetId;
    /** 0–50, percent. */
    noise: number;
    samples: number;
    /** Percentage of samples used for training, 10–90. */
    trainRatio: number;
    /** Class count for the blobs dataset (2–4). */
    classes: number;
    dataSeed: number;
    features: FeatureId[];
    layers: LayerSpec[];
    optimizer: OptimizerId;
    learningRate: number;
    batchSize: number;
    l2: number;
    seed: number;
    speed: SpeedId;
}

export interface DatasetInfo {
    id: DatasetId;
    label: string;
    description: string;
    regression: boolean;
}

export const DATASETS: readonly DatasetInfo[] = [
    { id: "circle", label: "Circle", description: "Ring around a centre cluster", regression: false },
    { id: "xor", label: "XOR", description: "Diagonal quadrants", regression: false },
    { id: "spiral", label: "Spiral", description: "Two interleaved spirals", regression: false },
    { id: "moons", label: "Moons", description: "Two interleaving half circles", regression: false },
    { id: "blobs", label: "Blobs", description: "Gaussian clusters, 2–4 classes", regression: false },
    { id: "plane", label: "Plane", description: "Regression: tilted plane", regression: true },
    { id: "wave", label: "Wave", description: "Regression: sin(x)·cos(y)", regression: true },
];

export interface FeatureInfo {
    id: FeatureId;
    label: string;
    title: string;
}

export const FEATURES: readonly FeatureInfo[] = [
    { id: "x", label: "x", title: "x coordinate" },
    { id: "y", label: "y", title: "y coordinate" },
    { id: "x2", label: "x²", title: "x squared" },
    { id: "y2", label: "y²", title: "y squared" },
    { id: "xy", label: "xy", title: "x times y" },
    { id: "sinx", label: "sin x", title: "sine of x" },
    { id: "siny", label: "sin y", title: "sine of y" },
];

export const ACTIVATIONS: readonly ActivationId[] = [
    "tanh",
    "relu",
    "leakyRelu",
    "elu",
    "selu",
    "gelu",
    "swish",
    "mish",
    "sigmoid",
    "softplus",
    "linear",
];

export const OPTIMIZERS: readonly { id: OptimizerId; label: string }[] = [
    { id: "adam", label: "Adam" },
    { id: "adamw", label: "AdamW" },
    { id: "sgd", label: "SGD" },
    { id: "rmsprop", label: "RMSprop" },
    { id: "adagrad", label: "Adagrad" },
];

export const LEARNING_RATES: readonly number[] = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
export const BATCH_SIZES: readonly number[] = [1, 2, 4, 8, 16, 32, 64, 128];
export const L2_RATES: readonly number[] = [0, 0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1];
export const DROPOUT_RATES: readonly number[] = [0, 0.05, 0.1, 0.2, 0.3, 0.5];
export const SPEEDS: readonly { id: SpeedId; label: string }[] = [
    { id: "1", label: "1 epoch / frame" },
    { id: "3", label: "3 epochs / frame" },
    { id: "10", label: "10 epochs / frame" },
    { id: "max", label: "As fast as possible" },
];

export const LIMITS = {
    layers: { min: 1, max: 6 },
    units: { min: 1, max: 32 },
    noise: { min: 0, max: 50 },
    samples: { min: 50, max: 1000, step: 50 },
    trainRatio: { min: 10, max: 90, step: 5 },
    classes: { min: 2, max: 4 },
    seed: { min: 0, max: 0xffffffff },
} as const;

export function defaultConfig(): PlaygroundConfig {
    return {
        dataset: "circle",
        noise: 5,
        samples: 400,
        trainRatio: 50,
        classes: 3,
        dataSeed: 1,
        features: ["x", "y"],
        layers: [
            { units: 8, activation: "tanh", dropout: 0 },
            { units: 6, activation: "tanh", dropout: 0 },
        ],
        optimizer: "adam",
        learningRate: 0.03,
        batchSize: 16,
        l2: 0,
        seed: 42,
        speed: "3",
    };
}

export function cloneConfig(config: PlaygroundConfig): PlaygroundConfig {
    return {
        ...config,
        features: [...config.features],
        layers: config.layers.map((layer) => ({ ...layer })),
    };
}

export function datasetInfo(id: DatasetId): DatasetInfo {
    return DATASETS.find((d) => d.id === id) ?? DATASETS[0];
}

/** Which learning problem the current dataset poses. */
export function taskKind(config: Pick<PlaygroundConfig, "dataset" | "classes">): TaskKind {
    if (datasetInfo(config.dataset).regression) return "regression";
    if (config.dataset === "blobs" && config.classes > 2) return "multiclass";
    return "binary";
}

/** Number of distinct labels for classification tasks (1 for regression). */
export function classCount(config: Pick<PlaygroundConfig, "dataset" | "classes">): number {
    const task = taskKind(config);
    if (task === "regression") return 1;
    return task === "multiclass" ? config.classes : 2;
}

export interface OutputSpec {
    units: number;
    activation: "sigmoid" | "softmax" | "linear";
    loss: "bce" | "scce" | "mse";
}

/** Output layer and loss implied by the task: sigmoid + bce, softmax + scce, or linear + mse. */
export function outputSpec(config: Pick<PlaygroundConfig, "dataset" | "classes">): OutputSpec {
    switch (taskKind(config)) {
        case "binary":
            return { units: 1, activation: "sigmoid", loss: "bce" };
        case "multiclass":
            return { units: config.classes, activation: "softmax", loss: "scce" };
        default:
            return { units: 1, activation: "linear", loss: "mse" };
    }
}

export function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/** Snaps `value` to the closest entry of `options`, measuring distance on a log scale when `log` is set. */
export function nearest(value: number, options: readonly number[], log = false): number {
    const map = log ? (v: number) => Math.log(Math.max(v, 1e-12)) : (v: number) => v;
    const target = map(value);
    let best = options[0];
    for (const option of options) {
        if (Math.abs(map(option) - target) < Math.abs(map(best) - target)) best = option;
    }
    return best;
}

/** Returns a copy of `config` with every field forced into its legal range. */
export function sanitizeConfig(config: PlaygroundConfig): PlaygroundConfig {
    const defaults = defaultConfig();
    const isDataset = DATASETS.some((d) => d.id === config.dataset);
    const features = FEATURES.map((f) => f.id).filter((id) => config.features.includes(id));
    const layers = config.layers.slice(0, LIMITS.layers.max).map((layer) => ({
        units: Math.round(clamp(finiteOr(layer.units, 4), LIMITS.units.min, LIMITS.units.max)),
        activation: ACTIVATIONS.includes(layer.activation) ? layer.activation : "tanh",
        dropout: nearest(clamp(finiteOr(layer.dropout, 0), 0, 0.5), DROPOUT_RATES),
    }));
    return {
        dataset: isDataset ? config.dataset : defaults.dataset,
        noise: Math.round(clamp(finiteOr(config.noise, defaults.noise), LIMITS.noise.min, LIMITS.noise.max)),
        samples: roundTo(
            clamp(finiteOr(config.samples, defaults.samples), LIMITS.samples.min, LIMITS.samples.max),
            LIMITS.samples.step,
        ),
        trainRatio: roundTo(
            clamp(finiteOr(config.trainRatio, defaults.trainRatio), LIMITS.trainRatio.min, LIMITS.trainRatio.max),
            LIMITS.trainRatio.step,
        ),
        classes: Math.round(clamp(finiteOr(config.classes, defaults.classes), LIMITS.classes.min, LIMITS.classes.max)),
        dataSeed: Math.floor(clamp(finiteOr(config.dataSeed, defaults.dataSeed), LIMITS.seed.min, LIMITS.seed.max)),
        features: features.length > 0 ? features : [...defaults.features],
        layers: layers.length > 0 ? layers : defaults.layers,
        optimizer: OPTIMIZERS.some((o) => o.id === config.optimizer) ? config.optimizer : defaults.optimizer,
        learningRate: nearest(finiteOr(config.learningRate, defaults.learningRate), LEARNING_RATES, true),
        batchSize: nearest(finiteOr(config.batchSize, defaults.batchSize), BATCH_SIZES, true),
        l2: nearestWithZero(finiteOr(config.l2, 0), L2_RATES),
        seed: Math.floor(clamp(finiteOr(config.seed, defaults.seed), LIMITS.seed.min, LIMITS.seed.max)),
        speed: SPEEDS.some((s) => s.id === config.speed) ? config.speed : defaults.speed,
    };
}

function finiteOr(value: number, fallback: number): number {
    return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function roundTo(value: number, step: number): number {
    return Math.round(value / step) * step;
}

function nearestWithZero(value: number, options: readonly number[]): number {
    if (value <= 0) return 0;
    return nearest(
        value,
        options.filter((o) => o > 0),
        true,
    );
}

/** Trainable parameter count (kernels + biases) of the network `config` describes. */
export function parameterCount(config: PlaygroundConfig): number {
    const sizes = [config.features.length, ...config.layers.map((l) => l.units), outputSpec(config).units];
    let total = 0;
    for (let i = 1; i < sizes.length; i++) total += sizes[i - 1] * sizes[i] + sizes[i];
    return total;
}
