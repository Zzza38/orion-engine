/**
 * The {@link Sequential} model: a linear stack of layers with Keras-style compile / fit /
 * predict / evaluate, seeded and reproducible, in Node.js and the browser.
 */

import type { Callback, CallbackContext, Logs } from "./callbacks.js";
import { History, progressLogger } from "./callbacks.js";
import { ShapeError, TrainingError, ValidationError } from "./core/errors.js";
import type { MatrixLike } from "./core/matrix.js";
import { gatherRows, Matrix } from "./core/matrix.js";
import { Random } from "./core/random.js";
import type {
    JsonValue,
    Layer,
    Loss,
    LossIdentifier,
    Metric,
    MetricIdentifier,
    ModelArtifact,
    Optimizer,
    OptimizerIdentifier,
    Parameter,
    TrainingConfig,
    WeightEntry,
} from "./core/types.js";
import { validateArtifact } from "./io/validate.js";
import { ActivationLayer } from "./layers/activation.js";
import { BufferCache, isBaseLayer } from "./layers/base.js";
import { Dense } from "./layers/dense.js";
import { layerFromConfig } from "./layers/index.js";
import { getLoss, LOSS_NAMES } from "./losses.js";
import { getMetric } from "./metrics.js";
import { getOptimizer } from "./optimizers.js";
import {
    booleanOption,
    checkOptions,
    describeValue,
    formatCount,
    isThenable,
    now,
    numberOption,
    positiveInteger,
    snakeCase,
} from "./utils.js";

/** Options for the {@link Sequential} constructor. */
export interface SequentialOptions {
    /**
     * Number of input features. When omitted, it is inferred from the first `fit`, `predict`,
     * `evaluate` or `trainOnBatch` call (or set explicitly with `build(inputSize)`).
     */
    inputSize?: number;
    /** Initial layers (more can be added with `add`). */
    layers?: Layer[];
    /**
     * Seed for every random choice the model makes: weight initialization, shuffling and dropout.
     * Same seed + same data + same options ⇒ bit-identical weights and history.
     * Default: a random seed (readable afterwards as `model.seed`).
     */
    seed?: number;
    /** Model name, shown by `summary()` and stored in saved models. Default "sequential". */
    name?: string;
}

/** Options for {@link Sequential.compile}. */
export interface CompileOptions {
    /** Loss to minimize: a name ("mse", "bce", "cce", "scce", …), a config, or a Loss. */
    loss: LossIdentifier;
    /** Optimizer: a name ("adam"), a config (`{ name: "adam", learningRate: 0.01 }`) or an instance. Default "adam". */
    optimizer?: OptimizerIdentifier;
    /** Metrics reported in the logs under their canonical names, e.g. "accuracy", "meanAbsoluteError". */
    metrics?: MetricIdentifier[];
}

/** Options for {@link Sequential.fit}. */
export interface FitOptions {
    /** Passes over the training data. Default 1. */
    epochs?: number;
    /** Samples per gradient step. Default 32. The last batch of an epoch may be smaller. */
    batchSize?: number;
    /** Reshuffle the training samples (with the model's seeded Random) every epoch. Default true. */
    shuffle?: boolean;
    /**
     * Fraction in (0, 1) of the samples to hold out for validation. Like Keras, the *last*
     * samples (before shuffling) are used, so shuffle your data first if it is ordered.
     */
    validationSplit?: number;
    /** Explicit validation set `[x, y]`; takes precedence over `validationSplit`. */
    validationData?: readonly [MatrixLike, MatrixLike];
    /** Callbacks, called in order. */
    callbacks?: Callback[];
    /** Shorthand for a callback with only `onEpochEnd`. */
    onEpochEnd?: (epoch: number, logs: Logs) => void;
    /** Log progress: `true` every epoch, a number N every N epochs. Default false. */
    verbose?: boolean | number;
}

/** Options for {@link Sequential.fitAsync}. */
export interface FitAsyncOptions extends Omit<FitOptions, "onEpochEnd"> {
    /** Shorthand for a callback with only `onEpochEnd`; may return a Promise, which is awaited. */
    onEpochEnd?: (epoch: number, logs: Logs) => void | Promise<void>;
    /** Aborts training: the returned promise rejects with `signal.reason` at the next batch. */
    signal?: AbortSignal;
    /**
     * Yield to the event loop whenever this many milliseconds of work have passed, so pages stay
     * responsive and timers/abort events run. Default 16 (about one animation frame).
     */
    yieldEvery?: number;
}

/** Options for {@link Sequential.predict} and {@link Sequential.evaluate}. */
export interface BatchOptions {
    /** Samples per forward pass. Default 256 for predict, 32 for evaluate. */
    batchSize?: number;
}

const FIT_KEYS = [
    "epochs",
    "batchSize",
    "shuffle",
    "validationSplit",
    "validationData",
    "callbacks",
    "onEpochEnd",
    "verbose",
];
const FIT_ASYNC_KEYS = [...FIT_KEYS, "signal", "yieldEvery"];
const HOOKS = ["onTrainBegin", "onEpochBegin", "onBatchEnd", "onEpochEnd", "onTrainEnd"] as const;
type Hook = (typeof HOOKS)[number];
/** Yielded by the training loop in async mode after every batch: a chance to yield to the event loop. */
const TICK = Symbol("tick");

interface FitPlan {
    readonly x: Matrix;
    readonly y: Matrix;
    readonly valX: Matrix | null;
    readonly valY: Matrix | null;
    readonly epochs: number;
    readonly batchSize: number;
    readonly shuffle: boolean;
    readonly callbacks: Callback[];
    readonly async: boolean;
    readonly signal: AbortSignal | undefined;
}

interface Regularized {
    regularizationLoss(): number;
}

/**
 * A linear stack of layers.
 *
 * @example
 * import { Sequential, dense } from "@zzza38/orion-engine";
 *
 * const model = new Sequential({ inputSize: 2, seed: 42, layers: [dense(8, "tanh"), dense(1, "sigmoid")] });
 * model.compile({ loss: "bce", optimizer: { name: "adam", learningRate: 0.05 }, metrics: ["accuracy"] });
 * const history = model.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0], { epochs: 300, batchSize: 4 });
 * history.last("accuracy"); // 1
 * model.predict([1, 0]);    // [0.97…]
 */
export class Sequential {
    /** Model name. */
    readonly name: string;
    /** Seed of the model's random generator. */
    readonly seed: number;

    private readonly layerList: Layer[] = [];
    private builtInputSize: number | undefined;
    private readonly rng: Random;
    private lossFn: Loss | null = null;
    private optimizerInstance: Optimizer | null = null;
    private metricList: Metric[] = [];
    /** Flattened parameters, rebuilt lazily after layers change. */
    private paramCache: Parameter[] | null = null;
    private regularizedLayers: Regularized[] = [];
    /** True while a fit/fitAsync call is running. */
    private isTraining = false;

    /**
     * @example
     * const model = new Sequential({ inputSize: 4, seed: 1, layers: [dense(16, "relu"), dense(3, "softmax")] });
     */
    constructor(options: SequentialOptions = {}) {
        const where = "Sequential";
        checkOptions(where, options, ["inputSize", "layers", "seed", "name"]);
        if (options.seed !== undefined && (typeof options.seed !== "number" || !Number.isInteger(options.seed))) {
            throw new ValidationError(`${where}: "seed" must be an integer, got ${describeValue(options.seed)}`);
        }
        this.rng = new Random(options.seed);
        this.seed = this.rng.seed;
        if (options.name !== undefined && (typeof options.name !== "string" || options.name.length === 0)) {
            throw new ValidationError(
                `${where}: "name" must be a non-empty string, got ${describeValue(options.name)}`,
            );
        }
        this.name = options.name ?? "sequential";
        if (options.layers !== undefined) {
            if (!Array.isArray(options.layers)) {
                throw new ValidationError(
                    `${where}: "layers" must be an array of layers, got ${describeValue(options.layers)}`,
                );
            }
            for (const layer of options.layers) this.add(layer);
        }
        if (options.inputSize !== undefined) this.build(positiveInteger(where, "inputSize", options.inputSize));
    }

    // -----------------------------------------------------------------------------------------
    // Structure
    // -----------------------------------------------------------------------------------------

    /** The layers, in order. */
    get layers(): readonly Layer[] {
        return this.layerList;
    }

    /** Number of input features, or undefined until known. */
    get inputSize(): number | undefined {
        return this.builtInputSize;
    }

    /** Number of outputs of the last layer, or undefined until the model is built. */
    get outputSize(): number | undefined {
        if (this.builtInputSize === undefined || this.layerList.length === 0) return undefined;
        return this.layerList[this.layerList.length - 1].outputSize;
    }

    /** True once the input size is known and every layer has its weights. */
    get built(): boolean {
        return this.builtInputSize !== undefined;
    }

    /** True after {@link Sequential.compile}. */
    get compiled(): boolean {
        return this.lossFn !== null;
    }

    /** The compiled loss, if any. */
    get loss(): Loss | undefined {
        return this.lossFn ?? undefined;
    }

    /** The compiled optimizer, if any. Its `learningRate` may be changed at any time. */
    get optimizer(): Optimizer | undefined {
        return this.optimizerInstance ?? undefined;
    }

    /** The compiled metrics. */
    get metrics(): readonly Metric[] {
        return this.metricList;
    }

    /** Total number of scalar weights, trainable and non-trainable. */
    get parameterCount(): number {
        let count = 0;
        for (const p of this.parameterList()) count += p.value.size;
        return count;
    }

    /**
     * Appends a layer. Unnamed layers are named `<type>_<n>` (`dense_1`, `dropout_1`, …); names must
     * be unique within the model. If the input size is known, the layer is built immediately.
     * @returns this, for chaining.
     * @example
     * model.add(dense(16, "relu")).add(dropout(0.2)).add(dense(1));
     */
    add(layer: Layer): this {
        if (!isLayer(layer)) {
            throw new ValidationError(
                `add() expects a layer, e.g. model.add(dense(8, "relu")), got ${describeValue(layer)}`,
            );
        }
        if (this.layerList.includes(layer)) {
            throw new ValidationError(`Layer "${layer.name}" is already in this model; create a new layer instead`);
        }
        if (layer.name === "") {
            layer.name = this.uniqueName(layer.type);
        } else if (this.layerList.some((l) => l.name === layer.name)) {
            throw new ValidationError(
                `Duplicate layer name "${layer.name}": layer names must be unique within a model ` +
                    "(leave the name out to get an automatic one)",
            );
        }
        if (this.builtInputSize !== undefined) layer.build(this.currentOutputSize(), this.rng.fork());
        this.layerList.push(layer);
        this.paramCache = null;
        return this;
    }

    /**
     * Fixes the input size and creates every layer's weights (each layer draws from its own fork
     * of the model's seeded Random). Called automatically when `inputSize` is known.
     * @returns this, for chaining.
     */
    build(inputSize: number): this {
        positiveInteger("build", "inputSize", inputSize);
        if (this.builtInputSize !== undefined) {
            if (this.builtInputSize === inputSize) return this;
            throw new ShapeError(
                `Model "${this.name}" is already built for inputSize ${this.builtInputSize}; got ${inputSize}`,
            );
        }
        let size = inputSize;
        for (const layer of this.layerList) {
            layer.build(size, this.rng.fork());
            size = layer.outputSize;
        }
        this.builtInputSize = inputSize;
        this.paramCache = null;
        return this;
    }

    /** Every parameter of every layer (trainable and not), in layer order. */
    parameters(): Parameter[] {
        return this.parameterList().slice();
    }

    // -----------------------------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------------------------

    /**
     * Sets the loss, optimizer and metrics used by `fit`, `trainOnBatch` and `evaluate`.
     * Compiling again replaces them (and resets optimizer state).
     * @returns this, for chaining.
     * @example
     * model.compile({ loss: "scce", optimizer: { name: "adam", learningRate: 0.01 }, metrics: ["accuracy"] });
     */
    compile(options: CompileOptions): this {
        checkOptions("compile", options, ["loss", "optimizer", "metrics"]);
        if (options === undefined || options.loss === undefined) {
            throw new ValidationError(
                `compile: "loss" is required, e.g. model.compile({ loss: "mse", optimizer: "adam" })`,
            );
        }
        const loss = getLoss(options.loss);
        const optimizer = getOptimizer(options.optimizer ?? "adam");
        const metricIds = options.metrics ?? [];
        if (!Array.isArray(metricIds)) {
            throw new ValidationError(
                `compile: "metrics" must be an array, e.g. ["accuracy"], got ${describeValue(metricIds)}`,
            );
        }
        const metrics = metricIds.map((id) => getMetric(id));
        const seen = new Set<string>();
        for (const metric of metrics) {
            if (seen.has(metric.name)) throw new ValidationError(`compile: metric "${metric.name}" is listed twice`);
            seen.add(metric.name);
        }
        this.lossFn = loss;
        this.optimizerInstance = optimizer;
        this.metricList = metrics;
        return this;
    }

    /**
     * Trains the model for a fixed number of epochs.
     *
     * Each step runs a forward pass in training mode over a mini-batch, computes the loss (plus any
     * weight regularization), backpropagates, and applies one optimizer update. Epoch logs contain
     * `loss`, each metric, `valLoss`/`val<Metric>` when validating, `learningRate` and `durationMs`;
     * training metrics are batch-size-weighted means over the epoch.
     *
     * @param x Inputs: `number[][]`, a `Matrix`, or a single sample `number[]`.
     * @param y Targets with one row per sample. A flat `number[]` is one value per sample
     *   (binary targets, or class indices for "scce").
     * @returns The per-epoch {@link History}.
     * @throws TrainingError if the loss becomes NaN or infinite.
     * @throws ValidationError if a callback returns a Promise (use {@link Sequential.fitAsync}).
     * @example
     * const history = model.fit(x, y, { epochs: 100, batchSize: 16, validationSplit: 0.2, verbose: 10 });
     */
    fit(x: MatrixLike, y: MatrixLike, options: FitOptions = {}): History {
        const plan = this.prepareFit("fit", x, y, options, FIT_KEYS, false, undefined);
        const result = this.trainLoop(plan).next();
        if (!result.done) throw new Error("Internal error: synchronous training loop yielded");
        return result.value;
    }

    /**
     * Like {@link Sequential.fit}, but periodically yields to the event loop (so browsers stay
     * responsive), awaits callbacks that return Promises, and can be cancelled with an AbortSignal.
     * @example
     * const controller = new AbortController();
     * stopButton.onclick = () => controller.abort();
     * await model.fitAsync(x, y, { epochs: 1000, signal: controller.signal, onEpochEnd: (e, logs) => draw(logs) });
     */
    async fitAsync(x: MatrixLike, y: MatrixLike, options: FitAsyncOptions = {}): Promise<History> {
        checkOptions("fitAsync", options, FIT_ASYNC_KEYS);
        const signal = options.signal;
        if (
            signal !== undefined &&
            (signal === null || typeof signal !== "object" || typeof signal.aborted !== "boolean")
        ) {
            throw new ValidationError(`fitAsync: "signal" must be an AbortSignal, got ${describeValue(signal)}`);
        }
        const yieldEvery = numberOption(
            "fitAsync",
            "yieldEvery",
            options.yieldEvery,
            16,
            "a finite number >= 0",
            (v) => v >= 0,
        );
        signal?.throwIfAborted();
        const plan = this.prepareFit("fitAsync", x, y, options as FitOptions, FIT_ASYNC_KEYS, true, signal);
        const loop = this.trainLoop(plan);
        let lastYield = now();
        let step = loop.next();
        while (!step.done) {
            const value = step.value;
            if (value === TICK) {
                if (now() - lastYield >= yieldEvery) {
                    await yieldToEventLoop();
                    lastYield = now();
                }
                step = loop.next();
                continue;
            }
            try {
                await value;
            } catch (error) {
                step = loop.throw(error);
                continue;
            }
            step = loop.next();
        }
        return step.value;
    }

    /**
     * Runs a single gradient step on one batch and returns its logs (`loss` and metrics, computed
     * before the update).
     * @example
     * const { loss } = model.trainOnBatch(xBatch, yBatch);
     */
    trainOnBatch(x: MatrixLike, y: MatrixLike): Logs {
        this.requireCompiled("trainOnBatch");
        const input = this.toInputs(x, "trainOnBatch", "x");
        const target = this.toTargets(y, input.rows, "trainOnBatch", "y");
        checkFinite(input, "trainOnBatch", "x");
        checkFinite(target, "trainOnBatch", "y");
        this.checkTargets(target, "trainOnBatch", "y");
        const { loss, prediction } = this.trainStep(input, target, null, "on this batch");
        const logs: Logs = { loss };
        for (const metric of this.metricList) logs[metric.name] = metric.compute(prediction, target);
        return logs;
    }

    /**
     * Computes the loss (including regularization) and metrics over a dataset in inference mode.
     * @example
     * const { loss, accuracy } = model.evaluate(xTest, yTest);
     */
    evaluate(x: MatrixLike, y: MatrixLike, options: BatchOptions = {}): Logs {
        checkOptions("evaluate", options, ["batchSize"]);
        this.requireCompiled("evaluate");
        const batchSize = positiveInteger("evaluate", "batchSize", options.batchSize, 32);
        const input = this.toInputs(x, "evaluate", "x");
        const target = this.toTargets(y, input.rows, "evaluate", "y");
        checkFinite(input, "evaluate", "x");
        checkFinite(target, "evaluate", "y");
        this.checkTargets(target, "evaluate", "y");
        if (input.rows === 0) throw new ValidationError("evaluate: x has no samples");
        return this.evaluateMatrices(input, target, batchSize);
    }

    // -----------------------------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------------------------

    /**
     * Runs the model in inference mode (dropout off, batch norm using its moving statistics).
     * The output mirrors the input: one sample `number[]` → `number[]`; `number[][]` → `number[][]`;
     * `Matrix` → a new `Matrix`.
     * @example
     * model.predict([0, 1]);           // [0.98]
     * model.predict([[0, 1], [1, 1]]); // [[0.98], [0.03]]
     */
    predict(input: Matrix, options?: BatchOptions): Matrix;
    predict(input: readonly (readonly number[])[], options?: BatchOptions): number[][];
    predict(input: readonly number[], options?: BatchOptions): number[];
    predict(input: MatrixLike, options?: BatchOptions): Matrix | number[][] | number[];
    predict(input: MatrixLike, options?: BatchOptions): Matrix | number[][] | number[] {
        let batchSize = 256;
        if (options !== undefined) {
            checkOptions("predict", options, ["batchSize"]);
            batchSize = positiveInteger("predict", "batchSize", options.batchSize, 256);
        }
        if (Array.isArray(input) && input.length === 0) return [];
        const x = this.toInputs(input, "predict", "input");
        if (x.rows <= batchSize) {
            // Common case: one forward pass, converted straight from the last layer's buffer.
            let output = x;
            for (const layer of this.layerList) output = layer.forward(output, false);
            if (input instanceof Matrix) return output.clone();
            return Array.isArray(input[0]) ? output.toArray() : Array.from(output.data);
        }
        const output = this.forwardAll(x, batchSize);
        if (input instanceof Matrix) return output;
        return Array.isArray(input[0]) ? output.toArray() : Array.from(output.data);
    }

    // -----------------------------------------------------------------------------------------
    // Weights & serialization
    // -----------------------------------------------------------------------------------------

    /** Copies of every parameter (in `parameters()` order), including non-trainable ones. */
    getWeights(): WeightEntry[] {
        return this.parameterList().map((p) => ({
            name: p.name,
            shape: [p.value.rows, p.value.cols],
            data: p.value.data.slice(),
        }));
    }

    /**
     * Replaces every parameter's values. `entries` must contain each of the model's weights exactly
     * once (matched by name, in any order) with the right shape; nothing is changed if any entry is
     * invalid.
     */
    setWeights(entries: readonly WeightEntry[]): void {
        const where = "setWeights";
        if (this.builtInputSize === undefined) {
            throw new ValidationError(
                `${where}: the model is not built yet; pass inputSize to the constructor or call build(inputSize) first`,
            );
        }
        if (!Array.isArray(entries))
            throw new ValidationError(`${where}: expected an array of weight entries, got ${describeValue(entries)}`);
        const params = this.parameterList();
        const byName = new Map(params.map((p) => [p.name, p]));
        const updates: [Parameter, WeightEntry][] = [];
        const seen = new Set<string>();
        for (const entry of entries) {
            if (entry === null || typeof entry !== "object") {
                throw new ValidationError(
                    `${where}: expected weight entries { name, shape, data }, got ${describeValue(entry)}`,
                );
            }
            const param = byName.get(entry.name);
            if (param === undefined) {
                throw new ValidationError(
                    `${where}: unknown weight ${describeValue(entry.name)}. This model's weights are: ${[...byName.keys()].join(", ")}`,
                );
            }
            if (seen.has(entry.name)) throw new ValidationError(`${where}: weight "${entry.name}" is given twice`);
            seen.add(entry.name);
            const { rows, cols } = param.value;
            const shape = entry.shape;
            if (!Array.isArray(shape) || shape[0] !== rows || shape[1] !== cols) {
                throw new ShapeError(
                    `${where}: weight "${entry.name}" has shape ${describeValue(shape)}, but the model expects [${rows}, ${cols}]`,
                );
            }
            const data = entry.data;
            if (data === null || typeof data !== "object" || data.length !== rows * cols) {
                throw new ShapeError(
                    `${where}: weight "${entry.name}" needs ${rows * cols} values, got ${describeValue(data?.length)}`,
                );
            }
            for (let i = 0; i < data.length; i++) {
                if (typeof data[i] !== "number" || !Number.isFinite(data[i])) {
                    throw new ValidationError(
                        `${where}: weight "${entry.name}"[${i}] is ${describeValue(data[i])}; weights must be finite numbers`,
                    );
                }
            }
            updates.push([param, entry]);
        }
        const missing = params.filter((p) => !seen.has(p.name)).map((p) => p.name);
        if (missing.length > 0)
            throw new ValidationError(`${where}: missing weight(s) ${missing.map((n) => `"${n}"`).join(", ")}`);
        for (const [param, entry] of updates) param.value.data.set(entry.data);
    }

    /**
     * Describes the model (architecture, weights, and training config when compiled) as a
     * {@link ModelArtifact}, the in-memory form of a saved model. The model name is stored as
     * `metadata.name`. Optimizer state (e.g. Adam moments) is not included.
     */
    toArtifact(metadata?: { [key: string]: JsonValue | undefined }): ModelArtifact {
        const inputSize = this.requireBuilt("toArtifact");
        if (metadata !== undefined && (metadata === null || typeof metadata !== "object" || Array.isArray(metadata))) {
            throw new ValidationError(`toArtifact: metadata must be a plain object, got ${describeValue(metadata)}`);
        }
        const artifact: ModelArtifact = {
            format: "orion-engine",
            formatVersion: 1,
            inputSize,
            layers: this.layerList.map((layer) => layer.getConfig()),
            weights: this.getWeights(),
            metadata: { name: this.name, ...metadata },
        };
        const training = this.trainingConfig();
        if (training !== undefined) artifact.training = training;
        return artifact;
    }

    /**
     * Rebuilds a model from an artifact (see {@link Sequential.toArtifact}). If the artifact has a
     * training config, the model is compiled with it; the optimizer starts with fresh state.
     * @param options.seed Seed for the rebuilt model's shuffling and dropout. Default random.
     */
    static fromArtifact(artifact: ModelArtifact, options: { seed?: number } = {}): Sequential {
        checkOptions("fromArtifact", options, ["seed"]);
        const valid = validateArtifact(artifact);
        const name = valid.metadata?.name;
        const model = new Sequential({
            inputSize: valid.inputSize,
            layers: valid.layers.map((config) => layerFromConfig(config)),
            seed: options.seed,
            name: typeof name === "string" && name.length > 0 ? name : undefined,
        });
        model.setWeights(valid.weights);
        if (valid.training !== undefined) {
            model.compile({
                loss: valid.training.loss,
                optimizer: valid.training.optimizer,
                metrics: valid.training.metrics,
            });
        }
        return model;
    }

    /**
     * A deep copy: same architecture, name, seed and weights, compiled the same way (with a fresh
     * optimizer). Training either model does not affect the other.
     */
    clone(): Sequential {
        const copy = new Sequential({
            name: this.name,
            seed: this.seed,
            layers: this.layerList.map((layer) => layerFromConfig(layer.getConfig())),
        });
        if (this.builtInputSize !== undefined) {
            copy.build(this.builtInputSize);
            copy.setWeights(this.getWeights());
        }
        if (this.lossFn !== null && this.optimizerInstance !== null) {
            copy.compile({
                loss: this.lossFn,
                optimizer: this.optimizerInstance.getConfig(),
                metrics: this.metricList.slice(),
            });
        }
        return copy;
    }

    /**
     * A table of the layers with their output sizes, activations and parameter counts.
     * @example
     * console.log(model.summary());
     */
    summary(): string {
        const header = ["Layer (type)", "Output size", "Activation", "Params"];
        const rows: string[][] = [];
        const built = this.builtInputSize !== undefined;
        let trainable = 0;
        let total = 0;
        for (const layer of this.layerList) {
            const type = layer.type.charAt(0).toUpperCase() + layer.type.slice(1);
            let count = 0;
            for (const p of layer.parameters()) {
                count += p.value.size;
                if (p.trainable) trainable += p.value.size;
            }
            total += count;
            const outputSize = built ? String(layer.outputSize) : layer instanceof Dense ? String(layer.units) : "?";
            rows.push([`${layer.name} (${type})`, outputSize, activationName(layer), built ? formatCount(count) : "?"]);
        }
        const widths = header.map((h, c) => Math.max(h.length, ...rows.map((r) => r[c].length)));
        const line = (left: string, mid: string, right: string) =>
            left + widths.map((w) => "─".repeat(w + 2)).join(mid) + right;
        const format = (cells: string[]) =>
            `│${cells.map((cell, c) => ` ${c === 3 ? cell.padStart(widths[c]) : cell.padEnd(widths[c])} `).join("│")}│`;
        const lines = [`Model: "${this.name}"`, line("┌", "┬", "┐"), format(header), line("├", "┼", "┤")];
        for (const row of rows) lines.push(format(row));
        if (rows.length === 0) lines.push(format(["(no layers)", "", "", ""]));
        lines.push(line("└", "┴", "┘"));
        lines.push(`Input size: ${built ? this.builtInputSize : "? (not built yet)"}`);
        if (built) {
            lines.push(`Total params: ${formatCount(total)}`);
            lines.push(`Trainable params: ${formatCount(trainable)}`);
            lines.push(`Non-trainable params: ${formatCount(total - trainable)}`);
        } else {
            lines.push("Total params: ? (call build(inputSize) or pass inputSize to count them)");
        }
        return lines.join("\n");
    }

    // -----------------------------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------------------------

    private parameterList(): Parameter[] {
        if (this.paramCache === null) {
            const params: Parameter[] = [];
            const regularized: Regularized[] = [];
            for (const layer of this.layerList) {
                if (layer.built) params.push(...layer.parameters());
                const candidate = layer as Partial<Regularized>;
                if (typeof candidate.regularizationLoss === "function")
                    regularized.push(layer as unknown as Regularized);
            }
            this.paramCache = params;
            this.regularizedLayers = regularized;
        }
        return this.paramCache;
    }

    private regularizationLoss(): number {
        this.parameterList();
        let total = 0;
        for (const layer of this.regularizedLayers) total += layer.regularizationLoss();
        return total;
    }

    private trainingConfig(): TrainingConfig | undefined {
        if (this.lossFn === null || this.optimizerInstance === null) return undefined;
        return {
            loss: this.lossFn.getConfig(),
            optimizer: this.optimizerInstance.getConfig(),
            metrics: this.metricList.map((m) => m.name),
        };
    }

    private uniqueName(type: string): string {
        const prefix = snakeCase(type || "layer");
        const taken = new Set(this.layerList.map((l) => l.name));
        let n = 1;
        while (taken.has(`${prefix}_${n}`)) n++;
        return `${prefix}_${n}`;
    }

    private currentOutputSize(): number {
        const last = this.layerList[this.layerList.length - 1];
        return last === undefined ? (this.builtInputSize as number) : last.outputSize;
    }

    private requireBuilt(where: string): number {
        if (this.builtInputSize === undefined) {
            throw new ValidationError(
                `${where}: the model is not built yet; pass inputSize to the constructor, call build(inputSize), or fit/predict first`,
            );
        }
        return this.builtInputSize;
    }

    private requireCompiled(where: string): void {
        if (this.lossFn === null) {
            throw new ValidationError(
                `${where}: the model is not compiled; call model.compile({ loss: "mse", optimizer: "adam" }) first`,
            );
        }
    }

    /** Converts inputs to a Matrix, builds the model on first use, and checks the feature count. */
    private toInputs(x: MatrixLike, where: string, label: string): Matrix {
        if (this.layerList.length === 0) {
            throw new ValidationError(
                `${where}: the model has no layers; add some with model.add(dense(units, activation))`,
            );
        }
        const matrix = toMatrix(x, where, label);
        if (this.builtInputSize === undefined) {
            if (matrix.cols === 0) throw new ShapeError(`${where}: ${label} has no features`);
            this.build(matrix.cols);
        }
        if (matrix.cols !== this.builtInputSize) {
            const expected = this.builtInputSize;
            const hint =
                expected === 1 && Array.isArray(x) && !Array.isArray(x[0])
                    ? ". A flat array is one sample; pass one row per sample instead, e.g. [[0.1], [0.2], [0.3]]"
                    : "";
            throw new ShapeError(
                `Expected input with ${expected} feature${expected === 1 ? "" : "s"} (inputSize), got ${matrix.cols}${hint}`,
            );
        }
        return matrix;
    }

    /** Converts targets to a Matrix with `rows` rows. A flat array is one value per sample. */
    private toTargets(y: MatrixLike, rows: number, where: string, label: string): Matrix {
        let matrix: Matrix;
        if (Array.isArray(y) && y.length > 0 && !Array.isArray(y[0]) && !(rows === 1 && y.length !== 1)) {
            matrix = toMatrix(y, where, label);
            matrix = new Matrix(matrix.cols, 1, matrix.data);
        } else {
            matrix = toMatrix(y, where, label);
        }
        if (matrix.rows !== rows) {
            throw new ShapeError(
                `${where}: x and ${label} must have the same number of samples, got ${rows} and ${matrix.rows}`,
            );
        }
        return matrix;
    }

    /** Checks target columns against the loss and the model's output size. */
    private checkTargets(y: Matrix, where: string, label: string): void {
        const loss = this.lossFn as Loss;
        if (!(LOSS_NAMES as readonly string[]).includes(loss.name)) return; // custom loss: its own rules
        const outputs = this.outputSize as number;
        if (loss.name === "sparseCategoricalCrossentropy") {
            if (y.cols !== 1) {
                throw new ShapeError(
                    `${where}: sparseCategoricalCrossentropy expects ${label} to hold one integer class index per sample, ` +
                        `got ${y.cols} columns. Pass labels like [0, 2, 1], or use loss "cce" with one-hot targets`,
                );
            }
            const data = y.data;
            for (let i = 0; i < data.length; i++) {
                const v = data[i];
                if (!Number.isInteger(v) || v < 0 || v >= outputs) {
                    throw new ValidationError(
                        `${where}: ${label}[${i}] = ${v} is not a class index in [0, ${outputs}) ` +
                            `(the last layer has ${outputs} units)`,
                    );
                }
            }
            return;
        }
        if (y.cols !== outputs) {
            let hint = "";
            if (y.cols === 1 && outputs > 1) {
                hint = `. For integer class labels use loss "scce", or one-hot encode them with oneHot(labels, ${outputs})`;
            }
            throw new ShapeError(
                `${where}: expected ${label} with ${outputs} ${outputs === 1 ? "column" : "columns"} ` +
                    `(outputSize of the last layer), got ${y.cols}${hint}`,
            );
        }
    }

    private prepareFit(
        where: string,
        x: MatrixLike,
        y: MatrixLike,
        options: FitOptions,
        allowedKeys: readonly string[],
        async: boolean,
        signal: AbortSignal | undefined,
    ): FitPlan {
        checkOptions(where, options, allowedKeys);
        this.requireCompiled(where);
        if (this.isTraining) {
            throw new ValidationError(
                `${where}: this model is already training (another fit/fitAsync call is in progress); ` +
                    "await it first, or end it early with ctx.stopTraining() from a callback",
            );
        }
        const epochs = positiveInteger(where, "epochs", options.epochs, 1);
        const batchSize = positiveInteger(where, "batchSize", options.batchSize, 32);
        const shuffle = booleanOption(where, "shuffle", options.shuffle, true);

        let inputs = this.toInputs(x, where, "x");
        let targets = this.toTargets(y, inputs.rows, where, "y");
        checkFinite(inputs, where, "x");
        checkFinite(targets, where, "y");
        this.checkTargets(targets, where, "y");

        let valX: Matrix | null = null;
        let valY: Matrix | null = null;
        if (options.validationData !== undefined) {
            const data = options.validationData;
            if (!Array.isArray(data) || data.length !== 2) {
                throw new ValidationError(
                    `${where}: "validationData" must be a pair [x, y], got ${describeValue(data)}`,
                );
            }
            valX = this.toInputs(data[0], where, "validationData x");
            valY = this.toTargets(data[1], valX.rows, where, "validationData y");
            checkFinite(valX, where, "validationData x");
            checkFinite(valY, where, "validationData y");
            this.checkTargets(valY, where, "validationData y");
            if (valX.rows === 0) valX = valY = null;
        } else if (options.validationSplit !== undefined) {
            const split = numberOption(
                where,
                "validationSplit",
                options.validationSplit,
                0,
                "a number in (0, 1)",
                (v) => v > 0 && v < 1,
            );
            const valCount = Math.floor(inputs.rows * split);
            const trainCount = inputs.rows - valCount;
            if (valCount === 0 || trainCount === 0) {
                throw new ValidationError(
                    `${where}: validationSplit ${split} of ${inputs.rows} samples leaves ${valCount} for validation and ` +
                        `${trainCount} for training; both must be at least 1`,
                );
            }
            valX = sliceRowsView(inputs, trainCount, inputs.rows);
            valY = sliceRowsView(targets, trainCount, targets.rows);
            inputs = sliceRowsView(inputs, 0, trainCount);
            targets = sliceRowsView(targets, 0, trainCount);
        }
        if (inputs.rows === 0) throw new ValidationError(`${where}: x has no samples`);

        const callbacks: Callback[] = [];
        if (options.callbacks !== undefined) {
            if (!Array.isArray(options.callbacks)) {
                throw new ValidationError(
                    `${where}: "callbacks" must be an array, got ${describeValue(options.callbacks)}`,
                );
            }
            for (const cb of options.callbacks) {
                if (cb === null || typeof cb !== "object") {
                    throw new ValidationError(
                        `${where}: every callback must be an object with hooks such as onEpochEnd, got ${describeValue(cb)}`,
                    );
                }
                callbacks.push(cb);
            }
        }
        if (options.onEpochEnd !== undefined) {
            const fn = options.onEpochEnd;
            if (typeof fn !== "function")
                throw new ValidationError(`${where}: "onEpochEnd" must be a function, got ${describeValue(fn)}`);
            callbacks.push({ onEpochEnd: (epoch, logs) => fn(epoch, logs) });
        }
        const verbose = options.verbose;
        if (verbose !== undefined && verbose !== false) {
            if (verbose === true) callbacks.push(progressLogger());
            else callbacks.push(progressLogger({ every: positiveInteger(where, "verbose", verbose) }));
        }
        return { x: inputs, y: targets, valX, valY, epochs, batchSize, shuffle, callbacks, async, signal };
    }

    /**
     * The training loop, shared by fit (sync) and fitAsync. In async mode it yields callback
     * results (to be awaited) and TICK after every batch; in sync mode it never yields.
     */
    private *trainLoop(plan: FitPlan): Generator<unknown, History, unknown> {
        this.isTraining = true;
        try {
            return yield* this.runEpochs(plan);
        } finally {
            this.isTraining = false;
        }
    }

    private *runEpochs(plan: FitPlan): Generator<unknown, History, unknown> {
        const { x, y, valX, valY, epochs, batchSize, callbacks, signal } = plan;
        const optimizer = this.optimizerInstance as Optimizer;
        const metrics = this.metricList;
        const samples = x.rows;
        const steps = Math.ceil(samples / batchSize);
        const history = new History();
        let stop = false;
        const ctx: CallbackContext = {
            model: this,
            optimizer,
            epochs,
            batchSize,
            samples,
            stepsPerEpoch: steps,
            hasValidation: valX !== null,
            history,
            get stopRequested() {
                return stop;
            },
            stopTraining() {
                stop = true;
            },
        };
        const withHook = (hook: Hook) => callbacks.filter((cb) => typeof cb[hook] === "function");
        const onEpochBegin = withHook("onEpochBegin");
        const onBatchEnd = withHook("onBatchEnd");
        const onEpochEnd = withHook("onEpochEnd");

        const outputSize = this.outputSize as number;
        const buffers = new BufferCache((rows) => ({
            x: new Matrix(rows, x.cols),
            y: new Matrix(rows, y.cols),
            grad: new Matrix(rows, outputSize),
        }));
        const indices = new Int32Array(samples);
        for (let i = 0; i < samples; i++) indices[i] = i;
        const metricSums = new Float64Array(metrics.length);

        yield* invoke(plan, "onTrainBegin", withHook("onTrainBegin"), [ctx]);
        let logs: Logs = {};
        for (let epoch = 0; epoch < epochs; epoch++) {
            signal?.throwIfAborted();
            const start = now();
            yield* invoke(plan, "onEpochBegin", onEpochBegin, [epoch, ctx]);
            if (plan.shuffle) this.rng.shuffle(indices);
            let lossSum = 0;
            metricSums.fill(0);
            for (let batch = 0; batch < steps; batch++) {
                if (signal?.aborted) signal.throwIfAborted();
                const from = batch * batchSize;
                const to = Math.min(from + batchSize, samples);
                const size = to - from;
                const b = buffers.get(size);
                const batchIndices = indices.subarray(from, to);
                gatherRows(x, batchIndices, b.x);
                gatherRows(y, batchIndices, b.y);
                const { loss, prediction } = this.trainStep(
                    b.x,
                    b.y,
                    b.grad,
                    `at epoch ${epoch + 1}/${epochs}, batch ${batch + 1}/${steps}`,
                );
                lossSum += loss * size;
                let batchLogs: Logs | null = onBatchEnd.length > 0 ? { loss, size } : null;
                for (let m = 0; m < metrics.length; m++) {
                    const metric = metrics[m];
                    const value = metric.compute(prediction, b.y);
                    metricSums[m] += (metric.name === "rootMeanSquaredError" ? value * value : value) * size;
                    if (batchLogs !== null) batchLogs[metric.name] = value;
                }
                if (batchLogs !== null) {
                    yield* invoke(plan, "onBatchEnd", onBatchEnd, [batch, batchLogs, ctx]);
                    batchLogs = null;
                }
                if (plan.async) yield TICK;
            }

            logs = { loss: lossSum / samples };
            for (let m = 0; m < metrics.length; m++) {
                const mean = metricSums[m] / samples;
                logs[metrics[m].name] = metrics[m].name === "rootMeanSquaredError" ? Math.sqrt(mean) : mean;
            }
            if (valX !== null && valY !== null) {
                const valLogs = this.evaluateMatrices(valX, valY, batchSize);
                for (const key of Object.keys(valLogs))
                    logs[`val${key.charAt(0).toUpperCase()}${key.slice(1)}`] = valLogs[key];
            }
            logs.learningRate = optimizer.learningRate;
            logs.durationMs = now() - start;
            history.append(epoch, logs);
            yield* invoke(plan, "onEpochEnd", onEpochEnd, [epoch, logs, ctx]);
            if (plan.async) yield TICK;
            if (stop) break;
        }
        yield* invoke(plan, "onTrainEnd", withHook("onTrainEnd"), [logs, ctx]);
        return history;
    }

    /** One optimizer step on a batch. Returns the pre-update loss and the batch predictions. */
    private trainStep(
        x: Matrix,
        y: Matrix,
        gradBuffer: Matrix | null,
        where: string,
    ): { loss: number; prediction: Matrix } {
        const layers = this.layerList;
        const loss = this.lossFn as Loss;
        const optimizer = this.optimizerInstance as Optimizer;
        const params = this.parameterList();

        let output = x;
        for (let i = 0; i < layers.length; i++) output = layers[i].forward(output, true);
        const lossValue = loss.compute(output, y) + (this.regularizedLayers.length > 0 ? this.regularizationLoss() : 0);
        if (!Number.isFinite(lossValue)) {
            throw new TrainingError(
                `Training diverged: the loss became ${lossValue} ${where}. This usually means the learning rate ` +
                    `(${optimizer.learningRate}) is too high: try a smaller one (e.g. ${optimizer.learningRate / 10}), ` +
                    "scale your inputs (StandardScaler), or set clipNorm on the optimizer",
            );
        }

        const last = layers.length - 1;
        const out = gradBuffer ?? undefined;
        let grad: Matrix | null = null;
        let next = last;
        const lastLayer = layers[last];
        if (typeof loss.fusedGradient === "function") {
            // Closed-form dL/dz for sigmoid + bce and softmax + (s)cce: faster and stabler.
            if (lastLayer instanceof Dense) {
                const dz = loss.fusedGradient(lastLayer.activation.name, output, y, out);
                if (dz !== null) {
                    grad = lastLayer.propagateFromPreActivation(dz, last > 0);
                    next = last - 1;
                }
            } else if (lastLayer instanceof ActivationLayer) {
                const dz = loss.fusedGradient(lastLayer.activation.name, output, y, out);
                if (dz !== null) {
                    grad = dz; // dL/d(activation input) is exactly the fused gradient
                    next = last - 1;
                }
            }
        }
        if (next === last) grad = loss.gradient(output, y, out);
        for (let i = next; i >= 0; i--) {
            const layer = layers[i];
            grad = isBaseLayer(layer) ? layer.propagate(grad as Matrix, i > 0) : layer.backward(grad as Matrix);
        }
        optimizer.step(params);
        return { loss: lossValue, prediction: output };
    }

    /** Inference-mode forward pass over all rows of `x`, in chunks, into a new Matrix. */
    private forwardAll(x: Matrix, batchSize: number): Matrix {
        const outputSize = this.outputSize as number;
        const result = new Matrix(x.rows, outputSize);
        for (let from = 0; from < x.rows; from += batchSize) {
            const to = Math.min(from + batchSize, x.rows);
            let output = sliceRowsView(x, from, to);
            for (const layer of this.layerList) output = layer.forward(output, false);
            result.data.set(output.data, from * outputSize);
        }
        return result;
    }

    /** Batch-size-weighted loss (with regularization) and metrics, in inference mode. */
    private evaluateMatrices(x: Matrix, y: Matrix, batchSize: number): Logs {
        const loss = this.lossFn as Loss;
        const metrics = this.metricList;
        let lossSum = 0;
        const metricSums = new Float64Array(metrics.length);
        for (let from = 0; from < x.rows; from += batchSize) {
            const to = Math.min(from + batchSize, x.rows);
            const size = to - from;
            let output = sliceRowsView(x, from, to);
            for (const layer of this.layerList) output = layer.forward(output, false);
            const target = sliceRowsView(y, from, to);
            lossSum += loss.compute(output, target) * size;
            for (let m = 0; m < metrics.length; m++) {
                const value = metrics[m].compute(output, target);
                metricSums[m] += (metrics[m].name === "rootMeanSquaredError" ? value * value : value) * size;
            }
        }
        const logs: Logs = { loss: lossSum / x.rows + this.regularizationLoss() };
        for (let m = 0; m < metrics.length; m++) {
            const mean = metricSums[m] / x.rows;
            logs[metrics[m].name] = metrics[m].name === "rootMeanSquaredError" ? Math.sqrt(mean) : mean;
        }
        return logs;
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/** Lets timers, rendering and abort events run: `scheduler.yield()` where available, else a 0 ms timeout. */
function yieldToEventLoop(): Promise<void> {
    const scheduler = (globalThis as { scheduler?: { yield?: () => Promise<void> } }).scheduler;
    if (typeof scheduler?.yield === "function") return scheduler.yield();
    return new Promise((resolve) => setTimeout(resolve, 0));
}

/** Calls `hook` on each callback; yields returned promises in async mode, rejects them in sync mode. */
function* invoke(
    plan: FitPlan,
    hook: Hook,
    callbacks: readonly Callback[],
    args: unknown[],
): Generator<unknown, void, unknown> {
    for (const cb of callbacks) {
        const fn = cb[hook] as (...a: unknown[]) => unknown;
        const result = fn.apply(cb, args);
        if (isThenable(result)) {
            if (plan.async) {
                yield result;
            } else {
                Promise.resolve(result).catch(() => {}); // it can no longer be awaited; avoid an unhandled rejection
                throw new ValidationError(
                    `A callback's ${hook}() returned a Promise, but fit() is synchronous and cannot await it. ` +
                        "Use `await model.fitAsync(x, y, options)` instead",
                );
            }
        }
    }
}

function isLayer(value: unknown): value is Layer {
    if (value === null || typeof value !== "object") return false;
    const layer = value as Partial<Layer>;
    return (
        typeof layer.forward === "function" &&
        typeof layer.backward === "function" &&
        typeof layer.build === "function" &&
        typeof layer.parameters === "function" &&
        typeof layer.getConfig === "function" &&
        typeof layer.name === "string"
    );
}

function activationName(layer: Layer): string {
    const activation = (layer as { activation?: { name?: unknown } }).activation;
    return activation !== undefined && typeof activation.name === "string" ? activation.name : "-";
}

function toMatrix(value: MatrixLike, where: string, label: string): Matrix {
    if (value instanceof Matrix) return value;
    if (!Array.isArray(value)) {
        throw new ValidationError(
            `${where}: ${label} must be a Matrix, an array of rows (number[][]) or one sample (number[]), got ${describeValue(value)}`,
        );
    }
    return Matrix.from(value);
}

/** Zero-copy view of rows [from, to). */
function sliceRowsView(m: Matrix, from: number, to: number): Matrix {
    return new Matrix(to - from, m.cols, m.data.subarray(from * m.cols, to * m.cols));
}

function checkFinite(m: Matrix, where: string, label: string): void {
    const data = m.data;
    for (let i = 0; i < data.length; i++) {
        if (!Number.isFinite(data[i])) {
            const row = Math.floor(i / m.cols);
            throw new ValidationError(
                `${where}: ${label} contains ${data[i]} at row ${row}, column ${i - row * m.cols}; ` +
                    "clean or impute the data first (all values must be finite numbers)",
            );
        }
    }
}
