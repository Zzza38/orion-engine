/**
 * Evaluation metrics.
 *
 * Every metric returns a batch-level scalar that is batch-decomposable: the batch-size-weighted
 * mean of per-batch values equals the value over the whole dataset (rootMeanSquaredError is the
 * exception; average its square instead).
 */
import type { Matrix } from "./core/matrix.js";
import { ShapeError, ValidationError } from "./core/errors.js";
import type { Metric, MetricIdentifier, MetricName } from "./core/types.js";

/** Every built-in metric name (canonical, without aliases). */
export const METRIC_NAMES: readonly MetricName[] = Object.freeze([
    "accuracy",
    "binaryAccuracy",
    "categoricalAccuracy",
    "sparseCategoricalAccuracy",
    "meanSquaredError",
    "meanAbsoluteError",
    "rootMeanSquaredError",
] as const);

/** Short aliases and the canonical metric each one names. */
const METRIC_ALIASES: Readonly<Record<string, MetricName>> = Object.freeze({
    mse: "meanSquaredError",
    mae: "meanAbsoluteError",
    rmse: "rootMeanSquaredError",
});

/** Decision threshold used by binary accuracy. */
const THRESHOLD = 0.5;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

function checkNonEmpty(op: string, prediction: Matrix): void {
    if (prediction.rows === 0 || prediction.cols === 0) {
        throw new ShapeError(`${op}: prediction is empty [${prediction.rows}, ${prediction.cols}]`);
    }
}

function checkDense(op: string, prediction: Matrix, target: Matrix): void {
    checkNonEmpty(op, prediction);
    if (target.rows !== prediction.rows || target.cols !== prediction.cols) {
        throw new ShapeError(
            `${op}: prediction [${prediction.rows}, ${prediction.cols}] and target [${target.rows}, ${target.cols}] shapes differ`,
        );
    }
}

/** Index of the largest value in data[base, base + cols). Ties resolve to the first index. */
function argmax(data: Float64Array, base: number, cols: number): number {
    let best = 0;
    let bestValue = data[base];
    for (let c = 1; c < cols; c++) {
        const v = data[base + c];
        if (v > bestValue) {
            bestValue = v;
            best = c;
        }
    }
    return best;
}

function squaredErrorMean(op: string, prediction: Matrix, target: Matrix): number {
    checkDense(op, prediction, target);
    const P = prediction.data, Y = target.data;
    let sum = 0;
    for (let i = 0; i < P.length; i++) {
        const d = P[i] - Y[i];
        sum += d * d;
    }
    return sum / P.length;
}

// ---------------------------------------------------------------------------------------------
// Metric implementations
// ---------------------------------------------------------------------------------------------

/**
 * Fraction of elements where prediction and target fall on the same side of 0.5.
 * Both are thresholded (`> 0.5` is class 1), so soft labels count by their rounded class.
 */
function binaryAccuracy(prediction: Matrix, target: Matrix): number {
    checkDense("binaryAccuracy", prediction, target);
    const P = prediction.data, Y = target.data;
    let correct = 0;
    for (let i = 0; i < P.length; i++) {
        if ((P[i] > THRESHOLD) === (Y[i] > THRESHOLD)) correct++;
    }
    return correct / P.length;
}

/** Fraction of rows where argmax(prediction) equals argmax(target). */
function categoricalAccuracy(prediction: Matrix, target: Matrix): number {
    checkDense("categoricalAccuracy", prediction, target);
    const P = prediction.data, Y = target.data;
    const cols = prediction.cols;
    let correct = 0;
    for (let base = 0; base < P.length; base += cols) {
        if (argmax(P, base, cols) === argmax(Y, base, cols)) correct++;
    }
    return correct / prediction.rows;
}

/** Fraction of rows where argmax(prediction) equals the [batch, 1] integer class index target. */
function sparseCategoricalAccuracy(prediction: Matrix, target: Matrix): number {
    const op = "sparseCategoricalAccuracy";
    checkNonEmpty(op, prediction);
    if (target.rows !== prediction.rows || target.cols !== 1) {
        throw new ShapeError(
            `${op}: target must be [${prediction.rows}, 1] class indices, got [${target.rows}, ${target.cols}]`,
        );
    }
    const P = prediction.data, T = target.data;
    const cols = prediction.cols;
    let correct = 0;
    for (let r = 0; r < T.length; r++) {
        const index = T[r];
        if (!Number.isInteger(index) || index < 0 || index >= cols) {
            throw new ValidationError(`${op}: target[${r}] = ${index} is not a class index in [0, ${cols})`);
        }
        if (argmax(P, r * cols, cols) === index) correct++;
    }
    return correct / prediction.rows;
}

/**
 * Picks the accuracy variant from the shapes: a 1-column prediction is binary; a multi-column
 * prediction with a 1-column target is sparse categorical; otherwise categorical.
 */
function autoAccuracy(prediction: Matrix, target: Matrix): number {
    if (prediction.cols === 1) return binaryAccuracy(prediction, target);
    if (target.cols === 1) return sparseCategoricalAccuracy(prediction, target);
    return categoricalAccuracy(prediction, target);
}

function meanSquaredError(prediction: Matrix, target: Matrix): number {
    return squaredErrorMean("meanSquaredError", prediction, target);
}

function meanAbsoluteError(prediction: Matrix, target: Matrix): number {
    checkDense("meanAbsoluteError", prediction, target);
    const P = prediction.data, Y = target.data;
    let sum = 0;
    for (let i = 0; i < P.length; i++) sum += Math.abs(P[i] - Y[i]);
    return sum / P.length;
}

function rootMeanSquaredError(prediction: Matrix, target: Matrix): number {
    return Math.sqrt(squaredErrorMean("rootMeanSquaredError", prediction, target));
}

const IMPLEMENTATIONS: Readonly<Record<MetricName, (prediction: Matrix, target: Matrix) => number>> = {
    accuracy: autoAccuracy,
    binaryAccuracy,
    categoricalAccuracy,
    sparseCategoricalAccuracy,
    meanSquaredError,
    meanAbsoluteError,
    rootMeanSquaredError,
};

// ---------------------------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------------------------

function resolveName(value: unknown): MetricName {
    if (typeof value === "string") {
        if (Object.prototype.hasOwnProperty.call(IMPLEMENTATIONS, value)) return value as MetricName;
        if (Object.prototype.hasOwnProperty.call(METRIC_ALIASES, value)) return METRIC_ALIASES[value];
    }
    throw new ValidationError(
        `Unknown metric ${JSON.stringify(value)}. Valid metrics: ${METRIC_NAMES.join(", ")} ` +
            `(aliases: ${Object.keys(METRIC_ALIASES).join(", ")})`,
    );
}

/**
 * Resolves a metric from a name, an alias (`"mse"`, `"mae"`, `"rmse"`), or an existing `Metric`
 * instance (returned as-is). The returned metric's `name` is always the canonical name.
 *
 * `accuracy` auto-selects binary / sparse categorical / categorical accuracy from the shapes.
 * @throws ValidationError for unknown names.
 */
export function getMetric(id: MetricIdentifier): Metric {
    if (typeof id === "object" && id !== null && typeof (id as Partial<Metric>).compute === "function") {
        return id;
    }
    const name = resolveName(id);
    return { name, compute: IMPLEMENTATIONS[name] };
}
