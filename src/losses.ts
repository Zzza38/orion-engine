/**
 * Loss functions.
 *
 * Every loss reduces to the mean over the batch of a per-sample loss:
 *  - meanSquaredError, meanAbsoluteError, huber, binaryCrossentropy: per-sample loss is the
 *    mean over units (Keras convention), so the batch loss is the mean over all elements.
 *  - categoricalCrossentropy, sparseCategoricalCrossentropy: per-sample loss is the sum over
 *    classes, so the batch loss is (Σ over all elements) / batch.
 *
 * `gradient` returns dL/dPrediction of exactly that batch-mean value (already divided by the
 * batch size). `out` may alias `prediction`.
 */
import { Matrix } from "./core/matrix.js";
import { ShapeError, ValidationError } from "./core/errors.js";
import type {
    ActivationName,
    JsonValue,
    Loss,
    LossAlias,
    LossConfig,
    LossIdentifier,
    LossName,
} from "./core/types.js";

/** Every built-in loss name (canonical, without aliases). */
export const LOSS_NAMES: readonly LossName[] = Object.freeze([
    "meanSquaredError",
    "meanAbsoluteError",
    "huber",
    "binaryCrossentropy",
    "categoricalCrossentropy",
    "sparseCategoricalCrossentropy",
] as const);

/** Short aliases and the canonical loss each one names. */
const LOSS_ALIASES: Readonly<Record<LossAlias, LossName>> = Object.freeze({
    mse: "meanSquaredError",
    mae: "meanAbsoluteError",
    bce: "binaryCrossentropy",
    cce: "categoricalCrossentropy",
    scce: "sparseCategoricalCrossentropy",
});

/** Probabilities are clamped to [EPSILON, 1 - EPSILON] before taking logarithms. */
const EPSILON = 1e-7;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

function prepareOut(out: Matrix | undefined, like: Matrix, op: string): Matrix {
    if (out === undefined) return new Matrix(like.rows, like.cols);
    if (out.rows !== like.rows || out.cols !== like.cols) {
        throw new ShapeError(`${op}: output buffer is [${out.rows}, ${out.cols}], expected [${like.rows}, ${like.cols}]`);
    }
    return out;
}

function checkNonEmpty(op: string, prediction: Matrix): void {
    if (prediction.rows === 0 || prediction.cols === 0) {
        throw new ShapeError(`${op}: prediction is empty [${prediction.rows}, ${prediction.cols}]`);
    }
}

/** Dense targets: same shape as the prediction. */
function checkDense(op: string, prediction: Matrix, target: Matrix): void {
    checkNonEmpty(op, prediction);
    if (target.rows !== prediction.rows || target.cols !== prediction.cols) {
        throw new ShapeError(
            `${op}: prediction [${prediction.rows}, ${prediction.cols}] and target [${target.rows}, ${target.cols}] shapes differ`,
        );
    }
}

/** Sparse targets: [batch, 1] integer class indices in [0, classes). */
function checkSparse(op: string, prediction: Matrix, target: Matrix): void {
    checkNonEmpty(op, prediction);
    if (target.rows !== prediction.rows || target.cols !== 1) {
        throw new ShapeError(
            `${op}: target must be [${prediction.rows}, 1] class indices, got [${target.rows}, ${target.cols}]`,
        );
    }
    const classes = prediction.cols;
    const T = target.data;
    for (let r = 0; r < T.length; r++) {
        const index = T[r];
        if (!Number.isInteger(index) || index < 0 || index >= classes) {
            throw new ValidationError(`${op}: target[${r}] = ${index} is not a class index in [0, ${classes})`);
        }
    }
}

function clampProbability(p: number): number {
    return p < EPSILON ? EPSILON : p > 1 - EPSILON ? 1 - EPSILON : p;
}

// ---------------------------------------------------------------------------------------------
// Losses
// ---------------------------------------------------------------------------------------------

/** Shared defaults for built-in losses. */
abstract class BaseLoss implements Loss {
    abstract readonly name: LossName;
    abstract compute(prediction: Matrix, target: Matrix): number;
    abstract gradient(prediction: Matrix, target: Matrix, out?: Matrix): Matrix;

    fusedGradient(_activation: ActivationName, _prediction: Matrix, _target: Matrix, _out?: Matrix): Matrix | null {
        return null;
    }

    getConfig(): LossConfig {
        return { name: this.name };
    }
}

class MeanSquaredError extends BaseLoss {
    readonly name = "meanSquaredError" as const;

    compute(prediction: Matrix, target: Matrix): number {
        checkDense("meanSquaredError", prediction, target);
        const P = prediction.data, Y = target.data;
        let sum = 0;
        for (let i = 0; i < P.length; i++) {
            const d = P[i] - Y[i];
            sum += d * d;
        }
        return sum / P.length;
    }

    gradient(prediction: Matrix, target: Matrix, out?: Matrix): Matrix {
        checkDense("meanSquaredError.gradient", prediction, target);
        const result = prepareOut(out, prediction, "meanSquaredError.gradient");
        const P = prediction.data, Y = target.data, G = result.data;
        const s = 2 / P.length;
        for (let i = 0; i < P.length; i++) G[i] = s * (P[i] - Y[i]);
        return result;
    }
}

class MeanAbsoluteError extends BaseLoss {
    readonly name = "meanAbsoluteError" as const;

    compute(prediction: Matrix, target: Matrix): number {
        checkDense("meanAbsoluteError", prediction, target);
        const P = prediction.data, Y = target.data;
        let sum = 0;
        for (let i = 0; i < P.length; i++) sum += Math.abs(P[i] - Y[i]);
        return sum / P.length;
    }

    /** Subgradient: 0 where prediction equals target. */
    gradient(prediction: Matrix, target: Matrix, out?: Matrix): Matrix {
        checkDense("meanAbsoluteError.gradient", prediction, target);
        const result = prepareOut(out, prediction, "meanAbsoluteError.gradient");
        const P = prediction.data, Y = target.data, G = result.data;
        const s = 1 / P.length;
        for (let i = 0; i < P.length; i++) {
            const d = P[i] - Y[i];
            G[i] = d > 0 ? s : d < 0 ? -s : 0;
        }
        return result;
    }
}

class Huber extends BaseLoss {
    readonly name = "huber" as const;

    constructor(readonly delta: number) {
        super();
    }

    compute(prediction: Matrix, target: Matrix): number {
        checkDense("huber", prediction, target);
        const P = prediction.data, Y = target.data;
        const delta = this.delta;
        let sum = 0;
        for (let i = 0; i < P.length; i++) {
            const e = Math.abs(P[i] - Y[i]);
            sum += e <= delta ? 0.5 * e * e : delta * (e - 0.5 * delta);
        }
        return sum / P.length;
    }

    gradient(prediction: Matrix, target: Matrix, out?: Matrix): Matrix {
        checkDense("huber.gradient", prediction, target);
        const result = prepareOut(out, prediction, "huber.gradient");
        const P = prediction.data, Y = target.data, G = result.data;
        const delta = this.delta;
        const s = 1 / P.length;
        for (let i = 0; i < P.length; i++) {
            const e = P[i] - Y[i];
            G[i] = s * (e > delta ? delta : e < -delta ? -delta : e);
        }
        return result;
    }

    override getConfig(): LossConfig {
        return { name: this.name, delta: this.delta };
    }
}

class BinaryCrossentropy extends BaseLoss {
    readonly name = "binaryCrossentropy" as const;

    compute(prediction: Matrix, target: Matrix): number {
        checkDense("binaryCrossentropy", prediction, target);
        const P = prediction.data, Y = target.data;
        let sum = 0;
        for (let i = 0; i < P.length; i++) {
            const p = clampProbability(P[i]);
            const y = Y[i];
            sum -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
        }
        return sum / P.length;
    }

    /**
     * (p - y) / (p·(1 - p)) / (batch·units). Predictions outside [ε, 1 - ε] use the clamped value
     * (straight-through), so saturated predictions still receive a finite, non-zero signal.
     */
    gradient(prediction: Matrix, target: Matrix, out?: Matrix): Matrix {
        checkDense("binaryCrossentropy.gradient", prediction, target);
        const result = prepareOut(out, prediction, "binaryCrossentropy.gradient");
        const P = prediction.data, Y = target.data, G = result.data;
        const s = 1 / P.length;
        for (let i = 0; i < P.length; i++) {
            const p = clampProbability(P[i]);
            G[i] = (s * (p - Y[i])) / (p * (1 - p));
        }
        return result;
    }

    /** sigmoid + binaryCrossentropy → (p - y) / (batch·units). */
    override fusedGradient(activation: ActivationName, prediction: Matrix, target: Matrix, out?: Matrix): Matrix | null {
        if (activation !== "sigmoid") return null;
        checkDense("binaryCrossentropy.fusedGradient", prediction, target);
        const result = prepareOut(out, prediction, "binaryCrossentropy.fusedGradient");
        const P = prediction.data, Y = target.data, G = result.data;
        const s = 1 / P.length;
        for (let i = 0; i < P.length; i++) G[i] = s * (P[i] - Y[i]);
        return result;
    }
}

class CategoricalCrossentropy extends BaseLoss {
    readonly name = "categoricalCrossentropy" as const;

    compute(prediction: Matrix, target: Matrix): number {
        checkDense("categoricalCrossentropy", prediction, target);
        const P = prediction.data, Y = target.data;
        let sum = 0;
        for (let i = 0; i < P.length; i++) {
            const y = Y[i];
            if (y !== 0) sum -= y * Math.log(clampProbability(P[i]));
        }
        return sum / prediction.rows;
    }

    /** -y / p / batch, with p clamped to [ε, 1 - ε] (straight-through). */
    gradient(prediction: Matrix, target: Matrix, out?: Matrix): Matrix {
        checkDense("categoricalCrossentropy.gradient", prediction, target);
        const result = prepareOut(out, prediction, "categoricalCrossentropy.gradient");
        const P = prediction.data, Y = target.data, G = result.data;
        const s = -1 / prediction.rows;
        for (let i = 0; i < P.length; i++) {
            const y = Y[i];
            G[i] = y === 0 ? 0 : (s * y) / clampProbability(P[i]);
        }
        return result;
    }

    /**
     * softmax + categoricalCrossentropy → (p·Σy - y) / batch, per row. This is (p - y) / batch for
     * the usual targets whose rows sum to 1 (one-hot, label-smoothed), and stays exact for rows
     * that do not (soft counts, multi-hot targets).
     */
    override fusedGradient(activation: ActivationName, prediction: Matrix, target: Matrix, out?: Matrix): Matrix | null {
        if (activation !== "softmax") return null;
        checkDense("categoricalCrossentropy.fusedGradient", prediction, target);
        const result = prepareOut(out, prediction, "categoricalCrossentropy.fusedGradient");
        const P = prediction.data, Y = target.data, G = result.data;
        const cols = prediction.cols;
        const s = 1 / prediction.rows;
        for (let base = 0; base < P.length; base += cols) {
            const end = base + cols;
            let total = 0;
            for (let i = base; i < end; i++) total += Y[i];
            for (let i = base; i < end; i++) G[i] = s * (P[i] * total - Y[i]);
        }
        return result;
    }
}

class SparseCategoricalCrossentropy extends BaseLoss {
    readonly name = "sparseCategoricalCrossentropy" as const;

    compute(prediction: Matrix, target: Matrix): number {
        checkSparse("sparseCategoricalCrossentropy", prediction, target);
        const P = prediction.data, T = target.data;
        const cols = prediction.cols;
        let sum = 0;
        for (let r = 0; r < T.length; r++) sum -= Math.log(clampProbability(P[r * cols + T[r]]));
        return sum / prediction.rows;
    }

    /** -1 / p / batch at each target class (p clamped to [ε, 1 - ε]), 0 elsewhere. */
    gradient(prediction: Matrix, target: Matrix, out?: Matrix): Matrix {
        checkSparse("sparseCategoricalCrossentropy.gradient", prediction, target);
        const result = prepareOut(out, prediction, "sparseCategoricalCrossentropy.gradient");
        const P = prediction.data, T = target.data, G = result.data;
        const cols = prediction.cols;
        const s = -1 / prediction.rows;
        for (let r = 0; r < T.length; r++) {
            const base = r * cols;
            const hit = base + T[r];
            const p = P[hit];
            for (let i = base; i < base + cols; i++) G[i] = 0;
            G[hit] = s / clampProbability(p);
        }
        return result;
    }

    /** softmax + sparseCategoricalCrossentropy → (p - onehot(y)) / batch. */
    override fusedGradient(activation: ActivationName, prediction: Matrix, target: Matrix, out?: Matrix): Matrix | null {
        if (activation !== "softmax") return null;
        checkSparse("sparseCategoricalCrossentropy.fusedGradient", prediction, target);
        const result = prepareOut(out, prediction, "sparseCategoricalCrossentropy.fusedGradient");
        const P = prediction.data, T = target.data, G = result.data;
        const cols = prediction.cols;
        const s = 1 / prediction.rows;
        for (let r = 0; r < T.length; r++) {
            const base = r * cols;
            for (let i = base; i < base + cols; i++) G[i] = s * P[i];
            G[base + T[r]] -= s;
        }
        return result;
    }
}

// ---------------------------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------------------------

type ConfigParams = { readonly [param: string]: JsonValue | undefined };

const ACCEPTED_PARAMS: Record<LossName, readonly string[]> = {
    meanSquaredError: [],
    meanAbsoluteError: [],
    huber: ["delta"],
    binaryCrossentropy: [],
    categoricalCrossentropy: [],
    sparseCategoricalCrossentropy: [],
};

function resolveName(value: unknown): LossName {
    if (typeof value === "string") {
        if (Object.prototype.hasOwnProperty.call(ACCEPTED_PARAMS, value)) return value as LossName;
        if (Object.prototype.hasOwnProperty.call(LOSS_ALIASES, value)) return LOSS_ALIASES[value as LossAlias];
    }
    throw new ValidationError(
        `Unknown loss ${JSON.stringify(value)}. Valid losses: ${LOSS_NAMES.join(", ")} ` +
            `(aliases: ${Object.keys(LOSS_ALIASES).join(", ")})`,
    );
}

function create(name: LossName, params: ConfigParams): Loss {
    const accepted = ACCEPTED_PARAMS[name];
    for (const key of Object.keys(params)) {
        if (key === "name" || params[key] === undefined || accepted.includes(key)) continue;
        const hint = accepted.length > 0 ? `accepted: ${accepted.join(", ")}` : "it takes no parameters";
        throw new ValidationError(`Loss "${name}": unknown parameter "${key}" (${hint})`);
    }
    switch (name) {
        case "meanSquaredError": return new MeanSquaredError();
        case "meanAbsoluteError": return new MeanAbsoluteError();
        case "huber": {
            const delta = params.delta ?? 1;
            if (typeof delta !== "number" || !Number.isFinite(delta) || delta <= 0) {
                throw new ValidationError(`Loss "huber": "delta" must be a positive finite number, got ${JSON.stringify(delta)}`);
            }
            return new Huber(delta);
        }
        case "binaryCrossentropy": return new BinaryCrossentropy();
        case "categoricalCrossentropy": return new CategoricalCrossentropy();
        case "sparseCategoricalCrossentropy": return new SparseCategoricalCrossentropy();
    }
}

/**
 * Resolves a loss from a name, an alias (`"mse"`, `"mae"`, `"bce"`, `"cce"`, `"scce"`), a config
 * (`{ name: "huber", delta: 2 }`), or an existing `Loss` instance (returned as-is).
 *
 * Defaults: huber `delta` = 1. Cross-entropies clamp probabilities to [1e-7, 1 - 1e-7].
 * @throws ValidationError for unknown names, unknown parameters, or invalid parameter values.
 */
export function getLoss(id: LossIdentifier): Loss {
    if (typeof id === "string") return create(resolveName(id), {});
    if (typeof id === "object" && id !== null) {
        const candidate = id as Partial<Loss>;
        if (typeof candidate.compute === "function" && typeof candidate.gradient === "function") {
            return id as Loss;
        }
        const config = id as LossConfig;
        return create(resolveName(config.name), config);
    }
    throw new ValidationError(`Invalid loss identifier ${String(id)}: expected a name, a config object, or a Loss`);
}
