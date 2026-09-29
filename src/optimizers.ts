/**
 * Gradient-based optimizers.
 *
 * Every optimizer updates `Parameter.value` in place from `Parameter.grad`. Shared behaviour:
 *
 *  - Parameters with `trainable === false` are skipped entirely (no update, no state).
 *  - Weight decay is never applied to parameters with `regularize === false` (biases).
 *  - `clipNorm` / `clipValue` are applied per parameter to an internal scratch copy of the
 *    gradient: first the L2-norm rescale, then the element-wise clamp. `param.grad` is never
 *    modified by an optimizer, so callers can still inspect the raw gradients after `step`.
 *  - Per-parameter state (moments, accumulators) is keyed by `Parameter` object identity and
 *    allocated lazily on the first step that touches the parameter.
 *  - `iterations` counts `step` calls; Adam-family bias correction uses it as `t`.
 *
 * Notation in the update rules below: w = parameter, g = (clipped) gradient, η = learning rate,
 * t = iteration number starting at 1, all operations element-wise.
 */
import { ShapeError, ValidationError } from "./core/errors.js";
import type {
    JsonValue,
    Optimizer,
    OptimizerConfig,
    OptimizerIdentifier,
    OptimizerName,
    Parameter,
} from "./core/types.js";

/** Every optimizer name accepted by {@link getOptimizer}. */
export const OPTIMIZER_NAMES: readonly OptimizerName[] = Object.freeze([
    "sgd",
    "adam",
    "adamw",
    "rmsprop",
    "adagrad",
] as const);

/** Options shared by every optimizer. */
export interface OptimizerOptions {
    /** Step size η. Finite and ≥ 0. Each optimizer has its own default. */
    learningRate?: number;
    /** Rescale each parameter's gradient so its L2 norm is at most this value. Must be > 0. */
    clipNorm?: number;
    /** Clamp every gradient element to [-clipValue, clipValue]. Must be > 0. */
    clipValue?: number;
}

const DISPLAY_NAMES: Readonly<Record<OptimizerName, string>> = {
    sgd: "SGD",
    adam: "Adam",
    adamw: "AdamW",
    rmsprop: "RMSprop",
    adagrad: "Adagrad",
};

const BASE_KEYS = ["learningRate", "clipNorm", "clipValue"] as const;

const OPTION_KEYS: Readonly<Record<OptimizerName, readonly string[]>> = {
    sgd: [...BASE_KEYS, "momentum", "nesterov", "weightDecay"],
    adam: [...BASE_KEYS, "beta1", "beta2", "epsilon", "amsgrad"],
    adamw: [...BASE_KEYS, "beta1", "beta2", "epsilon", "amsgrad", "weightDecay"],
    rmsprop: [...BASE_KEYS, "rho", "momentum", "epsilon", "centered"],
    adagrad: [...BASE_KEYS, "initialAccumulatorValue", "epsilon"],
};

// ---------------------------------------------------------------------------------------------
// Option validation helpers
// ---------------------------------------------------------------------------------------------

type OptionBag = Readonly<Record<string, unknown>>;

interface NumberRule {
    readonly test: (value: number) => boolean;
    readonly description: string;
}

const NON_NEGATIVE: NumberRule = { test: (v) => Number.isFinite(v) && v >= 0, description: "a finite number >= 0" };
const POSITIVE: NumberRule = { test: (v) => Number.isFinite(v) && v > 0, description: "a finite number > 0" };
const UNIT_INTERVAL: NumberRule = { test: (v) => v >= 0 && v < 1, description: "a number in [0, 1)" };

function describeValue(value: unknown): string {
    return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function readOptionalNumber(label: string, options: OptionBag, key: string, rule: NumberRule): number | undefined {
    const value = options[key];
    if (value === undefined || value === null) return undefined;
    if (typeof value !== "number" || !rule.test(value)) {
        throw new ValidationError(`${label}: "${key}" must be ${rule.description}, got ${describeValue(value)}`);
    }
    return value;
}

function readNumber(label: string, options: OptionBag, key: string, fallback: number, rule: NumberRule): number {
    return readOptionalNumber(label, options, key, rule) ?? fallback;
}

function readBoolean(label: string, options: OptionBag, key: string, fallback: boolean): boolean {
    const value = options[key];
    if (value === undefined || value === null) return fallback;
    if (typeof value !== "boolean") {
        throw new ValidationError(`${label}: "${key}" must be a boolean, got ${describeValue(value)}`);
    }
    return value;
}

function checkOptionKeys(label: string, options: unknown, allowed: readonly string[]): asserts options is OptionBag {
    if (options === null || typeof options !== "object" || Array.isArray(options)) {
        throw new ValidationError(`${label}: options must be an object, got ${describeValue(options)}`);
    }
    for (const key of Object.keys(options)) {
        if (!allowed.includes(key)) {
            throw new ValidationError(`${label}: unknown option "${key}". Valid options: ${allowed.join(", ")}`);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Base class
// ---------------------------------------------------------------------------------------------

/**
 * Shared machinery for the built-in optimizers: option validation, gradient clipping, lazy
 * per-parameter state, iteration counting and config serialization. Subclasses implement
 * {@link BaseOptimizer.update} for a single parameter, and must implement `name` as a getter
 * (it is read by the base constructor, before subclass fields exist).
 */
export abstract class BaseOptimizer implements Optimizer {
    abstract readonly name: OptimizerName;
    /** Per-parameter gradient L2-norm limit, or undefined when disabled. */
    readonly clipNorm: number | undefined;
    /** Per-element gradient magnitude limit, or undefined when disabled. */
    readonly clipValue: number | undefined;

    private lr: number;
    private stepCount = 0;
    private state = new WeakMap<Parameter, Float64Array[]>();
    private scratch = new Float64Array(0);

    protected constructor(options: OptimizerOptions, defaultLearningRate: number) {
        const label = this.label;
        checkOptionKeys(label, options, OPTION_KEYS[this.kind]);
        this.lr = readNumber(label, options, "learningRate", defaultLearningRate, NON_NEGATIVE);
        this.clipNorm = readOptionalNumber(label, options, "clipNorm", POSITIVE);
        this.clipValue = readOptionalNumber(label, options, "clipValue", POSITIVE);
    }

    /** Current learning rate η. Settable between steps (schedules, callbacks); must be finite and ≥ 0. */
    get learningRate(): number {
        return this.lr;
    }

    set learningRate(value: number) {
        if (typeof value !== "number" || !NON_NEGATIVE.test(value)) {
            throw new ValidationError(
                `${this.label}: learningRate must be ${NON_NEGATIVE.description}, got ${describeValue(value)}`,
            );
        }
        this.lr = value;
    }

    /** Number of `step` calls since construction or the last `reset`. */
    get iterations(): number {
        return this.stepCount;
    }

    /**
     * Applies one update to every trainable parameter. Throws `ShapeError` (before touching any
     * parameter) if a trainable parameter's `grad` shape differs from its `value` shape.
     * Does not zero gradients.
     */
    step(params: readonly Parameter[]): void {
        for (const param of params) {
            if (param.trainable === false) continue;
            const { value, grad } = param;
            if (value.rows !== grad.rows || value.cols !== grad.cols) {
                throw new ShapeError(
                    `${this.label}: parameter "${param.name}" has value [${value.rows}, ${value.cols}] ` +
                        `but grad [${grad.rows}, ${grad.cols}]`,
                );
            }
        }
        const t = ++this.stepCount;
        const lr = this.lr;
        for (const param of params) {
            if (param.trainable === false) continue;
            this.update(param, this.clippedGradient(param.grad.data), lr, t);
        }
    }

    /** Clears all per-parameter state and resets `iterations` to 0. Hyperparameters are kept. */
    reset(): void {
        this.stepCount = 0;
        this.state = new WeakMap();
        this.scratch = new Float64Array(0);
    }

    /** JSON-safe config; `getOptimizer(opt.getConfig())` builds an equivalent (fresh-state) optimizer. */
    getConfig(): OptimizerConfig {
        const config: OptimizerConfig = { name: this.name, learningRate: this.lr, ...this.hyperparameters() };
        if (this.clipNorm !== undefined) config.clipNorm = this.clipNorm;
        if (this.clipValue !== undefined) config.clipValue = this.clipValue;
        return config;
    }

    /**
     * Updates one trainable parameter in place.
     * @param grad Gradient data (possibly a clipped scratch copy that may be longer than the
     *   parameter; iterate over `param.value.data.length`). Must not be written to.
     * @param lr Learning rate for this step.
     * @param t Iteration number, starting at 1.
     */
    protected abstract update(param: Parameter, grad: Float64Array, lr: number, t: number): void;

    /** Optimizer-specific hyperparameters for `getConfig` (everything except name, learningRate, clipping). */
    protected abstract hyperparameters(): { [key: string]: JsonValue };

    /** Human-readable optimizer name for error messages. */
    protected get label(): string {
        return DISPLAY_NAMES[this.kind];
    }

    /** `name`, readable during construction because subclasses implement it as a prototype getter. */
    private get kind(): OptimizerName {
        return this.name;
    }

    /**
     * Returns the parameter's state buffers, allocating `count` zero-initialized (or
     * `initial`-filled) buffers of the parameter's size on first use.
     */
    protected slots(param: Parameter, count: number, initial = 0): Float64Array[] {
        let buffers = this.state.get(param);
        if (buffers === undefined) {
            const size = param.value.data.length;
            buffers = [];
            for (let k = 0; k < count; k++) {
                const buffer = new Float64Array(size);
                if (initial !== 0) buffer.fill(initial);
                buffers.push(buffer);
            }
            this.state.set(param, buffers);
        }
        return buffers;
    }

    /** Applies clipNorm then clipValue to a scratch copy. Returns `grad` itself when no clipping is configured. */
    private clippedGradient(grad: Float64Array): Float64Array {
        const { clipNorm, clipValue } = this;
        if (clipNorm === undefined && clipValue === undefined) return grad;
        const n = grad.length;
        if (this.scratch.length < n) this.scratch = new Float64Array(n);
        const out = this.scratch;

        let factor = 1;
        if (clipNorm !== undefined) {
            let sumSquares = 0;
            for (let i = 0; i < n; i++) sumSquares += grad[i] * grad[i];
            const norm = Math.sqrt(sumSquares);
            if (norm > clipNorm) factor = clipNorm / norm;
        }
        if (clipValue !== undefined) {
            const low = -clipValue;
            for (let i = 0; i < n; i++) {
                const g = grad[i] * factor;
                out[i] = g > clipValue ? clipValue : g < low ? low : g;
            }
        } else {
            for (let i = 0; i < n; i++) out[i] = grad[i] * factor;
        }
        return out;
    }
}

// ---------------------------------------------------------------------------------------------
// SGD
// ---------------------------------------------------------------------------------------------

/** Options for {@link SGD}. */
export interface SGDOptions extends OptimizerOptions {
    /** Momentum coefficient μ in [0, 1). Default 0 (plain SGD). */
    momentum?: number;
    /** Use Nesterov momentum. Default false. */
    nesterov?: boolean;
    /** Coupled L2 weight decay λ ≥ 0, added to the gradient. Default 0. */
    weightDecay?: number;
}

/**
 * Stochastic gradient descent with optional (Nesterov) momentum and coupled L2 weight decay.
 * Defaults: learningRate 0.01, momentum 0, nesterov false, weightDecay 0.
 *
 * ```text
 * g ← g + λ·w                  (λ = 0 when regularize === false)
 * μ = 0:     w ← w − η·g
 * μ > 0:     v ← μ·v + g        (v₀ = 0)
 *            w ← w − η·v              (classic)
 *            w ← w − η·(g + μ·v)      (Nesterov)
 * ```
 */
export class SGD extends BaseOptimizer {
    /** Momentum coefficient μ. */
    readonly momentum: number;
    /** Whether Nesterov momentum is used. */
    readonly nesterov: boolean;
    /** Coupled L2 weight decay λ. */
    readonly weightDecay: number;

    constructor(options: SGDOptions = {}) {
        super(options, 0.01);
        const bag = options as OptionBag;
        this.momentum = readNumber(this.label, bag, "momentum", 0, UNIT_INTERVAL);
        this.nesterov = readBoolean(this.label, bag, "nesterov", false);
        this.weightDecay = readNumber(this.label, bag, "weightDecay", 0, NON_NEGATIVE);
    }

    /** Always "sgd". */
    get name(): "sgd" {
        return "sgd";
    }

    protected update(param: Parameter, grad: Float64Array, lr: number): void {
        const w = param.value.data;
        const n = w.length;
        const decay = param.regularize === false ? 0 : this.weightDecay;
        const mu = this.momentum;
        if (mu === 0) {
            for (let i = 0; i < n; i++) w[i] -= lr * (grad[i] + decay * w[i]);
            return;
        }
        const velocity = this.slots(param, 1)[0];
        if (this.nesterov) {
            for (let i = 0; i < n; i++) {
                const g = grad[i] + decay * w[i];
                const v = mu * velocity[i] + g;
                velocity[i] = v;
                w[i] -= lr * (g + mu * v);
            }
        } else {
            for (let i = 0; i < n; i++) {
                const v = mu * velocity[i] + grad[i] + decay * w[i];
                velocity[i] = v;
                w[i] -= lr * v;
            }
        }
    }

    protected hyperparameters(): { [key: string]: JsonValue } {
        return { momentum: this.momentum, nesterov: this.nesterov, weightDecay: this.weightDecay };
    }
}

// ---------------------------------------------------------------------------------------------
// Adam / AdamW
// ---------------------------------------------------------------------------------------------

/** Options for {@link Adam}. */
export interface AdamOptions extends OptimizerOptions {
    /** Exponential decay rate β₁ for the first moment, in [0, 1). Default 0.9. */
    beta1?: number;
    /** Exponential decay rate β₂ for the second moment, in [0, 1). Default 0.999. */
    beta2?: number;
    /** Numerical-stability constant ε > 0 added to the denominator. Default 1e-7. */
    epsilon?: number;
    /** Use the AMSGrad variant (running maximum of the second moment). Default false. */
    amsgrad?: boolean;
}

/**
 * Adam (Kingma & Ba, 2015) with optional AMSGrad.
 * Defaults: learningRate 0.001, beta1 0.9, beta2 0.999, epsilon 1e-7, amsgrad false.
 *
 * ```text
 * m ← β₁·m + (1 − β₁)·g
 * v ← β₂·v + (1 − β₂)·g²
 * v̄ = v                        (AMSGrad: v̄ ← max(v̄, v))
 * m̂ = m / (1 − β₁ᵗ),  v̂ = v̄ / (1 − β₂ᵗ)
 * w ← w − η·m̂ / (√v̂ + ε)
 * ```
 * `t` is the optimizer's `iterations` count (shared by all parameters).
 */
export class Adam extends BaseOptimizer {
    /** First-moment decay β₁. */
    readonly beta1: number;
    /** Second-moment decay β₂. */
    readonly beta2: number;
    /** Denominator stability constant ε. */
    readonly epsilon: number;
    /** Whether the AMSGrad variant is used. */
    readonly amsgrad: boolean;
    /** Decoupled weight decay λ (0 for Adam; set by AdamW). */
    protected decoupledWeightDecay = 0;

    constructor(options: AdamOptions = {}) {
        super(options, 0.001);
        const bag = options as OptionBag;
        this.beta1 = readNumber(this.label, bag, "beta1", 0.9, UNIT_INTERVAL);
        this.beta2 = readNumber(this.label, bag, "beta2", 0.999, UNIT_INTERVAL);
        this.epsilon = readNumber(this.label, bag, "epsilon", 1e-7, POSITIVE);
        this.amsgrad = readBoolean(this.label, bag, "amsgrad", false);
    }

    /** "adam" (or "adamw" for {@link AdamW}). */
    get name(): "adam" | "adamw" {
        return "adam";
    }

    protected update(param: Parameter, grad: Float64Array, lr: number, t: number): void {
        const w = param.value.data;
        const n = w.length;
        const { beta1, beta2, epsilon } = this;
        const oneMinusBeta1 = 1 - beta1;
        const oneMinusBeta2 = 1 - beta2;
        const stepSize = lr / (1 - Math.pow(beta1, t));
        const invSqrtCorrection2 = 1 / Math.sqrt(1 - Math.pow(beta2, t));
        const decay = param.regularize === false ? 0 : this.decoupledWeightDecay;
        const keep = 1 - lr * decay;

        const buffers = this.slots(param, this.amsgrad ? 3 : 2);
        const m = buffers[0];
        const v = buffers[1];
        if (this.amsgrad) {
            const vMax = buffers[2];
            for (let i = 0; i < n; i++) {
                const g = grad[i];
                const mi = beta1 * m[i] + oneMinusBeta1 * g;
                const vi = beta2 * v[i] + oneMinusBeta2 * g * g;
                m[i] = mi;
                v[i] = vi;
                const vBar = vi > vMax[i] ? vi : vMax[i];
                vMax[i] = vBar;
                w[i] = w[i] * keep - (stepSize * mi) / (Math.sqrt(vBar) * invSqrtCorrection2 + epsilon);
            }
        } else {
            for (let i = 0; i < n; i++) {
                const g = grad[i];
                const mi = beta1 * m[i] + oneMinusBeta1 * g;
                const vi = beta2 * v[i] + oneMinusBeta2 * g * g;
                m[i] = mi;
                v[i] = vi;
                w[i] = w[i] * keep - (stepSize * mi) / (Math.sqrt(vi) * invSqrtCorrection2 + epsilon);
            }
        }
    }

    protected hyperparameters(): { [key: string]: JsonValue } {
        return { beta1: this.beta1, beta2: this.beta2, epsilon: this.epsilon, amsgrad: this.amsgrad };
    }
}

/** Options for {@link AdamW}. */
export interface AdamWOptions extends AdamOptions {
    /** Decoupled weight decay λ ≥ 0. Default 0.01. */
    weightDecay?: number;
}

/**
 * Adam with decoupled weight decay (Loshchilov & Hutter, 2019). The decay shrinks weights
 * directly instead of being added to the gradient, so it is not rescaled by the adaptive
 * denominator. Parameters with `regularize === false` are not decayed.
 * Defaults: as {@link Adam}, plus weightDecay 0.01.
 *
 * ```text
 * w ← w·(1 − η·λ) − η·m̂ / (√v̂ + ε)      (m̂, v̂ exactly as in Adam)
 * ```
 */
export class AdamW extends Adam {
    /** Decoupled weight decay λ. */
    readonly weightDecay: number;

    constructor(options: AdamWOptions = {}) {
        super(options);
        this.weightDecay = readNumber(this.label, options as OptionBag, "weightDecay", 0.01, NON_NEGATIVE);
        this.decoupledWeightDecay = this.weightDecay;
    }

    /** Always "adamw". */
    override get name(): "adamw" {
        return "adamw";
    }

    protected override hyperparameters(): { [key: string]: JsonValue } {
        return { ...super.hyperparameters(), weightDecay: this.weightDecay };
    }
}

// ---------------------------------------------------------------------------------------------
// RMSprop
// ---------------------------------------------------------------------------------------------

/** Options for {@link RMSprop}. */
export interface RMSpropOptions extends OptimizerOptions {
    /** Decay rate ρ of the squared-gradient average, in [0, 1). Default 0.9. */
    rho?: number;
    /** Momentum coefficient μ in [0, 1). Default 0. */
    momentum?: number;
    /** Numerical-stability constant ε > 0. Default 1e-7. */
    epsilon?: number;
    /** Normalize by the estimated gradient variance instead of the raw second moment. Default false. */
    centered?: boolean;
}

/**
 * RMSprop (Hinton, 2012) with optional momentum and centering.
 * Defaults: learningRate 0.001, rho 0.9, momentum 0, epsilon 1e-7, centered false.
 *
 * ```text
 * s ← ρ·s + (1 − ρ)·g²
 * centered:  ḡ ← ρ·ḡ + (1 − ρ)·g,  d = √(s − ḡ²) + ε
 * otherwise: d = √s + ε
 * μ = 0:     w ← w − η·g / d
 * μ > 0:     b ← μ·b + g / d,  w ← w − η·b
 * ```
 */
export class RMSprop extends BaseOptimizer {
    /** Squared-gradient decay ρ. */
    readonly rho: number;
    /** Momentum coefficient μ. */
    readonly momentum: number;
    /** Denominator stability constant ε. */
    readonly epsilon: number;
    /** Whether the centered variant is used. */
    readonly centered: boolean;

    constructor(options: RMSpropOptions = {}) {
        super(options, 0.001);
        const bag = options as OptionBag;
        this.rho = readNumber(this.label, bag, "rho", 0.9, UNIT_INTERVAL);
        this.momentum = readNumber(this.label, bag, "momentum", 0, UNIT_INTERVAL);
        this.epsilon = readNumber(this.label, bag, "epsilon", 1e-7, POSITIVE);
        this.centered = readBoolean(this.label, bag, "centered", false);
    }

    /** Always "rmsprop". */
    get name(): "rmsprop" {
        return "rmsprop";
    }

    protected update(param: Parameter, grad: Float64Array, lr: number): void {
        const w = param.value.data;
        const n = w.length;
        const { rho, momentum: mu, epsilon, centered } = this;
        const oneMinusRho = 1 - rho;
        const useMomentum = mu > 0;
        const buffers = this.slots(param, 1 + (centered ? 1 : 0) + (useMomentum ? 1 : 0));
        const square = buffers[0];
        const mean = centered ? buffers[1] : undefined;
        const velocity = useMomentum ? buffers[centered ? 2 : 1] : undefined;

        for (let i = 0; i < n; i++) {
            const g = grad[i];
            const s = rho * square[i] + oneMinusRho * g * g;
            square[i] = s;
            let variance = s;
            if (mean !== undefined) {
                const a = rho * mean[i] + oneMinusRho * g;
                mean[i] = a;
                variance = s - a * a;
                if (variance < 0) variance = 0; // guard against rounding
            }
            const scaled = g / (Math.sqrt(variance) + epsilon);
            if (velocity !== undefined) {
                const b = mu * velocity[i] + scaled;
                velocity[i] = b;
                w[i] -= lr * b;
            } else {
                w[i] -= lr * scaled;
            }
        }
    }

    protected hyperparameters(): { [key: string]: JsonValue } {
        return { rho: this.rho, momentum: this.momentum, epsilon: this.epsilon, centered: this.centered };
    }
}

// ---------------------------------------------------------------------------------------------
// Adagrad
// ---------------------------------------------------------------------------------------------

/** Options for {@link Adagrad}. */
export interface AdagradOptions extends OptimizerOptions {
    /** Starting value (≥ 0) of every squared-gradient accumulator. Default 0.1. */
    initialAccumulatorValue?: number;
    /** Numerical-stability constant ε > 0. Default 1e-7. */
    epsilon?: number;
}

/**
 * Adagrad (Duchi et al., 2011): per-element learning rates that shrink with the accumulated
 * squared gradient. Defaults: learningRate 0.01, initialAccumulatorValue 0.1, epsilon 1e-7.
 *
 * ```text
 * a ← a + g²                   (a₀ = initialAccumulatorValue)
 * w ← w − η·g / (√a + ε)
 * ```
 */
export class Adagrad extends BaseOptimizer {
    /** Initial accumulator value a₀. */
    readonly initialAccumulatorValue: number;
    /** Denominator stability constant ε. */
    readonly epsilon: number;

    constructor(options: AdagradOptions = {}) {
        super(options, 0.01);
        const bag = options as OptionBag;
        this.initialAccumulatorValue = readNumber(this.label, bag, "initialAccumulatorValue", 0.1, NON_NEGATIVE);
        this.epsilon = readNumber(this.label, bag, "epsilon", 1e-7, POSITIVE);
    }

    /** Always "adagrad". */
    get name(): "adagrad" {
        return "adagrad";
    }

    protected update(param: Parameter, grad: Float64Array, lr: number): void {
        const w = param.value.data;
        const n = w.length;
        const epsilon = this.epsilon;
        const accumulator = this.slots(param, 1, this.initialAccumulatorValue)[0];
        for (let i = 0; i < n; i++) {
            const g = grad[i];
            const a = accumulator[i] + g * g;
            accumulator[i] = a;
            w[i] -= (lr * g) / (Math.sqrt(a) + epsilon);
        }
    }

    protected hyperparameters(): { [key: string]: JsonValue } {
        return { initialAccumulatorValue: this.initialAccumulatorValue, epsilon: this.epsilon };
    }
}

// ---------------------------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------------------------

function resolveOptimizerName(name: unknown): OptimizerName {
    if (typeof name === "string") {
        const key = name.toLowerCase();
        if ((OPTIMIZER_NAMES as readonly string[]).includes(key)) return key as OptimizerName;
    }
    throw new ValidationError(`Unknown optimizer ${describeValue(name)}. Valid names: ${OPTIMIZER_NAMES.join(", ")}`);
}

function createOptimizer(name: OptimizerName, options: OptionBag): Optimizer {
    switch (name) {
        case "sgd":
            return new SGD(options);
        case "adam":
            return new Adam(options);
        case "adamw":
            return new AdamW(options);
        case "rmsprop":
            return new RMSprop(options);
        case "adagrad":
            return new Adagrad(options);
    }
}

function isOptimizer(value: object): value is Optimizer {
    return typeof (value as { step?: unknown }).step === "function";
}

/**
 * Resolves an optimizer identifier:
 *  - a name (`"adam"`, case-insensitive) → a new optimizer with default hyperparameters;
 *  - a config (`{ name: "adam", learningRate: 0.01 }`, e.g. from `getConfig()`) → a new optimizer
 *    with those hyperparameters and fresh state;
 *  - an `Optimizer` instance → returned as-is.
 *
 * Throws `ValidationError` for unknown names (listing the valid ones), unknown option keys, or
 * out-of-range hyperparameters.
 */
export function getOptimizer(identifier: OptimizerIdentifier): Optimizer {
    if (typeof identifier === "string") return createOptimizer(resolveOptimizerName(identifier), {});
    if (identifier !== null && typeof identifier === "object" && !Array.isArray(identifier)) {
        if (isOptimizer(identifier)) return identifier;
        const { name, ...options } = identifier;
        return createOptimizer(resolveOptimizerName(name), options);
    }
    throw new ValidationError(
        `Invalid optimizer ${describeValue(identifier)}: expected a name (${OPTIMIZER_NAMES.join(", ")}), ` +
            `a config object, or an Optimizer instance`,
    );
}
