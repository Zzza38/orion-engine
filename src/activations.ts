/**
 * Activation functions.
 *
 * Every activation works on a batch `Matrix` [batch, units] and exposes a forward pass plus the
 * matching backward pass (dL/dz). Element-wise activations may be computed in place: `out` may
 * alias `z` in `forward`, and `gradOutput` (or `a`) in `backward`.
 */
import { Matrix } from "./core/matrix.js";
import { ShapeError, ValidationError } from "./core/errors.js";
import type {
    Activation,
    ActivationConfig,
    ActivationIdentifier,
    ActivationName,
    JsonValue,
} from "./core/types.js";

/** Every built-in activation name. */
export const ACTIVATION_NAMES: readonly ActivationName[] = Object.freeze([
    "linear",
    "sigmoid",
    "tanh",
    "relu",
    "relu6",
    "leakyRelu",
    "elu",
    "selu",
    "gelu",
    "swish",
    "mish",
    "softplus",
    "softsign",
    "hardSigmoid",
    "softmax",
] as const);

/** SELU constants from Klambauer et al. (2017), matching Keras. */
const SELU_ALPHA = 1.6732632423543772848170429916717;
const SELU_SCALE = 1.0507009873554804934193349852946;
/** sqrt(2 / π), used by the tanh approximation of GELU. */
const GELU_C = 0.7978845608028654;
const GELU_K = 0.044715;

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

function checkBackwardShapes(name: string, z: Matrix, a: Matrix, g: Matrix): void {
    if (z.rows !== a.rows || z.cols !== a.cols || z.rows !== g.rows || z.cols !== g.cols) {
        throw new ShapeError(
            `${name}.backward: z [${z.rows}, ${z.cols}], a [${a.rows}, ${a.cols}] and ` +
                `gradOutput [${g.rows}, ${g.cols}] must share one shape`,
        );
    }
}

/** Numerically stable logistic function: never evaluates exp of a large positive number. */
function sigmoid(x: number): number {
    if (x >= 0) return 1 / (1 + Math.exp(-x));
    const e = Math.exp(x);
    return e / (1 + e);
}

/** Numerically stable log(1 + e^x). */
function softplus(x: number): number {
    return (x > 0 ? x : 0) + Math.log1p(Math.exp(-Math.abs(x)));
}

// ---------------------------------------------------------------------------------------------
// Element-wise activations
// ---------------------------------------------------------------------------------------------

/** Shared plumbing for activations where a_i depends only on z_i. */
abstract class ElementwiseActivation implements Activation {
    abstract readonly name: ActivationName;

    forward(z: Matrix, out?: Matrix): Matrix {
        const target = prepareOut(out, z, `${this.name}.forward`);
        this.forwardLoop(z.data, target.data);
        return target;
    }

    backward(z: Matrix, a: Matrix, gradOutput: Matrix, out?: Matrix): Matrix {
        checkBackwardShapes(this.name, z, a, gradOutput);
        const target = prepareOut(out, z, `${this.name}.backward`);
        this.backwardLoop(z.data, a.data, gradOutput.data, target.data);
        return target;
    }

    getConfig(): ActivationConfig {
        return { name: this.name };
    }

    /** out[i] = f(z[i]). Must read z[i] before writing out[i] so the two may alias. */
    protected abstract forwardLoop(z: Float64Array, out: Float64Array): void;
    /** out[i] = g[i] * f'(z[i]). Must read inputs at i before writing out[i] so they may alias. */
    protected abstract backwardLoop(z: Float64Array, a: Float64Array, g: Float64Array, out: Float64Array): void;
}

class Linear extends ElementwiseActivation {
    readonly name = "linear" as const;

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        if (out !== z) out.set(z);
    }

    protected backwardLoop(_z: Float64Array, _a: Float64Array, g: Float64Array, out: Float64Array): void {
        if (out !== g) out.set(g);
    }
}

class Sigmoid extends ElementwiseActivation {
    readonly name = "sigmoid" as const;

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) out[i] = sigmoid(z[i]);
    }

    protected backwardLoop(_z: Float64Array, a: Float64Array, g: Float64Array, out: Float64Array): void {
        for (let i = 0; i < a.length; i++) {
            const s = a[i];
            out[i] = g[i] * s * (1 - s);
        }
    }
}

class Tanh extends ElementwiseActivation {
    readonly name = "tanh" as const;

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) out[i] = Math.tanh(z[i]);
    }

    protected backwardLoop(_z: Float64Array, a: Float64Array, g: Float64Array, out: Float64Array): void {
        for (let i = 0; i < a.length; i++) {
            const t = a[i];
            out[i] = g[i] * (1 - t * t);
        }
    }
}

class Relu extends ElementwiseActivation {
    readonly name = "relu" as const;

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = v > 0 ? v : 0;
        }
    }

    protected backwardLoop(z: Float64Array, _a: Float64Array, g: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) out[i] = z[i] > 0 ? g[i] : 0;
    }
}

class Relu6 extends ElementwiseActivation {
    readonly name = "relu6" as const;

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = v > 0 ? (v < 6 ? v : 6) : 0;
        }
    }

    protected backwardLoop(z: Float64Array, _a: Float64Array, g: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = v > 0 && v < 6 ? g[i] : 0;
        }
    }
}

class LeakyRelu extends ElementwiseActivation {
    readonly name = "leakyRelu" as const;

    constructor(readonly alpha: number) {
        super();
    }

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        const alpha = this.alpha;
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = v > 0 ? v : alpha * v;
        }
    }

    protected backwardLoop(z: Float64Array, _a: Float64Array, g: Float64Array, out: Float64Array): void {
        const alpha = this.alpha;
        for (let i = 0; i < z.length; i++) out[i] = z[i] > 0 ? g[i] : alpha * g[i];
    }

    override getConfig(): ActivationConfig {
        return { name: this.name, alpha: this.alpha };
    }
}

class Elu extends ElementwiseActivation {
    readonly name = "elu" as const;

    constructor(readonly alpha: number) {
        super();
    }

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        const alpha = this.alpha;
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = v > 0 ? v : alpha * Math.expm1(v);
        }
    }

    protected backwardLoop(z: Float64Array, _a: Float64Array, g: Float64Array, out: Float64Array): void {
        const alpha = this.alpha;
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = v > 0 ? g[i] : g[i] * alpha * Math.exp(v);
        }
    }

    override getConfig(): ActivationConfig {
        return { name: this.name, alpha: this.alpha };
    }
}

class Selu extends ElementwiseActivation {
    readonly name = "selu" as const;

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = v > 0 ? SELU_SCALE * v : SELU_SCALE * SELU_ALPHA * Math.expm1(v);
        }
    }

    protected backwardLoop(z: Float64Array, _a: Float64Array, g: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = v > 0 ? SELU_SCALE * g[i] : g[i] * SELU_SCALE * SELU_ALPHA * Math.exp(v);
        }
    }
}

class Gelu extends ElementwiseActivation {
    readonly name = "gelu" as const;

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = 0.5 * v * (1 + Math.tanh(GELU_C * (v + GELU_K * v * v * v)));
        }
    }

    protected backwardLoop(z: Float64Array, _a: Float64Array, g: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            const v2 = v * v;
            const t = Math.tanh(GELU_C * (v + GELU_K * v2 * v));
            const d = 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * GELU_C * (1 + 3 * GELU_K * v2);
            out[i] = g[i] * d;
        }
    }
}

class Swish extends ElementwiseActivation {
    readonly name = "swish" as const;

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = v * sigmoid(v);
        }
    }

    protected backwardLoop(z: Float64Array, _a: Float64Array, g: Float64Array, out: Float64Array): void {
        // d/dz [z·σ(z)] = σ(z) + z·σ(z)·(1 - σ(z)) = σ(z)·(1 + z·(1 - σ(z)))
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            const s = sigmoid(v);
            out[i] = g[i] * s * (1 + v * (1 - s));
        }
    }
}

class Mish extends ElementwiseActivation {
    readonly name = "mish" as const;

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = v * Math.tanh(softplus(v));
        }
    }

    protected backwardLoop(z: Float64Array, _a: Float64Array, g: Float64Array, out: Float64Array): void {
        // d/dz [z·tanh(sp(z))] = tanh(sp(z)) + z·sech²(sp(z))·σ(z)
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            const t = Math.tanh(softplus(v));
            out[i] = g[i] * (t + v * (1 - t * t) * sigmoid(v));
        }
    }
}

class Softplus extends ElementwiseActivation {
    readonly name = "softplus" as const;

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) out[i] = softplus(z[i]);
    }

    protected backwardLoop(z: Float64Array, _a: Float64Array, g: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) out[i] = g[i] * sigmoid(z[i]);
    }
}

class Softsign extends ElementwiseActivation {
    readonly name = "softsign" as const;

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = v / (1 + Math.abs(v));
        }
    }

    protected backwardLoop(z: Float64Array, _a: Float64Array, g: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const d = 1 + Math.abs(z[i]);
            out[i] = g[i] / (d * d);
        }
    }
}

class HardSigmoid extends ElementwiseActivation {
    readonly name = "hardSigmoid" as const;

    protected forwardLoop(z: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const v = 0.2 * z[i] + 0.5;
            out[i] = v > 0 ? (v < 1 ? v : 1) : 0;
        }
    }

    protected backwardLoop(z: Float64Array, _a: Float64Array, g: Float64Array, out: Float64Array): void {
        for (let i = 0; i < z.length; i++) {
            const v = z[i];
            out[i] = v > -2.5 && v < 2.5 ? 0.2 * g[i] : 0;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Softmax (row-wise)
// ---------------------------------------------------------------------------------------------

class Softmax implements Activation {
    readonly name = "softmax" as const;

    forward(z: Matrix, out?: Matrix): Matrix {
        const target = prepareOut(out, z, "softmax.forward");
        const cols = z.cols;
        const Z = z.data, A = target.data;
        for (let base = 0; base < Z.length; base += cols) {
            const end = base + cols;
            let max = -Infinity;
            for (let i = base; i < end; i++) if (Z[i] > max) max = Z[i];
            let sum = 0;
            for (let i = base; i < end; i++) {
                const e = Math.exp(Z[i] - max);
                A[i] = e;
                sum += e;
            }
            const inv = 1 / sum;
            for (let i = base; i < end; i++) A[i] *= inv;
        }
        return target;
    }

    /** Exact per-row Jacobian-vector product: dz_i = a_i · (g_i - Σ_j a_j g_j). */
    backward(z: Matrix, a: Matrix, gradOutput: Matrix, out?: Matrix): Matrix {
        checkBackwardShapes(this.name, z, a, gradOutput);
        const target = prepareOut(out, z, "softmax.backward");
        const cols = a.cols;
        const A = a.data, G = gradOutput.data, D = target.data;
        for (let base = 0; base < A.length; base += cols) {
            const end = base + cols;
            let dot = 0;
            for (let i = base; i < end; i++) dot += A[i] * G[i];
            for (let i = base; i < end; i++) D[i] = A[i] * (G[i] - dot);
        }
        return target;
    }

    getConfig(): ActivationConfig {
        return { name: this.name };
    }
}

// ---------------------------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------------------------

type ConfigParams = { readonly [param: string]: JsonValue | undefined };

/** Parameters each activation accepts in its config (besides `name`). */
const ACCEPTED_PARAMS: Record<ActivationName, readonly string[]> = {
    linear: [],
    sigmoid: [],
    tanh: [],
    relu: [],
    relu6: [],
    leakyRelu: ["alpha"],
    elu: ["alpha"],
    selu: [],
    gelu: [],
    swish: [],
    mish: [],
    softplus: [],
    softsign: [],
    hardSigmoid: [],
    softmax: [],
};

function isActivationName(value: unknown): value is ActivationName {
    return typeof value === "string" && Object.prototype.hasOwnProperty.call(ACCEPTED_PARAMS, value);
}

function unknownName(value: unknown): ValidationError {
    return new ValidationError(
        `Unknown activation ${JSON.stringify(value)}. Valid activations: ${ACTIVATION_NAMES.join(", ")}`,
    );
}

function numberParam(params: ConfigParams, key: string, fallback: number, context: string): number {
    const value = params[key];
    if (value === undefined) return fallback;
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new ValidationError(`${context}: "${key}" must be a finite number, got ${JSON.stringify(value)}`);
    }
    return value;
}

function create(name: ActivationName, params: ConfigParams): Activation {
    const accepted = ACCEPTED_PARAMS[name];
    for (const key of Object.keys(params)) {
        if (key === "name" || params[key] === undefined || accepted.includes(key)) continue;
        const hint = accepted.length > 0 ? `accepted: ${accepted.join(", ")}` : "it takes no parameters";
        throw new ValidationError(`Activation "${name}": unknown parameter "${key}" (${hint})`);
    }
    switch (name) {
        case "linear": return new Linear();
        case "sigmoid": return new Sigmoid();
        case "tanh": return new Tanh();
        case "relu": return new Relu();
        case "relu6": return new Relu6();
        case "leakyRelu": return new LeakyRelu(numberParam(params, "alpha", 0.01, "Activation \"leakyRelu\""));
        case "elu": return new Elu(numberParam(params, "alpha", 1.0, "Activation \"elu\""));
        case "selu": return new Selu();
        case "gelu": return new Gelu();
        case "swish": return new Swish();
        case "mish": return new Mish();
        case "softplus": return new Softplus();
        case "softsign": return new Softsign();
        case "hardSigmoid": return new HardSigmoid();
        case "softmax": return new Softmax();
    }
}

/**
 * Resolves an activation from a name (`"relu"`), a config (`{ name: "leakyRelu", alpha: 0.2 }`),
 * or an existing `Activation` instance (returned as-is).
 *
 * Defaults: leakyRelu `alpha` = 0.01, elu `alpha` = 1.0.
 * @throws ValidationError for unknown names, unknown parameters, or non-finite parameter values.
 */
export function getActivation(id: ActivationIdentifier): Activation {
    if (typeof id === "string") {
        if (!isActivationName(id)) throw unknownName(id);
        return create(id, {});
    }
    if (typeof id === "object" && id !== null) {
        const candidate = id as Partial<Activation>;
        if (typeof candidate.forward === "function" && typeof candidate.backward === "function") {
            return id as Activation;
        }
        const config = id as ActivationConfig;
        if (!isActivationName(config.name)) throw unknownName(config.name);
        return create(config.name, config);
    }
    throw new ValidationError(
        `Invalid activation identifier ${String(id)}: expected a name, a config object, or an Activation`,
    );
}
