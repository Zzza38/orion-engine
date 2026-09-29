/**
 * Fully connected layer: `a = activation(x · kernel + bias)`.
 */
import { getActivation } from "../activations.js";
import { ValidationError } from "../core/errors.js";
import { Matrix } from "../core/matrix.js";
import type { Random } from "../core/random.js";
import type {
    Activation,
    ActivationIdentifier,
    Initializer,
    InitializerIdentifier,
    JsonValue,
    LayerConfig,
    Parameter,
} from "../core/types.js";
import { getInitializer } from "../initializers.js";
import { booleanOption, checkOptions, numberOption, positiveInteger } from "../utils.js";
import type { LayerOptions } from "./base.js";
import { BaseLayer, BufferCache, checkConfigKeys, checkLayerName, LayerParameter } from "./base.js";

/** Weight penalty added to the loss: `l1·Σ|w| + l2·Σw²` (Keras convention). */
export interface RegularizerOptions {
    /** L1 factor ≥ 0. Default 0. */
    l1?: number;
    /** L2 factor ≥ 0. Default 0. */
    l2?: number;
}

/** Options for {@link Dense}. */
export interface DenseOptions extends LayerOptions {
    /** Number of output units (positive integer). */
    units: number;
    /** Activation applied to `x · kernel + bias`. Default "linear". */
    activation?: ActivationIdentifier;
    /** Add a bias vector. Default true. */
    useBias?: boolean;
    /**
     * Kernel initializer. Default "glorotUniform" (as in Keras). For deep ReLU-family networks
     * "heNormal" or "heUniform" usually trains faster.
     */
    kernelInitializer?: InitializerIdentifier;
    /** Bias initializer. Default "zeros". */
    biasInitializer?: InitializerIdentifier;
    /** L1/L2 penalty on the kernel, added to the training loss and its gradient. Default none. */
    kernelRegularizer?: RegularizerOptions | null;
}

const OPTION_KEYS = [
    "units",
    "activation",
    "useBias",
    "kernelInitializer",
    "biasInitializer",
    "kernelRegularizer",
    "name",
] as const;

interface DenseBuffers {
    /** Pre-activation x · kernel + bias. */
    z: Matrix;
    /** Activation output (the same object as `z` for linear layers). */
    a: Matrix;
    /** dL/dz, allocated on first backward. */
    dz: Matrix | null;
    /** dL/dx, allocated on first backward that needs it. */
    dx: Matrix | null;
}

/**
 * Fully connected layer computing `activation(x · kernel + bias)` for a batch `x` of shape
 * [batch, inputSize]. The kernel is [inputSize, units] and the bias [1, units], stored as the
 * parameters `"<name>/kernel"` and `"<name>/bias"`.
 *
 * @example
 * const layer = new Dense({ units: 16, activation: "relu", kernelRegularizer: { l2: 1e-4 } });
 * // or, with the factory helper:
 * const same = dense(16, { activation: "relu", kernelRegularizer: { l2: 1e-4 } });
 */
export class Dense extends BaseLayer {
    readonly type = "dense";
    readonly units: number;
    readonly activation: Activation;
    readonly useBias: boolean;
    readonly kernelInitializer: Initializer;
    readonly biasInitializer: Initializer;
    /** Normalized kernel penalty, or null when the layer is not regularized. */
    readonly kernelRegularizer: Readonly<Required<RegularizerOptions>> | null;

    private kernelParam: LayerParameter | null = null;
    private biasParam: LayerParameter | null = null;
    private readonly isLinear: boolean;
    private readonly cache = new BufferCache<DenseBuffers>((rows) => this.createBuffers(rows));
    private lastInput: Matrix | null = null;
    private lastBuffers: DenseBuffers | null = null;

    constructor(options: DenseOptions) {
        const where = "Dense";
        checkOptions(where, options, OPTION_KEYS);
        if (options === undefined)
            throw new ValidationError(`${where}: options with "units" are required, e.g. new Dense({ units: 8 })`);
        super(checkLayerName(where, options.name));
        this.units = positiveInteger(where, "units", options.units);
        this.activation = getActivation(options.activation ?? "linear");
        this.useBias = booleanOption(where, "useBias", options.useBias, true);
        this.kernelInitializer = getInitializer(options.kernelInitializer ?? "glorotUniform");
        this.biasInitializer = getInitializer(options.biasInitializer ?? "zeros");
        this.kernelRegularizer = normalizeRegularizer(where, options.kernelRegularizer);
        this.isLinear = this.activation.name === "linear";
    }

    /** Rebuilds a Dense layer from {@link Dense.getConfig} output. */
    static fromConfig(config: LayerConfig): Dense {
        checkConfigKeys(
            "dense",
            config,
            OPTION_KEYS.filter((k) => k !== "name"),
        );
        return new Dense({
            name: config.name,
            units: config.units as number,
            activation: config.activation as ActivationIdentifier | undefined,
            useBias: config.useBias as boolean | undefined,
            kernelInitializer: config.kernelInitializer as InitializerIdentifier | undefined,
            biasInitializer: config.biasInitializer as InitializerIdentifier | undefined,
            kernelRegularizer: config.kernelRegularizer as RegularizerOptions | null | undefined,
        });
    }

    override get outputSize(): number {
        return this.units;
    }

    /** The [inputSize, units] kernel parameter (throws before `build`). */
    get kernel(): Parameter {
        if (this.kernelParam === null) throw new ValidationError(`${this.label} has no kernel yet: build it first`);
        return this.kernelParam;
    }

    /** The [1, units] bias parameter, or null when `useBias` is false (throws before `build`). */
    get bias(): Parameter | null {
        if (this.kernelParam === null) throw new ValidationError(`${this.label} has no bias yet: build it first`);
        return this.biasParam;
    }

    protected onBuild(inputSize: number, rng: Random): void {
        const kernel = new LayerParameter(this, "kernel", inputSize, this.units);
        this.kernelInitializer.initialize(kernel.value, inputSize, this.units, rng);
        this.kernelParam = kernel;
        if (this.useBias) {
            const bias = new LayerParameter(this, "bias", 1, this.units, { regularize: false });
            this.biasInitializer.initialize(bias.value, inputSize, this.units, rng);
            this.biasParam = bias;
        }
    }

    forward(input: Matrix, _training: boolean): Matrix {
        this.checkInput(input);
        const buffers = this.cache.get(input.rows);
        denseForward(
            input,
            (this.kernelParam as LayerParameter).value.data,
            this.biasParam?.value.data ?? null,
            buffers.z,
        );
        if (!this.isLinear) this.activation.forward(buffers.z, buffers.a);
        this.lastInput = input;
        this.lastBuffers = buffers;
        return buffers.a;
    }

    propagate(gradOutput: Matrix, inputGradient: boolean): Matrix | null {
        const buffers = this.lastBuffers;
        this.checkGradient(gradOutput, buffers === null ? -1 : buffers.a.rows, this.units);
        const b = buffers as DenseBuffers;
        let dz = gradOutput;
        if (!this.isLinear) {
            b.dz ??= new Matrix(b.z.rows, this.units);
            dz = this.activation.backward(b.z, b.a, gradOutput, b.dz);
        }
        return this.propagateFromPreActivation(dz, inputGradient);
    }

    /**
     * Backward pass starting from dL/dz (the gradient with respect to the pre-activation), used
     * by models when the loss provides a fused activation+loss gradient (sigmoid + binary
     * cross-entropy, softmax + categorical cross-entropy). Writes the parameter gradients
     * (including the regularization term) and returns dL/dInput, or null when `inputGradient`
     * is false.
     */
    propagateFromPreActivation(dz: Matrix, inputGradient: boolean): Matrix | null {
        const buffers = this.lastBuffers;
        this.checkGradient(dz, buffers === null ? -1 : buffers.a.rows, this.units);
        const b = buffers as DenseBuffers;
        const input = this.lastInput as Matrix;
        const kernel = this.kernelParam as LayerParameter;

        kernelGradient(input, dz, kernel.grad.data);
        if (this.biasParam !== null) columnSums(dz, this.biasParam.grad.data);
        const reg = this.kernelRegularizer;
        if (reg !== null) addRegularizationGradient(kernel.value.data, kernel.grad.data, reg.l1, reg.l2);

        if (!inputGradient) return null;
        b.dx ??= new Matrix(input.rows, input.cols);
        inputGradientKernel(dz, kernel.value.data, b.dx);
        return b.dx;
    }

    /** `l1·Σ|w| + l2·Σw²` over the kernel (0 when unregularized or unbuilt). */
    regularizationLoss(): number {
        const reg = this.kernelRegularizer;
        if (reg === null || this.kernelParam === null) return 0;
        const w = this.kernelParam.value.data;
        let abs = 0;
        let sq = 0;
        for (let i = 0; i < w.length; i++) {
            const v = w[i];
            abs += v < 0 ? -v : v;
            sq += v * v;
        }
        return reg.l1 * abs + reg.l2 * sq;
    }

    override parameters(): Parameter[] {
        if (this.kernelParam === null) return [];
        return this.biasParam === null ? [this.kernelParam] : [this.kernelParam, this.biasParam];
    }

    getConfig(): LayerConfig {
        const config: LayerConfig = {
            type: this.type,
            name: this.name,
            units: this.units,
            activation: this.activation.getConfig(),
            useBias: this.useBias,
            kernelInitializer: this.kernelInitializer.getConfig(),
            biasInitializer: this.biasInitializer.getConfig(),
        };
        if (this.kernelRegularizer !== null) {
            config.kernelRegularizer = { l1: this.kernelRegularizer.l1, l2: this.kernelRegularizer.l2 };
        }
        return config;
    }

    private createBuffers(rows: number): DenseBuffers {
        const z = new Matrix(rows, this.units);
        return { z, a: this.isLinear ? z : new Matrix(rows, this.units), dz: null, dx: null };
    }
}

function normalizeRegularizer(where: string, value: unknown): Readonly<Required<RegularizerOptions>> | null {
    if (value === undefined || value === null) return null;
    checkOptions(`${where} kernelRegularizer`, value, ["l1", "l2"]);
    const options = value as { [key: string]: JsonValue | undefined };
    const nonNegative = (v: number) => v >= 0;
    const l1 = numberOption(`${where} kernelRegularizer`, "l1", options.l1, 0, "a finite number >= 0", nonNegative);
    const l2 = numberOption(`${where} kernelRegularizer`, "l2", options.l2, 0, "a finite number >= 0", nonNegative);
    if (l1 === 0 && l2 === 0) return null;
    return Object.freeze({ l1, l2 });
}

// ---------------------------------------------------------------------------------------------
// Kernels: plain loops over typed arrays, register-blocked (2 rows × 4 columns of independent
// accumulators) so V8 keeps the partial sums in registers; about 3× faster than a naive
// i-p-j loop. Every output of the forward kernel still accumulates x_p·w_p in increasing p from
// 0 and adds the bias last, exactly like the legacy engine, so imported models match bit for bit
// before the activation.
// ---------------------------------------------------------------------------------------------

/** z = x · W + b, with W [k, n] row-major and z [m, n]. Overwrites z. */
function denseForward(x: Matrix, W: Float64Array, bias: Float64Array | null, z: Matrix): void {
    const m = x.rows,
        k = x.cols,
        n = z.cols;
    const X = x.data,
        Z = z.data;
    let i = 0;
    for (; i + 1 < m; i += 2) {
        const x0 = i * k,
            x1 = x0 + k,
            z0 = i * n,
            z1 = z0 + n;
        let j = 0;
        for (; j + 3 < n; j += 4) {
            let s00 = 0,
                s01 = 0,
                s02 = 0,
                s03 = 0,
                s10 = 0,
                s11 = 0,
                s12 = 0,
                s13 = 0;
            for (let p = 0, w = j; p < k; p++, w += n) {
                const a0 = X[x0 + p],
                    a1 = X[x1 + p];
                const w0 = W[w],
                    w1 = W[w + 1],
                    w2 = W[w + 2],
                    w3 = W[w + 3];
                s00 += a0 * w0;
                s01 += a0 * w1;
                s02 += a0 * w2;
                s03 += a0 * w3;
                s10 += a1 * w0;
                s11 += a1 * w1;
                s12 += a1 * w2;
                s13 += a1 * w3;
            }
            Z[z0 + j] = s00;
            Z[z0 + j + 1] = s01;
            Z[z0 + j + 2] = s02;
            Z[z0 + j + 3] = s03;
            Z[z1 + j] = s10;
            Z[z1 + j + 1] = s11;
            Z[z1 + j + 2] = s12;
            Z[z1 + j + 3] = s13;
        }
        for (; j < n; j++) {
            let s0 = 0,
                s1 = 0;
            for (let p = 0, w = j; p < k; p++, w += n) {
                const wv = W[w];
                s0 += X[x0 + p] * wv;
                s1 += X[x1 + p] * wv;
            }
            Z[z0 + j] = s0;
            Z[z1 + j] = s1;
        }
    }
    if (i < m) forwardRow(X, W, Z, i * k, i * n, k, n);
    if (bias !== null) {
        for (let r = 0; r < m; r++) {
            const row = r * n;
            for (let j = 0; j < n; j++) Z[row + j] += bias[j];
        }
    }
}

/** One row of z = x · W (the last odd row, or single-sample prediction): 8 independent accumulators. */
function forwardRow(
    X: Float64Array,
    W: Float64Array,
    Z: Float64Array,
    x0: number,
    z0: number,
    k: number,
    n: number,
): void {
    let j = 0;
    for (; j + 7 < n; j += 8) {
        let s0 = 0,
            s1 = 0,
            s2 = 0,
            s3 = 0,
            s4 = 0,
            s5 = 0,
            s6 = 0,
            s7 = 0;
        for (let p = 0, w = j; p < k; p++, w += n) {
            const a = X[x0 + p];
            s0 += a * W[w];
            s1 += a * W[w + 1];
            s2 += a * W[w + 2];
            s3 += a * W[w + 3];
            s4 += a * W[w + 4];
            s5 += a * W[w + 5];
            s6 += a * W[w + 6];
            s7 += a * W[w + 7];
        }
        Z[z0 + j] = s0;
        Z[z0 + j + 1] = s1;
        Z[z0 + j + 2] = s2;
        Z[z0 + j + 3] = s3;
        Z[z0 + j + 4] = s4;
        Z[z0 + j + 5] = s5;
        Z[z0 + j + 6] = s6;
        Z[z0 + j + 7] = s7;
    }
    for (; j < n; j++) {
        let s = 0;
        for (let p = 0, w = j; p < k; p++, w += n) s += X[x0 + p] * W[w];
        Z[z0 + j] = s;
    }
}

/** dW = xᵀ · dz, with x [m, k], dz [m, n], dW [k, n]. Overwrites dW. */
function kernelGradient(x: Matrix, dz: Matrix, dW: Float64Array): void {
    const m = x.rows,
        k = x.cols,
        n = dz.cols;
    const X = x.data,
        D = dz.data;
    let p = 0;
    for (; p + 1 < k; p += 2) {
        const w0 = p * n,
            w1 = w0 + n;
        let j = 0;
        for (; j + 3 < n; j += 4) {
            let s00 = 0,
                s01 = 0,
                s02 = 0,
                s03 = 0,
                s10 = 0,
                s11 = 0,
                s12 = 0,
                s13 = 0;
            for (let r = 0, xi = p, di = j; r < m; r++, xi += k, di += n) {
                const a0 = X[xi],
                    a1 = X[xi + 1];
                const d0 = D[di],
                    d1 = D[di + 1],
                    d2 = D[di + 2],
                    d3 = D[di + 3];
                s00 += a0 * d0;
                s01 += a0 * d1;
                s02 += a0 * d2;
                s03 += a0 * d3;
                s10 += a1 * d0;
                s11 += a1 * d1;
                s12 += a1 * d2;
                s13 += a1 * d3;
            }
            dW[w0 + j] = s00;
            dW[w0 + j + 1] = s01;
            dW[w0 + j + 2] = s02;
            dW[w0 + j + 3] = s03;
            dW[w1 + j] = s10;
            dW[w1 + j + 1] = s11;
            dW[w1 + j + 2] = s12;
            dW[w1 + j + 3] = s13;
        }
        for (; j < n; j++) {
            let s0 = 0,
                s1 = 0;
            for (let r = 0, xi = p, di = j; r < m; r++, xi += k, di += n) {
                const d = D[di];
                s0 += X[xi] * d;
                s1 += X[xi + 1] * d;
            }
            dW[w0 + j] = s0;
            dW[w1 + j] = s1;
        }
    }
    if (p < k) {
        const w0 = p * n;
        for (let j = 0; j < n; j++) {
            let s = 0;
            for (let r = 0, xi = p, di = j; r < m; r++, xi += k, di += n) s += X[xi] * D[di];
            dW[w0 + j] = s;
        }
    }
}

/** db = Σ_rows dz. Overwrites db. */
function columnSums(dz: Matrix, db: Float64Array): void {
    const m = dz.rows,
        n = dz.cols,
        D = dz.data;
    db.fill(0);
    for (let r = 0; r < m; r++) {
        const row = r * n;
        for (let j = 0; j < n; j++) db[j] += D[row + j];
    }
}

/** dx = dz · Wᵀ, with dz [m, n], W [k, n], dx [m, k]. Overwrites dx. */
function inputGradientKernel(dz: Matrix, W: Float64Array, dx: Matrix): void {
    const m = dz.rows,
        n = dz.cols,
        k = dx.cols;
    const D = dz.data,
        X = dx.data;
    let i = 0;
    for (; i + 1 < m; i += 2) {
        const d0 = i * n,
            d1 = d0 + n,
            x0 = i * k,
            x1 = x0 + k;
        let p = 0;
        for (; p + 3 < k; p += 4) {
            const w0 = p * n,
                w1 = w0 + n,
                w2 = w1 + n,
                w3 = w2 + n;
            let s00 = 0,
                s01 = 0,
                s02 = 0,
                s03 = 0,
                s10 = 0,
                s11 = 0,
                s12 = 0,
                s13 = 0;
            for (let j = 0; j < n; j++) {
                const a0 = D[d0 + j],
                    a1 = D[d1 + j];
                const b0 = W[w0 + j],
                    b1 = W[w1 + j],
                    b2 = W[w2 + j],
                    b3 = W[w3 + j];
                s00 += a0 * b0;
                s01 += a0 * b1;
                s02 += a0 * b2;
                s03 += a0 * b3;
                s10 += a1 * b0;
                s11 += a1 * b1;
                s12 += a1 * b2;
                s13 += a1 * b3;
            }
            X[x0 + p] = s00;
            X[x0 + p + 1] = s01;
            X[x0 + p + 2] = s02;
            X[x0 + p + 3] = s03;
            X[x1 + p] = s10;
            X[x1 + p + 1] = s11;
            X[x1 + p + 2] = s12;
            X[x1 + p + 3] = s13;
        }
        for (; p < k; p++) {
            const w0 = p * n;
            let s0 = 0,
                s1 = 0;
            for (let j = 0; j < n; j++) {
                const b = W[w0 + j];
                s0 += D[d0 + j] * b;
                s1 += D[d1 + j] * b;
            }
            X[x0 + p] = s0;
            X[x1 + p] = s1;
        }
    }
    if (i < m) {
        const d0 = i * n,
            x0 = i * k;
        let p = 0;
        for (; p + 3 < k; p += 4) {
            const w0 = p * n,
                w1 = w0 + n,
                w2 = w1 + n,
                w3 = w2 + n;
            let s0 = 0,
                s1 = 0,
                s2 = 0,
                s3 = 0;
            for (let j = 0; j < n; j++) {
                const a = D[d0 + j];
                s0 += a * W[w0 + j];
                s1 += a * W[w1 + j];
                s2 += a * W[w2 + j];
                s3 += a * W[w3 + j];
            }
            X[x0 + p] = s0;
            X[x0 + p + 1] = s1;
            X[x0 + p + 2] = s2;
            X[x0 + p + 3] = s3;
        }
        for (; p < k; p++) {
            const w0 = p * n;
            let s = 0;
            for (let j = 0; j < n; j++) s += D[d0 + j] * W[w0 + j];
            X[x0 + p] = s;
        }
    }
}

/** grad += l1·sign(w) + 2·l2·w. */
function addRegularizationGradient(w: Float64Array, grad: Float64Array, l1: number, l2: number): void {
    const twoL2 = 2 * l2;
    for (let i = 0; i < w.length; i++) {
        const v = w[i];
        grad[i] += (v > 0 ? l1 : v < 0 ? -l1 : 0) + twoL2 * v;
    }
}
