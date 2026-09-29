/**
 * Batch normalization over the feature axis.
 */
import { Matrix } from "../core/matrix.js";
import type { Random } from "../core/random.js";
import type { LayerConfig, Parameter } from "../core/types.js";
import { booleanOption, checkOptions, numberOption } from "../utils.js";
import type { LayerOptions } from "./base.js";
import { BaseLayer, BufferCache, checkConfigKeys, checkLayerName, LayerParameter } from "./base.js";

/** Options for {@link BatchNormalization}. */
export interface BatchNormalizationOptions extends LayerOptions {
    /** Moving-average momentum in [0, 1): `moving = momentum·moving + (1 − momentum)·batch`. Default 0.99. */
    momentum?: number;
    /** Added to the variance before the square root (> 0). Default 1e-3. */
    epsilon?: number;
    /** Learn an offset `beta`. Default true. */
    center?: boolean;
    /** Learn a scale `gamma`. Default true. */
    scale?: boolean;
}

const OPTION_KEYS = ["momentum", "epsilon", "center", "scale", "name"] as const;

interface BatchNormBuffers {
    out: Matrix;
    /** Normalized input x̂ of the last forward. */
    xHat: Matrix;
    dx: Matrix | null;
}

/**
 * Normalizes each feature to zero mean and unit variance, then applies a learned scale and
 * offset: `y = gamma · (x − mean) / sqrt(variance + epsilon) + beta`.
 *
 * - Training (`fit`): uses the current batch's mean and (biased) variance, and updates the
 *   moving averages `movingMean` / `movingVariance`.
 * - Inference (`predict`, `evaluate`, validation): uses the moving averages.
 *
 * Parameters (all [1, features]): trainable `"<name>/gamma"` (initialized to 1) and
 * `"<name>/beta"` (0), plus the non-trainable `"<name>/movingMean"` (0) and
 * `"<name>/movingVariance"` (1). The moving statistics are saved with the model. Gamma and beta
 * are exempt from weight decay. Defaults match Keras: momentum 0.99, epsilon 1e-3.
 *
 * @example
 * const model = new Sequential({ inputSize: 10, layers: [dense(32), batchNormalization(), activation("relu"), dense(1)] });
 */
export class BatchNormalization extends BaseLayer {
    readonly type = "batchNormalization";
    readonly momentum: number;
    readonly epsilon: number;
    readonly center: boolean;
    readonly scale: boolean;

    private gamma: LayerParameter | null = null;
    private beta: LayerParameter | null = null;
    private movingMean: LayerParameter | null = null;
    private movingVariance: LayerParameter | null = null;
    /** Per-feature scratch: batch mean, then 1 / sqrt(var + eps) of the last forward. */
    private mean = new Float64Array(0);
    private invStd = new Float64Array(0);
    private sumDy = new Float64Array(0);
    private sumDyXHat = new Float64Array(0);
    private readonly cache = new BufferCache<BatchNormBuffers>((rows) => ({
        out: new Matrix(rows, this.outputSize),
        xHat: new Matrix(rows, this.outputSize),
        dx: null,
    }));
    private lastBuffers: BatchNormBuffers | null = null;
    private lastTraining = false;

    constructor(options: BatchNormalizationOptions = {}) {
        const where = "BatchNormalization";
        checkOptions(where, options, OPTION_KEYS);
        super(checkLayerName(where, options.name));
        this.momentum = numberOption(
            where,
            "momentum",
            options.momentum,
            0.99,
            "a number in [0, 1)",
            (v) => v >= 0 && v < 1,
        );
        this.epsilon = numberOption(where, "epsilon", options.epsilon, 1e-3, "a finite number > 0", (v) => v > 0);
        this.center = booleanOption(where, "center", options.center, true);
        this.scale = booleanOption(where, "scale", options.scale, true);
    }

    /** Rebuilds a BatchNormalization layer from {@link BatchNormalization.getConfig} output. */
    static fromConfig(config: LayerConfig): BatchNormalization {
        checkConfigKeys(
            "batchNormalization",
            config,
            OPTION_KEYS.filter((k) => k !== "name"),
        );
        return new BatchNormalization({
            name: config.name,
            momentum: config.momentum as number | undefined,
            epsilon: config.epsilon as number | undefined,
            center: config.center as boolean | undefined,
            scale: config.scale as boolean | undefined,
        });
    }

    protected onBuild(inputSize: number, _rng: Random): void {
        if (this.scale) {
            this.gamma = new LayerParameter(this, "gamma", 1, inputSize, { regularize: false });
            this.gamma.value.fill(1);
        }
        if (this.center) this.beta = new LayerParameter(this, "beta", 1, inputSize, { regularize: false });
        this.movingMean = new LayerParameter(this, "movingMean", 1, inputSize, { trainable: false, regularize: false });
        this.movingVariance = new LayerParameter(this, "movingVariance", 1, inputSize, {
            trainable: false,
            regularize: false,
        });
        this.movingVariance.value.fill(1);
        this.mean = new Float64Array(inputSize);
        this.invStd = new Float64Array(inputSize);
        this.sumDy = new Float64Array(inputSize);
        this.sumDyXHat = new Float64Array(inputSize);
    }

    forward(input: Matrix, training: boolean): Matrix {
        this.checkInput(input);
        const buffers = this.cache.get(input.rows);
        const rows = input.rows,
            n = input.cols;
        const X = input.data,
            Y = buffers.out.data,
            XH = buffers.xHat.data;
        const gamma = this.gamma?.value.data ?? null;
        const beta = this.beta?.value.data ?? null;
        const movingMean = (this.movingMean as LayerParameter).value.data;
        const movingVariance = (this.movingVariance as LayerParameter).value.data;
        const mean = this.mean,
            invStd = this.invStd;
        const eps = this.epsilon;

        if (training && rows > 0) {
            // Two-pass batch mean and biased variance, vectorized over features. (An empty batch
            // has no statistics: it must not turn the moving averages into NaN.)
            mean.fill(0);
            for (let r = 0; r < rows; r++) {
                const base = r * n;
                for (let j = 0; j < n; j++) mean[j] += X[base + j];
            }
            const invRows = 1 / rows;
            for (let j = 0; j < n; j++) mean[j] *= invRows;
            const variance = invStd; // scratch, converted to 1 / sqrt(var + eps) in place below
            variance.fill(0);
            for (let r = 0; r < rows; r++) {
                const base = r * n;
                for (let j = 0; j < n; j++) {
                    const d = X[base + j] - mean[j];
                    variance[j] += d * d;
                }
            }
            const m = this.momentum,
                oneMinusM = 1 - m;
            for (let j = 0; j < n; j++) {
                const v = variance[j] * invRows;
                movingMean[j] = m * movingMean[j] + oneMinusM * mean[j];
                movingVariance[j] = m * movingVariance[j] + oneMinusM * v;
                invStd[j] = 1 / Math.sqrt(v + eps);
            }
        } else {
            for (let j = 0; j < n; j++) {
                mean[j] = movingMean[j];
                invStd[j] = 1 / Math.sqrt(movingVariance[j] + eps);
            }
        }

        for (let r = 0; r < rows; r++) {
            const base = r * n;
            for (let j = 0; j < n; j++) {
                const i = base + j;
                const xh = (X[i] - mean[j]) * invStd[j];
                XH[i] = xh;
                Y[i] = (gamma === null ? xh : gamma[j] * xh) + (beta === null ? 0 : beta[j]);
            }
        }
        this.lastBuffers = buffers;
        this.lastTraining = training;
        return buffers.out;
    }

    propagate(gradOutput: Matrix, inputGradient: boolean): Matrix | null {
        const buffers = this.lastBuffers;
        this.checkGradient(gradOutput, buffers === null ? -1 : buffers.out.rows, this.outputSize);
        const b = buffers as BatchNormBuffers;
        const rows = gradOutput.rows,
            n = gradOutput.cols;
        const G = gradOutput.data,
            XH = b.xHat.data;
        const gamma = this.gamma?.value.data ?? null;
        const invStd = this.invStd;
        const sumDy = this.sumDy,
            sumDyXHat = this.sumDyXHat;

        sumDy.fill(0);
        sumDyXHat.fill(0);
        for (let r = 0; r < rows; r++) {
            const base = r * n;
            for (let j = 0; j < n; j++) {
                const g = G[base + j];
                sumDy[j] += g;
                sumDyXHat[j] += g * XH[base + j];
            }
        }
        if (this.gamma !== null) this.gamma.grad.data.set(sumDyXHat);
        if (this.beta !== null) this.beta.grad.data.set(sumDy);
        if (!inputGradient) return null;

        b.dx ??= new Matrix(rows, n);
        const D = b.dx.data;
        if (!this.lastTraining) {
            // Inference mode: the statistics are constants, so y is affine in x.
            for (let r = 0; r < rows; r++) {
                const base = r * n;
                for (let j = 0; j < n; j++) D[base + j] = G[base + j] * (gamma === null ? 1 : gamma[j]) * invStd[j];
            }
            return b.dx;
        }
        // Training mode (batch statistics): dx = gamma·invStd/N · (N·dy − Σdy − x̂·Σ(dy·x̂))
        const invRows = 1 / rows;
        for (let r = 0; r < rows; r++) {
            const base = r * n;
            for (let j = 0; j < n; j++) {
                const i = base + j;
                const k = (gamma === null ? 1 : gamma[j]) * invStd[j];
                D[i] = k * (G[i] - invRows * (sumDy[j] + XH[i] * sumDyXHat[j]));
            }
        }
        return b.dx;
    }

    override parameters(): Parameter[] {
        const params: Parameter[] = [];
        if (this.gamma !== null) params.push(this.gamma);
        if (this.beta !== null) params.push(this.beta);
        if (this.movingMean !== null) params.push(this.movingMean, this.movingVariance as LayerParameter);
        return params;
    }

    getConfig(): LayerConfig {
        return {
            type: this.type,
            name: this.name,
            momentum: this.momentum,
            epsilon: this.epsilon,
            center: this.center,
            scale: this.scale,
        };
    }
}
