/**
 * Inverted dropout.
 */
import { ValidationError } from "../core/errors.js";
import { Matrix } from "../core/matrix.js";
import type { Random } from "../core/random.js";
import type { LayerConfig } from "../core/types.js";
import { checkOptions, numberOption } from "../utils.js";
import type { LayerOptions } from "./base.js";
import { BaseLayer, BufferCache, checkConfigKeys, checkLayerName } from "./base.js";

/** Options for {@link Dropout}. */
export interface DropoutOptions extends LayerOptions {
    /** Fraction of inputs to drop during training, in [0, 1). */
    rate: number;
}

interface DropoutBuffers {
    out: Matrix;
    /** Per-element multiplier: 0 (dropped) or 1 / (1 - rate) (kept). */
    mask: Float64Array;
    dx: Matrix | null;
}

/**
 * Inverted dropout: during training each input is zeroed with probability `rate` and the
 * survivors are scaled by `1 / (1 - rate)`, so the expected activation is unchanged and
 * inference is the identity. The mask is drawn from the layer's own seeded `Random` (forked
 * from the model's at build time), so training stays reproducible.
 *
 * @example
 * const model = new Sequential({ inputSize: 20, layers: [dense(64, "relu"), dropout(0.3), dense(1, "sigmoid")] });
 */
export class Dropout extends BaseLayer {
    readonly type = "dropout";
    readonly rate: number;
    private rng: Random | null = null;
    private readonly cache = new BufferCache<DropoutBuffers>((rows) => ({
        out: new Matrix(rows, this.outputSize),
        mask: new Float64Array(rows * this.outputSize),
        dx: null,
    }));
    /** Buffers of the last training-mode forward, or null when it was the identity. */
    private lastBuffers: DropoutBuffers | null = null;
    private lastRows = -1;

    constructor(options: DropoutOptions) {
        const where = "Dropout";
        checkOptions(where, options, ["rate", "name"]);
        super(checkLayerName(where, options?.name));
        if (options?.rate === undefined) {
            throw new ValidationError(`${where}: "rate" is required: a number in [0, 1), e.g. dropout(0.2)`);
        }
        this.rate = numberOption(where, "rate", options.rate, 0, "a number in [0, 1)", (v) => v >= 0 && v < 1);
    }

    /** Rebuilds a Dropout layer from {@link Dropout.getConfig} output. */
    static fromConfig(config: LayerConfig): Dropout {
        checkConfigKeys("dropout", config, ["rate"]);
        return new Dropout({ name: config.name, rate: config.rate as number });
    }

    protected onBuild(_inputSize: number, rng: Random): void {
        this.rng = rng;
    }

    forward(input: Matrix, training: boolean): Matrix {
        this.checkInput(input);
        this.lastRows = input.rows;
        if (!training || this.rate === 0) {
            this.lastBuffers = null;
            return input;
        }
        const buffers = this.cache.get(input.rows);
        const rng = this.rng as Random;
        const rate = this.rate;
        const keepScale = 1 / (1 - rate);
        const X = input.data,
            Y = buffers.out.data,
            mask = buffers.mask;
        for (let i = 0; i < X.length; i++) {
            const m = rng.next() < rate ? 0 : keepScale;
            mask[i] = m;
            Y[i] = X[i] * m;
        }
        this.lastBuffers = buffers;
        return buffers.out;
    }

    propagate(gradOutput: Matrix, inputGradient: boolean): Matrix | null {
        this.checkGradient(gradOutput, this.lastRows, this.outputSize);
        if (!inputGradient) return null;
        const buffers = this.lastBuffers;
        if (buffers === null) return gradOutput;
        buffers.dx ??= new Matrix(gradOutput.rows, gradOutput.cols);
        const G = gradOutput.data,
            D = buffers.dx.data,
            mask = buffers.mask;
        for (let i = 0; i < G.length; i++) D[i] = G[i] * mask[i];
        return buffers.dx;
    }

    getConfig(): LayerConfig {
        return { type: this.type, name: this.name, rate: this.rate };
    }
}
