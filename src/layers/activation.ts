/**
 * Standalone activation layer.
 */
import { getActivation } from "../activations.js";
import { Matrix } from "../core/matrix.js";
import type { Random } from "../core/random.js";
import type { Activation, ActivationIdentifier, LayerConfig } from "../core/types.js";
import { checkOptions } from "../utils.js";
import type { LayerOptions } from "./base.js";
import { BaseLayer, BufferCache, checkConfigKeys, checkLayerName } from "./base.js";

/** Options for {@link ActivationLayer}. */
export interface ActivationLayerOptions extends LayerOptions {
    /** The activation to apply, e.g. "relu" or `{ name: "leakyRelu", alpha: 0.2 }`. */
    activation: ActivationIdentifier;
}

interface ActivationBuffers {
    out: Matrix;
    dx: Matrix | null;
}

/**
 * Applies an activation function element-wise (or row-wise for softmax) with no parameters.
 * Useful after {@link BatchNormalization}, which normally sits between a linear Dense layer and
 * its activation.
 *
 * @example
 * model.add(dense(64)).add(batchNormalization()).add(activation("relu"));
 */
export class ActivationLayer extends BaseLayer {
    readonly type = "activation";
    readonly activation: Activation;
    private readonly cache = new BufferCache<ActivationBuffers>((rows) => ({
        out: new Matrix(rows, this.outputSize),
        dx: null,
    }));
    private lastInput: Matrix | null = null;
    private lastBuffers: ActivationBuffers | null = null;

    constructor(options: ActivationLayerOptions) {
        const where = "ActivationLayer";
        checkOptions(where, options, ["activation", "name"]);
        super(checkLayerName(where, options?.name));
        this.activation = getActivation(options?.activation);
    }

    /** Rebuilds an activation layer from {@link ActivationLayer.getConfig} output. */
    static fromConfig(config: LayerConfig): ActivationLayer {
        checkConfigKeys("activation", config, ["activation"]);
        return new ActivationLayer({ name: config.name, activation: config.activation as ActivationIdentifier });
    }

    protected onBuild(_inputSize: number, _rng: Random): void {}

    forward(input: Matrix, _training: boolean): Matrix {
        this.checkInput(input);
        const buffers = this.cache.get(input.rows);
        this.activation.forward(input, buffers.out);
        this.lastInput = input;
        this.lastBuffers = buffers;
        return buffers.out;
    }

    propagate(gradOutput: Matrix, inputGradient: boolean): Matrix | null {
        const buffers = this.lastBuffers;
        this.checkGradient(gradOutput, buffers === null ? -1 : buffers.out.rows, this.outputSize);
        if (!inputGradient) return null;
        const b = buffers as ActivationBuffers;
        b.dx ??= new Matrix(gradOutput.rows, gradOutput.cols);
        return this.activation.backward(this.lastInput as Matrix, b.out, gradOutput, b.dx);
    }

    getConfig(): LayerConfig {
        return { type: this.type, name: this.name, activation: this.activation.getConfig() };
    }
}
