/**
 * The playground's only point of contact with the engine: builds, trains, evaluates and
 * serializes a `Sequential` model from a {@link PlaygroundConfig}.
 */
import { dense, dropout, gatherRows, Matrix, Sequential, serializeModel, VERSION } from "../../src/index.js";
import type { ActivationId, OutputSpec, PlaygroundConfig } from "./config.js";
import { outputSpec, taskKind } from "./config.js";
import { Rng } from "./datasets.js";
import type { Kernel } from "./network-diagram.js";

export { Matrix };

/** Version of the engine bundled into the playground. */
export const ENGINE_VERSION: string = VERSION;

export interface Evaluation {
    loss: number;
}

export class Trainer {
    readonly outputs: number;
    private readonly model: Sequential;
    /** Sample order and position of the epoch in progress when training batch by batch. */
    private order: number[] = [];
    private cursor = 0;
    private readonly rng: Rng;

    constructor(config: PlaygroundConfig) {
        const output = outputSpec(config);
        const l2 = config.l2;
        const denseLayer = (units: number, activation: ActivationId | OutputSpec["activation"]) =>
            l2 > 0 ? dense(units, { activation, kernelRegularizer: { l2 } }) : dense(units, activation);

        const layers = [];
        for (const layer of config.layers) {
            layers.push(denseLayer(layer.units, layer.activation));
            if (layer.dropout > 0) layers.push(dropout(layer.dropout));
        }
        layers.push(denseLayer(output.units, output.activation));

        this.outputs = output.units;
        this.rng = new Rng(config.seed ^ 0x2545f491);
        this.model = new Sequential({ inputSize: config.features.length, seed: config.seed, layers });
        this.model.compile({
            loss: output.loss,
            optimizer: { name: config.optimizer, learningRate: config.learningRate },
            metrics: taskKind(config) === "regression" ? [] : ["accuracy"],
        });
    }

    /** Runs one epoch over (x, y) and returns the epoch's mean training loss. */
    fitEpoch(x: Matrix, y: Matrix, batchSize: number): number {
        const history = this.model.fit(x, y, { epochs: 1, batchSize, shuffle: true });
        return history.last("loss") ?? Number.NaN;
    }

    /**
     * Time-sliced training for networks whose epochs do not fit in a frame: runs shuffled
     * mini-batches with `trainOnBatch` until `deadline` (a `performance.now()` timestamp) or the end
     * of the epoch, and resumes where it stopped on the next call. Returns true when an epoch completed.
     */
    trainUntil(x: Matrix, y: Matrix, batchSize: number, deadline: number): boolean {
        const n = x.rows;
        if (this.order.length !== n) {
            this.order = Array.from({ length: n }, (_, i) => i);
            this.cursor = 0;
        }
        if (this.cursor === 0) this.rng.shuffle(this.order);
        while (this.cursor < n) {
            const end = Math.min(this.cursor + batchSize, n);
            const batch = this.order.slice(this.cursor, end);
            this.model.trainOnBatch(gatherRows(x, batch), gatherRows(y, batch));
            this.cursor = end;
            if (performance.now() >= deadline) break;
        }
        if (this.cursor < n) return false;
        this.cursor = 0;
        return true;
    }

    evaluate(x: Matrix, y: Matrix): Evaluation {
        if (x.rows === 0) return { loss: Number.NaN };
        return { loss: this.model.evaluate(x, y).loss };
    }

    /** Batched inference: [n, features] → [n, outputs]. */
    predict(x: Matrix): Matrix {
        return this.model.predict(x);
    }

    /** Dense kernels ([inputs, units]) in layer order, for the network diagram. */
    kernels(): Kernel[] {
        return this.model
            .getWeights()
            .filter((w) => w.name.endsWith("/kernel"))
            .map((w) => ({ shape: w.shape, data: w.data }));
    }

    toBinary(): Uint8Array {
        return serializeModel(this.model, { format: "binary" });
    }

    toJson(): string {
        return serializeModel(this.model, { format: "json", pretty: true });
    }
}
