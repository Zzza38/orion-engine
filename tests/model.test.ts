import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { Callback, Logs } from "../src/callbacks.js";
import { earlyStopping } from "../src/callbacks.js";
import { ShapeError, TrainingError, ValidationError } from "../src/core/errors.js";
import { Matrix } from "../src/core/matrix.js";
import { Random } from "../src/core/random.js";
import type { ActivationName, Loss, LossIdentifier, WeightEntry } from "../src/core/types.js";
import { activation, batchNormalization, dense, dropout } from "../src/layers/index.js";
import { getLoss } from "../src/losses.js";
import { Sequential } from "../src/model.js";

const XOR_X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];
const XOR_Y = [0, 1, 1, 0];

function xorModel(seed = 42): Sequential {
    const model = new Sequential({ inputSize: 2, seed, name: "xor", layers: [dense(8, "tanh"), dense(1, "sigmoid")] });
    model.compile({ loss: "bce", optimizer: { name: "adam", learningRate: 0.05 }, metrics: ["accuracy"] });
    return model;
}

function spiral(perClass: number, classes: number, rng: Random): { x: number[][]; y: number[] } {
    const x: number[][] = [];
    const y: number[] = [];
    for (let c = 0; c < classes; c++) {
        for (let i = 0; i < perClass; i++) {
            const r = i / perClass;
            const t = c * 4 + (i / perClass) * 4 + rng.normal() * 0.2;
            x.push([r * Math.sin(t), r * Math.cos(t)]);
            y.push(c);
        }
    }
    return { x, y };
}

/** Random regression-ish data. */
function randomData(rng: Random, rows: number, inputs: number, outputs: number): { x: number[][]; y: number[][] } {
    const x = Array.from({ length: rows }, () => Array.from({ length: inputs }, () => rng.uniform(-1, 1)));
    const y = x.map((row) =>
        Array.from({ length: outputs }, (_, j) => Math.sin(row[0] * (j + 1)) + row[inputs - 1] * 0.5),
    );
    return { x, y };
}

function withoutDuration(history: Record<string, number[]>): Record<string, number[]> {
    const { durationMs: _ignored, ...rest } = history;
    return rest;
}

function weightsOf(model: Sequential): number[][] {
    return model.getWeights().map((w) => Array.from(w.data));
}

describe("Sequential: training", () => {
    it("learns XOR deterministically with a fixed seed in well under 2 s", () => {
        const start = performance.now();
        const model = xorModel();
        const history = model.fit(XOR_X, XOR_Y, { epochs: 300, batchSize: 4 });
        const elapsed = performance.now() - start;
        const predictions = model.predict(XOR_X).map((row) => row[0]);
        predictions.forEach((p, i) => {
            assert.ok(Math.abs(p - XOR_Y[i]) < 0.1, `XOR ${XOR_X[i]} -> ${p}`);
        });
        assert.equal(history.last("accuracy"), 1);
        assert.ok(elapsed < 2000, `took ${elapsed} ms`);
        // Same seed, same result: bit for bit.
        const again = xorModel();
        again.fit(XOR_X, XOR_Y, { epochs: 300, batchSize: 4 });
        assert.deepEqual(again.predict(XOR_X), model.predict(XOR_X));
    });

    it("classifies a 3-class spiral with softmax + sparse categorical cross-entropy (> 90%)", () => {
        const { x, y } = spiral(100, 3, new Random(3));
        const model = new Sequential({
            inputSize: 2,
            seed: 7,
            layers: [dense(32, "relu"), dense(32, "relu"), dense(3, "softmax")],
        });
        model.compile({ loss: "scce", optimizer: { name: "adam", learningRate: 0.02 }, metrics: ["accuracy"] });
        model.fit(x, y, { epochs: 150, batchSize: 32 });
        const { accuracy } = model.evaluate(x, y);
        assert.ok(accuracy > 0.9, `accuracy ${accuracy}`);
    });

    it("fits a sine wave (regression, mse)", () => {
        const x: number[][] = [];
        const y: number[] = [];
        for (let i = 0; i < 200; i++) {
            const v = -Math.PI + (2 * Math.PI * i) / 199;
            x.push([v]);
            y.push(Math.sin(v));
        }
        const model = new Sequential({
            inputSize: 1,
            seed: 1,
            layers: [dense(32, "tanh"), dense(32, "tanh"), dense(1)],
        });
        model.compile({ loss: "mse", optimizer: { name: "adam", learningRate: 0.01 }, metrics: ["mae", "rmse"] });
        const history = model.fit(x, y, { epochs: 200, batchSize: 32 });
        const logs = model.evaluate(x, y);
        assert.ok(logs.loss < 0.005, `mse ${logs.loss}`);
        assert.ok(Math.abs(logs.rootMeanSquaredError - Math.sqrt(logs.loss)) < 1e-12);
        assert.ok(history.history.loss[199] < history.history.loss[0] / 50);
        assert.ok(Math.abs(model.predict([Math.PI / 2])[0] - 1) < 0.1);
    });

    it("trains BatchNormalization and Dropout models, updating moving statistics", () => {
        const { x, y } = spiral(60, 3, new Random(5));
        const model = new Sequential({
            inputSize: 2,
            seed: 3,
            layers: [dense(32), batchNormalization(), activation("relu"), dropout(0.1), dense(3, "softmax")],
        });
        model.compile({ loss: "scce", optimizer: { name: "adam", learningRate: 0.02 }, metrics: ["accuracy"] });
        model.fit(x, y, { epochs: 100, batchSize: 30 });
        const moving = model.getWeights().find((w) => w.name === "batch_normalization_1/movingMean");
        assert.ok(moving !== undefined && Array.from(moving.data).some((v) => v !== 0));
        assert.ok(model.evaluate(x, y).accuracy > 0.85);
        assert.deepEqual(model.predict(x.slice(0, 5)), model.predict(x.slice(0, 5)), "inference is deterministic");
    });

    it("is bit-for-bit reproducible: same seed + data + options => same weights and history", () => {
        const { x, y } = randomData(new Random(1), 50, 3, 2);
        const run = () => {
            const model = new Sequential({ seed: 99, layers: [dense(6, "relu"), dropout(0.2), dense(2)] });
            model.compile({ loss: "mse", optimizer: "adam", metrics: ["mae"] });
            const history = model.fit(x, y, { epochs: 5, batchSize: 8, validationSplit: 0.2 });
            return { weights: weightsOf(model), history: withoutDuration(history.history) };
        };
        const a = run();
        const b = run();
        assert.deepEqual(a, b);
        const other = new Sequential({ seed: 100, layers: [dense(6, "relu"), dropout(0.2), dense(2)] });
        other.build(3);
        assert.notDeepEqual(weightsOf(other).slice(0, 1), a.weights.slice(0, 1));
    });

    it("infers inputSize on first use, with the same initial weights as an explicit inputSize", () => {
        const explicit = new Sequential({ inputSize: 3, seed: 5, layers: [dense(4), dense(1)] });
        const inferred = new Sequential({ seed: 5, layers: [dense(4), dense(1)] });
        assert.equal(inferred.built, false);
        assert.equal(inferred.inputSize, undefined);
        inferred.predict([1, 2, 3]);
        assert.equal(inferred.inputSize, 3);
        assert.equal(inferred.outputSize, 1);
        assert.deepEqual(weightsOf(inferred), weightsOf(explicit));
    });

    it("trainOnBatch runs one step and returns pre-update logs", () => {
        const model = xorModel();
        const before = weightsOf(model);
        const logs = model.trainOnBatch(XOR_X, XOR_Y);
        assert.deepEqual(Object.keys(logs), ["loss", "accuracy"]);
        assert.ok(
            Math.abs(
                logs.loss -
                    getLoss("bce").compute(
                        Matrix.from(
                            new Sequential({
                                inputSize: 2,
                                seed: 42,
                                layers: [dense(8, "tanh"), dense(1, "sigmoid")],
                            }).predict(XOR_X),
                        ),
                        new Matrix(4, 1, XOR_Y),
                    ),
            ) < 1e-12,
        );
        assert.notDeepEqual(weightsOf(model), before);
        assert.equal(model.optimizer?.iterations, 1);
    });

    it("stops with a TrainingError suggesting a lower learning rate when the loss diverges", () => {
        const x = Array.from({ length: 16 }, (_, i) => [i * 10, 100 - i * 5]);
        const y = x.map(([a, b]) => a * 3 + b);
        const model = new Sequential({ inputSize: 2, seed: 1, layers: [dense(8), dense(1)] });
        model.compile({ loss: "mse", optimizer: { name: "sgd", learningRate: 1 } });
        assert.throws(
            () => model.fit(x, y, { epochs: 50, batchSize: 4 }),
            (e: unknown) =>
                e instanceof TrainingError &&
                /Training diverged: the loss became (NaN|Infinity) at epoch \d+\/50, batch \d\/4/.test(e.message) &&
                /learning rate \(1\) is too high/.test(e.message),
        );
    });
});

describe("Sequential: gradients", () => {
    /** Spy that counts fused vs plain gradient calls. */
    function spyLoss(id: LossIdentifier): { loss: Loss; fused: number; plain: number } {
        const base = getLoss(id);
        const spy = {
            fused: 0,
            plain: 0,
            loss: {
                name: base.name,
                compute: (p, t) => base.compute(p, t),
                gradient: (p, t, o) => {
                    spy.plain++;
                    return base.gradient(p, t, o);
                },
                fusedGradient: (a, p, t, o) => {
                    const result = base.fusedGradient?.(a, p, t, o) ?? null;
                    if (result !== null) spy.fused++;
                    return result;
                },
                getConfig: () => base.getConfig(),
            } as Loss,
        };
        return spy;
    }

    const cases: {
        output: ActivationName;
        loss: LossIdentifier;
        outputs: number;
        fused: boolean;
        sparse?: boolean;
        standalone?: boolean;
    }[] = [
        { output: "sigmoid", loss: "bce", outputs: 2, fused: true },
        { output: "softmax", loss: "cce", outputs: 3, fused: true },
        { output: "softmax", loss: "scce", outputs: 3, fused: true, sparse: true },
        { output: "tanh", loss: "mse", outputs: 2, fused: false },
        { output: "softmax", loss: "mse", outputs: 3, fused: false },
        { output: "linear", loss: { name: "huber", delta: 0.5 }, outputs: 2, fused: false },
        { output: "sigmoid", loss: "mae", outputs: 1, fused: false },
        { output: "softmax", loss: "cce", outputs: 3, fused: true, standalone: true },
        { output: "sigmoid", loss: "bce", outputs: 2, fused: true, standalone: true },
    ];

    for (const c of cases) {
        const label = `${c.output} + ${typeof c.loss === "string" ? c.loss : (c.loss as { name: string }).name}`;
        const where = c.standalone ? "Dense + Activation layer head" : "3-layer model";
        it(`matches finite differences for every parameter of a ${where} (${label}${c.fused ? ", fused" : ""})`, () => {
            const rng = new Random(21);
            const model = new Sequential({
                inputSize: 3,
                seed: 4,
                layers: [
                    dense(5, { activation: "tanh", kernelRegularizer: { l1: 0.01, l2: 0.02 } }),
                    dense(4, "elu"),
                    ...(c.standalone
                        ? [
                              dense(c.outputs, { biasInitializer: { name: "randomNormal", stddev: 0.5 } }),
                              activation(c.output),
                          ]
                        : [
                              dense(c.outputs, {
                                  activation: c.output,
                                  biasInitializer: { name: "randomNormal", stddev: 0.5 },
                              }),
                          ]),
                ],
            });
            const rows = 6;
            const x = Array.from({ length: rows }, () => [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)]);
            let y: number[][] | number[];
            if (c.sparse) y = Array.from({ length: rows }, () => rng.int(c.outputs));
            else if (c.output === "softmax" && c.loss === "cce") {
                y = Array.from({ length: rows }, () => {
                    const row = new Array(c.outputs).fill(0);
                    row[rng.int(c.outputs)] = 1;
                    return row;
                });
            } else if (c.output === "sigmoid")
                y = Array.from({ length: rows }, () => Array.from({ length: c.outputs }, () => rng.int(2)));
            else
                y = Array.from({ length: rows }, () => Array.from({ length: c.outputs }, () => rng.uniform(-0.9, 0.9)));

            const spy = spyLoss(c.loss);
            model.compile({ loss: spy.loss, optimizer: { name: "sgd", learningRate: 0 } });
            model.trainOnBatch(x, y);
            assert.deepEqual([spy.fused, spy.plain], c.fused ? [1, 0] : [0, 1]);

            const h = 1e-6;
            const lossAt = () => model.evaluate(x, y, { batchSize: rows }).loss;
            for (const param of model.parameters()) {
                const grad = param.grad.data.slice();
                const values = param.value.data;
                for (let i = 0; i < values.length; i++) {
                    const original = values[i];
                    values[i] = original + h;
                    const plus = lossAt();
                    values[i] = original - h;
                    const minus = lossAt();
                    values[i] = original;
                    const numeric = (plus - minus) / (2 * h);
                    const scale = Math.max(1, Math.abs(numeric), Math.abs(grad[i]));
                    assert.ok(
                        Math.abs(numeric - grad[i]) / scale < 1e-6,
                        `${param.name}[${i}]: ${grad[i]} vs ${numeric}`,
                    );
                }
            }
        });
    }
});

describe("Sequential: fused softmax + categoricalCrossentropy with unnormalized targets", () => {
    for (const standalone of [false, true]) {
        it(`matches finite differences when target rows do not sum to 1 (${standalone ? "Activation layer head" : "Dense softmax head"})`, () => {
            const rng = new Random(8);
            const model = new Sequential({
                inputSize: 3,
                seed: 2,
                layers: standalone
                    ? [dense(4, "tanh"), dense(3), activation("softmax")]
                    : [dense(4, "tanh"), dense(3, "softmax")],
            });
            model.compile({ loss: "cce", optimizer: { name: "sgd", learningRate: 0 } });
            const x = Array.from({ length: 5 }, () => [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)]);
            // Multi-hot, all-zero and soft-count rows.
            const y = [
                [1, 1, 0],
                [0, 0, 0],
                [0.5, 0.2, 0.9],
                [2, 0, 0],
                [0, 0.3, 0],
            ];
            model.trainOnBatch(x, y);
            const h = 1e-6;
            const lossAt = () => model.evaluate(x, y, { batchSize: 5 }).loss;
            for (const param of model.parameters()) {
                const grad = param.grad.data.slice();
                const values = param.value.data;
                for (let i = 0; i < values.length; i++) {
                    const original = values[i];
                    values[i] = original + h;
                    const plus = lossAt();
                    values[i] = original - h;
                    const minus = lossAt();
                    values[i] = original;
                    const numeric = (plus - minus) / (2 * h);
                    const scale = Math.max(1, Math.abs(numeric), Math.abs(grad[i]));
                    assert.ok(
                        Math.abs(numeric - grad[i]) / scale < 1e-6,
                        `${param.name}[${i}]: ${grad[i]} vs ${numeric}`,
                    );
                }
            }
        });
    }
});

describe("Sequential: validation, logs and callbacks", () => {
    const { x, y } = randomData(new Random(8), 40, 2, 1);

    function regressionModel(seed = 1): Sequential {
        const model = new Sequential({ inputSize: 2, seed, layers: [dense(8, "tanh"), dense(1)] });
        model.compile({ loss: "mse", optimizer: { name: "adam", learningRate: 0.01 }, metrics: ["mae", "rmse"] });
        return model;
    }

    it("logs loss, metrics, validation metrics, learning rate and duration", () => {
        const model = regressionModel();
        const history = model.fit(x, y, { epochs: 3, batchSize: 8, validationSplit: 0.25 });
        assert.deepEqual(history.epochs, [0, 1, 2]);
        assert.deepEqual(Object.keys(history.history), [
            "loss",
            "meanAbsoluteError",
            "rootMeanSquaredError",
            "valLoss",
            "valMeanAbsoluteError",
            "valRootMeanSquaredError",
            "learningRate",
            "durationMs",
        ]);
        assert.deepEqual(history.history.learningRate, [0.01, 0.01, 0.01]);
        assert.ok(history.history.durationMs.every((d) => d >= 0));
    });

    it("validationSplit holds out the last samples, evaluated after each epoch", () => {
        const model = regressionModel();
        let samples = 0;
        const history = model.fit(x, y, {
            epochs: 2,
            batchSize: 8,
            validationSplit: 0.25,
            callbacks: [
                {
                    onTrainBegin: (ctx) => {
                        samples = ctx.samples;
                    },
                },
            ],
        });
        assert.equal(samples, 30);
        const expected = model.evaluate(x.slice(30), y.slice(30), { batchSize: 8 });
        assert.equal(history.last("valLoss"), expected.loss);
        assert.equal(history.last("valMeanAbsoluteError"), expected.meanAbsoluteError);
        assert.equal(history.last("valRootMeanSquaredError"), expected.rootMeanSquaredError);
    });

    it("validationData takes precedence and is reported as val*", () => {
        const model = regressionModel();
        const valX = x.slice(0, 10);
        const valY = y.slice(0, 10);
        const history = model.fit(x, y, {
            epochs: 2,
            batchSize: 16,
            validationData: [valX, valY],
            validationSplit: 0.5,
        });
        assert.equal(history.last("valLoss"), model.evaluate(valX, valY, { batchSize: 16 }).loss);
    });

    it("reports epoch metrics as batch-size-weighted means (RMSE via the mean square)", () => {
        const model = regressionModel();
        model.compile({ loss: "mse", optimizer: { name: "sgd", learningRate: 0 }, metrics: ["mse", "rmse"] });
        const history = model.fit(x, y, { epochs: 1, batchSize: 7, shuffle: false });
        const full = model.evaluate(x, y, { batchSize: 40 });
        assert.ok(Math.abs(history.last("loss")! - full.loss) < 1e-12);
        assert.ok(Math.abs(history.last("meanSquaredError")! - full.meanSquaredError) < 1e-12);
        assert.ok(Math.abs(history.last("rootMeanSquaredError")! - full.rootMeanSquaredError) < 1e-12);
        const chunked = model.evaluate(x, y, { batchSize: 3 });
        assert.ok(Math.abs(chunked.rootMeanSquaredError - full.rootMeanSquaredError) < 1e-12);
    });

    it("includes kernel regularization in loss and valLoss", () => {
        const model = new Sequential({
            inputSize: 2,
            seed: 1,
            layers: [dense(4, { kernelRegularizer: { l2: 0.5 } }), dense(1)],
        });
        model.compile({ loss: "mse", optimizer: { name: "sgd", learningRate: 0 } });
        const plain = getLoss("mse").compute(
            model.predict(Matrix.from(x)),
            new Matrix(
                40,
                1,
                y.map((r) => r[0]),
            ),
        );
        const layer = model.layers[0] as unknown as { regularizationLoss(): number };
        assert.ok(Math.abs(model.evaluate(x, y).loss - (plain + layer.regularizationLoss())) < 1e-12);
    });

    it("calls every hook in order with a useful context", () => {
        const events: string[] = [];
        let contextSeen = false;
        const recorder: Callback = {
            onTrainBegin: (ctx) => {
                events.push("trainBegin");
                contextSeen =
                    ctx.epochs === 2 && ctx.batchSize === 16 && ctx.stepsPerEpoch === 2 && ctx.hasValidation === false;
            },
            onEpochBegin: (epoch) => void events.push(`epochBegin ${epoch}`),
            onBatchEnd: (batch, logs) => void events.push(`batchEnd ${batch} size=${logs.size} ${typeof logs.loss}`),
            onEpochEnd: (epoch, logs) => void events.push(`epochEnd ${epoch} ${Object.keys(logs).join(",")}`),
            onTrainEnd: (logs) => void events.push(`trainEnd ${logs.loss !== undefined}`),
        };
        regressionModel().fit(x.slice(0, 20), y.slice(0, 20), { epochs: 2, batchSize: 16, callbacks: [recorder] });
        const epochKeys = "loss,meanAbsoluteError,rootMeanSquaredError,learningRate,durationMs";
        assert.deepEqual(events, [
            "trainBegin",
            "epochBegin 0",
            "batchEnd 0 size=16 number",
            "batchEnd 1 size=4 number",
            `epochEnd 0 ${epochKeys}`,
            "epochBegin 1",
            "batchEnd 0 size=16 number",
            "batchEnd 1 size=4 number",
            `epochEnd 1 ${epochKeys}`,
            "trainEnd true",
        ]);
        assert.ok(contextSeen);
    });

    it("supports onEpochEnd shorthand, stopTraining() and verbose logging", () => {
        const seen: number[] = [];
        const lines: string[] = [];
        const original = console.log;
        console.log = (line: string) => void lines.push(line);
        try {
            const history = regressionModel().fit(x, y, {
                epochs: 10,
                verbose: 2,
                onEpochEnd: (epoch) => void seen.push(epoch),
                callbacks: [{ onEpochEnd: (epoch, _logs, ctx) => (epoch === 4 ? ctx.stopTraining() : undefined) }],
            });
            assert.deepEqual(history.epochs, [0, 1, 2, 3, 4]);
        } finally {
            console.log = original;
        }
        assert.deepEqual(seen, [0, 1, 2, 3, 4]);
        assert.equal(lines.length, 3); // epochs 2, 4 and the stopped epoch 5
        assert.match(
            lines[0],
            /^Epoch {2}2\/10 - loss: \d\.\d{4} - meanAbsoluteError: \d\.\d{4} - rootMeanSquaredError: \d\.\d{4} - [\d.]+ms$/,
        );
    });

    it("treats verbose: 0 as silent (like false), as in Keras", () => {
        const lines: string[] = [];
        const original = console.log;
        console.log = (line: string) => void lines.push(line);
        try {
            regressionModel().fit(x, y, { epochs: 2, verbose: 0 });
        } finally {
            console.log = original;
        }
        assert.deepEqual(lines, []);
        assert.throws(() => regressionModel().fit(x, y, { verbose: -1 }), /"verbose" must be a positive integer/);
    });

    it("early stopping restores the weights of the best epoch", () => {
        const model = regressionModel();
        const script = [5, 4, 3, 3.5, 3.6, 3.7, 3.8, 3.9];
        const snapshots: WeightEntry[][] = [];
        const stopper = earlyStopping({ patience: 2, restoreBestWeights: true });
        model.fit(x, y, {
            epochs: 8,
            validationSplit: 0.2,
            callbacks: [
                {
                    onEpochEnd: (epoch: number, logs: Logs) => {
                        logs.valLoss = script[epoch];
                    },
                },
                stopper,
                { onEpochEnd: () => void snapshots.push(model.getWeights()) },
            ],
        });
        assert.equal(stopper.stoppedEpoch, 4);
        assert.equal(stopper.bestEpoch, 2);
        assert.equal(stopper.bestValue, 3);
        assert.equal(snapshots.length, 5);
        assert.deepEqual(model.getWeights(), snapshots[2]);
        assert.notDeepEqual(snapshots[4], snapshots[2]);
    });

    it("sync fit rejects callbacks that return a Promise, pointing to fitAsync", () => {
        assert.throws(
            () => regressionModel().fit(x, y, { epochs: 2, callbacks: [{ onEpochEnd: async () => {} }] }),
            (e: unknown) =>
                e instanceof ValidationError && /onEpochEnd\(\) returned a Promise.*fitAsync/.test(e.message),
        );
    });
});

describe("Sequential: fitAsync", () => {
    const { x, y } = randomData(new Random(9), 64, 2, 1);

    function model(): Sequential {
        const m = new Sequential({ inputSize: 2, seed: 3, layers: [dense(8, "relu"), dropout(0.1), dense(1)] });
        m.compile({ loss: "mse", optimizer: "adam", metrics: ["mae"] });
        return m;
    }

    it("produces exactly the same result as fit", async () => {
        const a = model();
        const b = model();
        const syncHistory = a.fit(x, y, { epochs: 4, batchSize: 8, validationSplit: 0.25 });
        const asyncHistory = await b.fitAsync(x, y, { epochs: 4, batchSize: 8, validationSplit: 0.25, yieldEvery: 0 });
        assert.deepEqual(withoutDuration(asyncHistory.history), withoutDuration(syncHistory.history));
        assert.deepEqual(weightsOf(b), weightsOf(a));
    });

    it("awaits async callbacks before continuing", async () => {
        const events: string[] = [];
        await model().fitAsync(x, y, {
            epochs: 2,
            callbacks: [
                { onEpochBegin: (epoch) => void events.push(`begin ${epoch}`) },
                {
                    onEpochEnd: async (epoch) => {
                        events.push(`end ${epoch}`);
                        await new Promise((resolve) => setTimeout(resolve, 5));
                        events.push(`awaited ${epoch}`);
                    },
                },
            ],
            onEpochEnd: async (epoch) => void events.push(`shorthand ${epoch}`),
        });
        assert.deepEqual(events, [
            "begin 0",
            "end 0",
            "awaited 0",
            "shorthand 0",
            "begin 1",
            "end 1",
            "awaited 1",
            "shorthand 1",
        ]);
    });

    it("yields to the event loop while training", async () => {
        let timerFired = false;
        let firedAt = -1;
        setTimeout(() => {
            timerFired = true;
        }, 0);
        await model().fitAsync(x, y, {
            epochs: 30,
            yieldEvery: 0,
            onEpochEnd: (epoch) => {
                if (timerFired && firedAt < 0) firedAt = epoch;
            },
        });
        assert.ok(firedAt >= 0 && firedAt < 29, `timer fired at epoch ${firedAt}`);
    });

    it("rejects with the abort reason when the signal aborts, keeping the progress so far", async () => {
        const m = model();
        const controller = new AbortController();
        await assert.rejects(
            m.fitAsync(x, y, {
                epochs: 100,
                batchSize: 16,
                signal: controller.signal,
                onEpochEnd: (epoch) => (epoch === 2 ? controller.abort() : undefined),
            }),
            (e: unknown) => e instanceof Error && e.name === "AbortError",
        );
        assert.equal(m.optimizer?.iterations, 3 * 4);
        const aborted = AbortSignal.abort();
        const fresh = model();
        await assert.rejects(fresh.fitAsync(x, y, { signal: aborted }), { name: "AbortError" });
        assert.equal(fresh.optimizer?.iterations, 0);
    });

    it("refuses to start a second fit while one is running", async () => {
        const m = model();
        const first = m.fitAsync(x, y, { epochs: 3, yieldEvery: 0 });
        await assert.rejects(m.fitAsync(x, y), /already training/);
        assert.throws(() => m.fit(x, y), /already training/);
        await first;
        assert.doesNotThrow(() => m.fit(x, y));
        assert.throws(() => m.fit(x, y, { callbacks: [{ onEpochEnd: () => void m.fit(x, y) }] }), /already training/);
        assert.doesNotThrow(() => m.fit(x, y), "the flag is cleared after an error");
    });

    it("is not left marked as training when yielding to the event loop fails", async () => {
        const m = model();
        const g = globalThis as { scheduler?: unknown };
        const previous = g.scheduler;
        g.scheduler = { yield: () => Promise.reject(new Error("yield failed")) };
        try {
            await assert.rejects(m.fitAsync(x, y, { epochs: 3, yieldEvery: 0 }), /yield failed/);
        } finally {
            if (previous === undefined) delete g.scheduler;
            else g.scheduler = previous;
        }
        assert.doesNotThrow(() => m.fit(x, y), "the training flag is cleared");
        await m.fitAsync(x, y, { epochs: 1, yieldEvery: 0 });
    });

    it("refuses compile() and add() from a callback instead of desynchronizing the loop", async () => {
        // Before: recompiling mid-fit made trainStep use the new optimizer while callbacks
        // (learningRateScheduler, reduceLROnPlateau) and logs.learningRate kept the old one.
        const m = model();
        assert.throws(
            () =>
                m.fit(x, y, {
                    epochs: 2,
                    onEpochEnd: () => void m.compile({ loss: "mse", optimizer: { name: "sgd", learningRate: 0.5 } }),
                }),
            (e: unknown) =>
                e instanceof ValidationError &&
                /compile\(\) cannot be called while the model is training/.test(e.message),
        );
        assert.equal(m.optimizer?.name, "adam", "the original optimizer is kept");
        await assert.rejects(
            m.fitAsync(x, y, { epochs: 2, onEpochEnd: () => void m.add(dense(2)) }),
            /add\(\) cannot be called while the model is training/,
        );
        assert.equal(m.layers.length, 3);
        // Both work again once training has ended.
        m.compile({ loss: "mse", optimizer: "sgd" });
        m.add(dense(1));
        assert.equal(m.layers.length, 4);
    });

    it("rejects invalid input asynchronously", async () => {
        await assert.rejects(model().fitAsync(x, y, { epochs: 0 }), /"epochs" must be a positive integer, got 0/);
        await assert.rejects(model().fitAsync(x, y, { signal: {} as AbortSignal }), /"signal" must be an AbortSignal/);
    });
});

describe("Sequential: inference and evaluation", () => {
    it("predict mirrors its input: sample -> number[], rows -> number[][], Matrix -> new Matrix", () => {
        const model = xorModel();
        const one = model.predict([1, 0]);
        const rows = model.predict([
            [1, 0],
            [0, 0],
        ]);
        const matrix = model.predict(
            Matrix.fromArray([
                [1, 0],
                [0, 0],
            ]),
        );
        assert.ok(Array.isArray(one) && typeof one[0] === "number" && one.length === 1);
        assert.equal(rows.length, 2);
        assert.deepEqual(rows[0], one);
        assert.ok(matrix instanceof Matrix);
        assert.deepEqual(matrix.toArray(), rows);
        model.predict(
            Matrix.fromArray([
                [1, 1],
                [0, 1],
            ]),
        );
        assert.deepEqual(matrix.toArray(), rows, "returned matrices are copies, not internal buffers");
        assert.deepEqual(model.predict([]), []);
    });

    it("predicts in chunks with identical results", () => {
        const rng = new Random(2);
        const model = new Sequential({
            inputSize: 3,
            seed: 2,
            layers: [dense(5, "relu"), batchNormalization(), dense(2, "softmax")],
        });
        const x = new Matrix(23, 3).map(() => rng.uniform(-2, 2));
        assert.deepEqual(model.predict(x, { batchSize: 4 }).data, model.predict(x).data);
    });

    it("evaluate returns the loss and each metric", () => {
        const model = xorModel();
        const logs = model.evaluate(XOR_X, XOR_Y);
        assert.deepEqual(Object.keys(logs), ["loss", "accuracy"]);
        assert.equal(logs.loss, getLoss("bce").compute(Matrix.from(model.predict(XOR_X)), new Matrix(4, 1, XOR_Y)));
    });

    it('"accuracy" with a binaryCrossentropy loss is element-wise binary accuracy (multi-label outputs)', () => {
        const make = () => {
            const m = new Sequential({ inputSize: 2, seed: 1, layers: [dense(3, "sigmoid")] });
            m.compile({ loss: "bce", metrics: ["accuracy", "binaryAccuracy", "categoricalAccuracy"] });
            return m;
        };
        const x = [
            [1, 2],
            [3, 4],
            [-1, 0.5],
        ];
        const y = [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ];
        const model = make();
        const logs = model.evaluate(x, y);
        assert.notEqual(logs.binaryAccuracy, logs.categoricalAccuracy, "the data tells the two variants apart");
        assert.equal(logs.accuracy, logs.binaryAccuracy);
        // Survives clone() and a save/load round trip (which recompiles from the metric names).
        assert.equal(model.clone().evaluate(x, y).accuracy, logs.binaryAccuracy);
        assert.equal(Sequential.fromArtifact(model.toArtifact()).evaluate(x, y).accuracy, logs.binaryAccuracy);
        // Other losses keep the shape-based choice.
        const softmax = new Sequential({ inputSize: 2, seed: 1, layers: [dense(3, "softmax")] });
        softmax.compile({ loss: "cce", metrics: ["accuracy", "categoricalAccuracy"] });
        const cce = softmax.evaluate(x, [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]);
        assert.equal(cce.accuracy, cce.categoricalAccuracy);
    });

    it("trainOnBatch rejects an empty batch without corrupting batch-norm statistics", () => {
        const model = new Sequential({
            inputSize: 2,
            seed: 1,
            layers: [dense(3), batchNormalization(), dense(1, "sigmoid")],
        });
        model.compile({ loss: "bce" });
        assert.throws(() => model.trainOnBatch(new Matrix(0, 2), new Matrix(0, 1)), /trainOnBatch: x has no samples/);
        const bn = model.layers[1];
        bn.forward(new Matrix(0, 3), true);
        for (const w of model.getWeights()) {
            assert.ok(
                Array.from(w.data).every(Number.isFinite),
                `${w.name} stays finite: ${Array.from(w.data).join(", ")}`,
            );
        }
        assert.ok(Number.isFinite(model.trainOnBatch(XOR_X, XOR_Y).loss));
    });
});

describe("Sequential: structure and weights", () => {
    it("auto-names layers per model and keeps user names", () => {
        const model = new Sequential({
            layers: [
                dense(4),
                dropout(0.1),
                dense(3, { name: "dense_2" }),
                batchNormalization(),
                dense(2),
                activation("relu"),
            ],
        });
        assert.deepEqual(
            model.layers.map((l) => l.name),
            ["dense_1", "dropout_1", "dense_2", "batch_normalization_1", "dense_3", "activation_1"],
        );
        assert.throws(() => model.add(dense(1, { name: "dense_1" })), /Duplicate layer name "dense_1"/);
        const shared = dense(2);
        model.add(shared);
        assert.throws(() => model.add(shared), /already in this model/);
        assert.throws(() => model.add({} as never), /add\(\) expects a layer/);
    });

    it("add() is chainable and builds immediately when the input size is known", () => {
        const model = new Sequential({ inputSize: 3, seed: 1 });
        assert.equal(model.add(dense(4)).add(dense(2)), model);
        assert.equal(model.built, true);
        assert.equal(model.parameterCount, 3 * 4 + 4 + 4 * 2 + 2);
        assert.deepEqual(
            model.parameters().map((p) => p.name),
            ["dense_1/kernel", "dense_1/bias", "dense_2/kernel", "dense_2/bias"],
        );
        assert.equal(model.outputSize, 2);
        assert.throws(() => model.build(4), /already built for inputSize 3; got 4/);
    });

    it("renders a summary table", () => {
        const model = new Sequential({
            inputSize: 4,
            name: "demo",
            layers: [dense(16, "relu"), batchNormalization(), dropout(0.2), dense(3, "softmax")],
        });
        assert.equal(
            model.summary(),
            [
                'Model: "demo"',
                "┌────────────────────────────────────────────┬─────────────┬────────────┬────────┐",
                "│ Layer (type)                               │ Output size │ Activation │ Params │",
                "├────────────────────────────────────────────┼─────────────┼────────────┼────────┤",
                "│ dense_1 (Dense)                            │ 16          │ relu       │     80 │",
                "│ batch_normalization_1 (BatchNormalization) │ 16          │ -          │     64 │",
                "│ dropout_1 (Dropout)                        │ 16          │ -          │      0 │",
                "│ dense_2 (Dense)                            │ 3           │ softmax    │     51 │",
                "└────────────────────────────────────────────┴─────────────┴────────────┴────────┘",
                "Input size: 4",
                "Total params: 195",
                "Trainable params: 163",
                "Non-trainable params: 32",
            ].join("\n"),
        );
        assert.match(new Sequential({ layers: [dense(2)] }).summary(), /Total params: \? \(call build\(inputSize\)/);
    });

    it("getWeights / setWeights round-trip and validate atomically", () => {
        const a = new Sequential({ inputSize: 2, seed: 1, layers: [dense(3), dense(1)] });
        const b = new Sequential({ inputSize: 2, seed: 2, layers: [dense(3), dense(1)] });
        const weights = a.getWeights();
        assert.deepEqual(
            weights.map((w) => [w.name, w.shape]),
            [
                ["dense_1/kernel", [2, 3]],
                ["dense_1/bias", [1, 3]],
                ["dense_2/kernel", [3, 1]],
                ["dense_2/bias", [1, 1]],
            ],
        );
        b.setWeights(weights.slice().reverse());
        assert.deepEqual(b.predict([0.3, -0.7]), a.predict([0.3, -0.7]));
        weights[0].data[0] = 123;
        assert.notEqual(a.getWeights()[0].data[0], 123, "getWeights returns copies");

        const before = weightsOf(b);
        const bad = a.getWeights();
        bad[3] = { name: "dense_2/bias", shape: [1, 2], data: [0, 0] };
        assert.throws(
            () => b.setWeights(bad),
            /weight "dense_2\/bias" has shape \[1, 2\], but the model expects \[1, 1\]/,
        );
        assert.throws(() => b.setWeights(a.getWeights().slice(1)), /missing weight\(s\) "dense_1\/kernel"/);
        assert.throws(
            () => b.setWeights([...a.getWeights(), { name: "x/kernel", shape: [1, 1], data: [1] }]),
            /unknown weight "x\/kernel"/,
        );
        const nan = a.getWeights();
        nan[1].data[0] = Number.NaN;
        assert.throws(() => b.setWeights(nan), /must be finite/);
        assert.deepEqual(weightsOf(b), before, "a failed setWeights changes nothing");
        assert.throws(() => new Sequential({ layers: [dense(1)] }).setWeights([]), /not built yet/);
    });

    it("clone() is an independent deep copy, compiled the same way", () => {
        const model = xorModel();
        model.fit(XOR_X, XOR_Y, { epochs: 20, batchSize: 4 });
        const copy = model.clone();
        assert.equal(copy.name, "xor");
        assert.equal(copy.compiled, true);
        assert.deepEqual(copy.optimizer?.getConfig(), model.optimizer?.getConfig());
        assert.notEqual(copy.optimizer, model.optimizer);
        assert.deepEqual(copy.predict(XOR_X), model.predict(XOR_X));
        const before = model.predict(XOR_X);
        copy.fit(XOR_X, XOR_Y, { epochs: 20, batchSize: 4 });
        assert.deepEqual(model.predict(XOR_X), before);
        assert.notDeepEqual(copy.predict(XOR_X), before);
        assert.equal(new Sequential({ layers: [dense(2)] }).clone().built, false);
    });
});

describe("Sequential: actionable errors", () => {
    const model = () => xorModel();

    it("explains input and target shape mismatches", () => {
        assert.throws(
            () => model().predict([1, 2, 3]),
            (e: unknown) =>
                e instanceof ShapeError && e.message === "Expected input with 2 features (inputSize), got 3",
        );
        const scalar = new Sequential({ inputSize: 1, layers: [dense(1)] });
        assert.throws(
            () => scalar.predict([0.1, 0.2]),
            /Expected input with 1 feature \(inputSize\), got 2. A flat array is one sample; pass one row per sample/,
        );
        assert.throws(() => model().fit([[0, 1, 2]], [1]), /Expected input with 2 features \(inputSize\), got 3/);
        assert.throws(() => model().fit(XOR_X, [0, 1, 1]), /x and y must have the same number of samples, got 4 and 3/);
        assert.throws(
            () =>
                model().fit(XOR_X, [
                    [0, 1],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                ]),
            /expected y with 1 column \(outputSize of the last layer\), got 2/,
        );
        assert.throws(() => model().fit([[0, 1], [1]], [0, 1]), /Ragged input: row 1 has length 1, expected 2/);
    });

    it("guides classification target formats", () => {
        const classifier = new Sequential({ inputSize: 2, layers: [dense(3, "softmax")] });
        classifier.compile({ loss: "cce" });
        assert.throws(
            () => classifier.fit(XOR_X, [0, 1, 2, 1]),
            /use loss "scce", or one-hot encode them with oneHot\(labels, 3\)/,
        );
        classifier.compile({ loss: "scce" });
        assert.throws(
            () =>
                classifier.fit(XOR_X, [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                ]),
            /one integer class index per sample, got 3 columns/,
        );
        assert.throws(() => classifier.fit(XOR_X, [0, 1, 3, 1]), /y\[2\] = 3 is not a class index in \[0, 3\)/);
    });

    it("explains missing compile, layers, bad options and non-finite data", () => {
        const plain = new Sequential({ inputSize: 2, layers: [dense(1)] });
        assert.throws(
            () => plain.fit(XOR_X, XOR_Y),
            /not compiled; call model.compile\(\{ loss: "mse", optimizer: "adam" \}\) first/,
        );
        assert.throws(() => plain.compile({} as never), /"loss" is required/);
        assert.throws(() => new Sequential().predict([1]), /the model has no layers/);
        assert.throws(
            () => model().fit(XOR_X, XOR_Y, { batch_size: 2 } as never),
            /unknown option "batch_size". Valid options: epochs, batchSize/,
        );
        assert.throws(
            () => model().fit(XOR_X, XOR_Y, { batchSize: 0 }),
            /"batchSize" must be a positive integer, got 0/,
        );
        assert.throws(
            () => model().fit([[0, Number.POSITIVE_INFINITY]], [1]),
            /x contains Infinity at row 0, column 1; clean or impute/,
        );
        assert.throws(() => model().fit([[0, Number.NaN]], [1]), /Non-numeric value at \[0, 1\]: NaN/);
        assert.throws(
            () =>
                model().fit(
                    Matrix.fromArray([[0, 0]]).map(() => Number.NaN),
                    [1],
                ),
            /x contains NaN at row 0, column 0/,
        );
        assert.throws(() => model().fit(XOR_X, XOR_Y, { validationSplit: 0.1 }), /leaves 0 for validation/);
        assert.throws(() => model().fit([], []), /fit: x has no samples/);
        assert.throws(() => model().evaluate([], []), /evaluate: x has no samples/);
        assert.throws(() => model().trainOnBatch([], []), /trainOnBatch: x has no samples/);
        assert.throws(
            () => model().fit(XOR_X, XOR_Y, { validationData: [[], []] }),
            /fit: validationData x has no samples/,
        );
        assert.throws(() => new Sequential({ seed: 1.5 }), /"seed" must be an integer/);
        assert.throws(() => model().compile({ loss: "mse", metrics: ["accuracy", "accuracy"] }), /listed twice/);
        assert.throws(() => model().toArtifact.call(new Sequential({ layers: [dense(1)] })), /not built yet/);
    });
});

describe("initialEpoch", () => {
    it("numbers epochs from initialEpoch so schedules and history continue across fit calls", () => {
        const model = xorModel();
        const seen: number[] = [];
        const lrs: number[] = [];
        const callback: Callback = {
            onEpochBegin(epoch, ctx) {
                seen.push(epoch);
                ctx.optimizer.learningRate = 0.1 / (epoch + 1);
                lrs.push(ctx.optimizer.learningRate);
            },
        };
        const first = model.fit(XOR_X, XOR_Y, { epochs: 3, batchSize: 4, callbacks: [callback] });
        const second = model.fit(XOR_X, XOR_Y, { epochs: 2, batchSize: 4, initialEpoch: 3, callbacks: [callback] });
        assert.deepEqual(seen, [0, 1, 2, 3, 4]);
        assert.deepEqual(first.epochs, [0, 1, 2]);
        assert.deepEqual(second.epochs, [3, 4]);
        assert.equal(second.last("learningRate"), 0.1 / 5);
        assert.equal(lrs.length, 5);
    });

    it("reports absolute epoch numbers to the progress logger", () => {
        const model = xorModel();
        const lines: string[] = [];
        const original = console.log;
        console.log = (line: string) => lines.push(line);
        try {
            model.fit(XOR_X, XOR_Y, { epochs: 2, batchSize: 4, initialEpoch: 8, verbose: true });
        } finally {
            console.log = original;
        }
        assert.equal(lines.length, 2);
        assert.match(lines[0], /^Epoch {1,2}9\/10 /);
        assert.match(lines[1], /^Epoch 10\/10 /);
    });

    it("rejects invalid values", () => {
        const model = xorModel();
        assert.throws(() => model.fit(XOR_X, XOR_Y, { initialEpoch: -1 }), /"initialEpoch" must be a non-negative integer/);
        assert.throws(() => model.fit(XOR_X, XOR_Y, { initialEpoch: 1.5 }), /"initialEpoch" must be a non-negative integer/);
    });
});
