import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { SerializationError, ValidationError } from "../src/core/errors.js";
import { Matrix } from "../src/core/matrix.js";
import { Random } from "../src/core/random.js";
import { decodeArtifact } from "../src/io/index.js";
import { activation, batchNormalization, dense, dropout } from "../src/layers/index.js";
import { Sequential } from "../src/model.js";
import { deserializeModel, serializeModel } from "../src/serialization.js";

function trainedModel(): { model: Sequential; x: Matrix } {
    const rng = new Random(4);
    const x = new Matrix(48, 3).map(() => rng.uniform(-2, 2));
    const y = Array.from({ length: 48 }, (_, i) => (x.get(i, 0) + x.get(i, 1) * x.get(i, 2) > 0 ? 1 : 0));
    const model = new Sequential({
        inputSize: 3,
        seed: 12,
        name: "round-trip",
        layers: [
            dense(8, { activation: "relu", kernelRegularizer: { l2: 1e-3 } }),
            batchNormalization({ momentum: 0.9 }),
            dropout(0.25),
            dense(6),
            activation({ name: "leakyRelu", alpha: 0.2 }),
            dense(1, "sigmoid"),
        ],
    });
    model.compile({ loss: "bce", optimizer: { name: "adamw", learningRate: 0.01, weightDecay: 0.001 }, metrics: ["accuracy"] });
    model.fit(x, y, { epochs: 15, batchSize: 16 });
    return { model, x };
}

/**
 * Outputs of the legacy engine (src/classes.ts, orion-engine ≤ 0.0.2) for these .onn files, computed
 * by running the old engine before it was removed.
 */
const LEGACY_CASES = [
    {
        label: "linear -> tanh -> leakyRelu -> softmax",
        onn:
            "3:linear:4:tanh:3:leakyRelu:2:softmax\n0.641:-1.427:-0.273:1.498|-0.113:-1.469:0.491:1.342|-0.837:-1.126:1.127:0.836|" +
            "-1.343:-0.49:1.469:0.111|-0.74:1.389:0.382:-1.488:0.001|1.487:-0.385:-1.388:0.742:1.197|-1.051:-0.926:1.289:0.594:-1.442|" +
            "1.128:0.835:-1.343:-0.489|1.469:0.11:-1.498:0.276",
        inputs: [[0.5, -1.25, 2], [-3, 0.1, 0.7], [0, 0, 0]],
        expected: [
            [0.6048744011971335, 0.39512559880286646],
            [0.2745344474691761, 0.7254655525308239],
            [0.49617743334164616, 0.5038225666583538],
        ],
        exact: false,
    },
    {
        label: "relu -> elu -> swish -> sigmoid",
        onn:
            "2:relu:3:elu:3:swish:1:sigmoid\n0.641:-1.427:-0.273|1.498:-0.113:-1.469|0.491:1.342:-0.837|0.225:1.442:-0.596:-1.288|" +
            "0.928:1.049:-1.199:-0.74|1.389:0.382:-1.488:0.001|0.64:-1.428:-0.272:1.498",
        inputs: [[1, -2], [-0.5, 0.25], [3, 1.5]],
        expected: [[0.006184458977923792], [0.8578196548495731], [0.9236638255390072]],
        exact: false,
    },
    {
        label: "linear -> relu -> linear",
        onn: "2:linear:3:relu:1:linear\n0.641:-1.427:-0.273|1.498:-0.113:-1.469|0.491:1.342:-0.837|0.225:1.442:-0.596:-1.288",
        inputs: [[1, -2], [-0.5, 0.25], [3, 1.5]],
        expected: [[-0.1953400000000003], [-1.288], [1.2508269999999986]],
        exact: true,
    },
    {
        label: "linear -> tanh -> leakyRelu",
        onn:
            "2:linear:3:tanh:2:leakyRelu\n0.641:-1.427:-0.273|1.498:-0.113:-1.469|0.491:1.342:-0.837|0.225:1.442:-0.596:-1.288|" +
            "0.928:1.049:-1.199:-0.74",
        inputs: [[1, -2], [-0.5, 0.25], [3, 1.5]],
        expected: [
            [-0.0011054863222807244, 1.6403127095554473],
            [-0.02487074357208969, -0.01693007884606899],
            [-0.005478684632503138, -0.013068670597853079],
        ],
        exact: true,
    },
    {
        label: "documented example (relu -> swish)",
        onn: "2:relu:2:swish\n0.71:-0.2:0.19|-1.82:0.95:0.97",
        inputs: [[1, 0], [-1, 2], [0.3, 0.3]],
        expected: [
            [0.6398545523625034, -0.254517928897123],
            [-0.26216126275508944, 4.647306652313369],
            [0.20062724246610025, 0.47515773253510907],
        ],
        exact: false,
    },
];

describe("serializeModel / deserializeModel", () => {
    it("round-trips through JSON bit-exactly, including BatchNorm moving statistics", () => {
        const { model, x } = trainedModel();
        const text = serializeModel(model, { format: "json" });
        assert.equal(typeof text, "string");
        const restored = deserializeModel(text);
        assert.deepEqual(restored.getWeights(), model.getWeights());
        assert.deepEqual(restored.predict(x).data, model.predict(x).data);
        const moving = restored.getWeights().find((w) => w.name === "batch_normalization_1/movingVariance");
        assert.ok(moving !== undefined && Array.from(moving.data).some((v) => v !== 1));
        assert.match(serializeModel(model, { format: "json", pretty: true }), /\n {2}"formatVersion": 1,/);
    });

    it("round-trips through binary: bit-exact with float64, close with the float32 default", () => {
        const { model, x } = trainedModel();
        const exact = serializeModel(model, { precision: "float64" });
        assert.ok(exact instanceof Uint8Array);
        assert.deepEqual(deserializeModel(exact).predict(x).data, model.predict(x).data);

        const compact = serializeModel(model);
        assert.ok(compact.byteLength < exact.byteLength);
        const approx = deserializeModel(compact.buffer.slice(compact.byteOffset, compact.byteOffset + compact.byteLength));
        const expected = model.predict(x).data;
        approx.predict(x).data.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-5, `${v} vs ${expected[i]}`));
    });

    it("restores architecture, name, metadata and the training config (fresh optimizer)", () => {
        const { model } = trainedModel();
        const restored = deserializeModel(serializeModel(model, { format: "json", metadata: { author: "tests" } }), { seed: 5 });
        assert.equal(restored.name, "round-trip");
        assert.equal(restored.seed, 5);
        assert.deepEqual(restored.layers.map((l) => l.getConfig()), model.layers.map((l) => l.getConfig()));
        assert.equal(restored.compiled, true);
        assert.equal(restored.loss?.name, "binaryCrossentropy");
        assert.deepEqual(restored.optimizer?.getConfig(), model.optimizer?.getConfig());
        assert.equal(restored.optimizer?.iterations, 0);
        assert.deepEqual(restored.metrics.map((m) => m.name), ["accuracy"]);
        assert.deepEqual(decodeArtifact(serializeModel(model, { format: "json", metadata: { author: "tests" } })).metadata, {
            name: "round-trip",
            author: "tests",
        });
        // The restored model keeps training.
        const x = [[0, 1, 2]];
        assert.ok(Number.isFinite(restored.trainOnBatch(x, [1]).loss));
    });

    it("fromArtifact leaves models without a training config uncompiled", () => {
        const model = new Sequential({ inputSize: 2, seed: 1, layers: [dense(2, "relu"), dense(1)] });
        const artifact = model.toArtifact();
        assert.equal(artifact.training, undefined);
        const restored = Sequential.fromArtifact(artifact);
        assert.equal(restored.compiled, false);
        assert.deepEqual(restored.predict([0.5, 0.5]), model.predict([0.5, 0.5]));
    });

    it("rejects invalid models and options with clear errors", () => {
        const model = new Sequential({ inputSize: 2, layers: [dense(1)] });
        assert.throws(() => serializeModel(model, { format: "json", precision: "float64" }), /"precision" only applies to format "binary"/);
        assert.throws(() => serializeModel(model, { pretty: true }), /"pretty" only applies to format "json"/);
        assert.throws(() => serializeModel(model, { format: "xml" as never }), /"format" must be "binary" or "json"/);
        assert.throws(() => serializeModel(new Sequential({ layers: [dense(1)] })), /not built yet/);
        assert.throws(() => serializeModel({} as never), ValidationError);
        assert.throws(() => deserializeModel("not a model"), SerializationError);
        const artifact = model.toArtifact();
        artifact.layers[0] = { type: "conv2d", name: "c" };
        assert.throws(() => Sequential.fromArtifact(artifact), /Unknown layer type "conv2d"/);
        const wrongShape = model.toArtifact();
        wrongShape.weights[0] = { name: "dense_1/kernel", shape: [3, 1], data: [1, 2, 3] };
        assert.throws(() => Sequential.fromArtifact(wrongShape), /has shape \[3, 1\], but the model expects \[2, 1\]/);
    });
});

describe("legacy .onn import", () => {
    for (const c of LEGACY_CASES) {
        it(`reproduces the legacy engine's outputs (${c.label})`, () => {
            const model = deserializeModel(c.onn);
            assert.equal(model.compiled, false);
            assert.deepEqual(
                model.layers.map((l) => l.name),
                model.layers.map((_, i) => `dense_${i + 1}`),
            );
            const actual = model.predict(c.inputs);
            if (c.exact) {
                // linear, relu, tanh and leakyRelu are computed exactly as before: bit-identical.
                assert.deepEqual(actual, c.expected);
            } else {
                // sigmoid / swish / elu / softmax use numerically stabler formulas now (e.g. expm1,
                // no exp overflow), which can differ from the old ones in the last bit or two.
                actual.flat().forEach((v, i) => {
                    const e = c.expected.flat()[i];
                    assert.ok(Math.abs(v - e) <= 4 * Number.EPSILON * Math.max(1, Math.abs(e)), `${v} vs legacy ${e}`);
                });
            }
        });
    }

    it("also imports legacy files given as bytes", () => {
        const c = LEGACY_CASES[2];
        assert.deepEqual(deserializeModel(new TextEncoder().encode(c.onn)).predict(c.inputs), c.expected);
    });
});
