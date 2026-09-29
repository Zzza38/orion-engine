import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { ShapeError, ValidationError } from "../src/core/errors.js";
import { Matrix } from "../src/core/matrix.js";
import { Random } from "../src/core/random.js";
import type { Layer, Parameter } from "../src/core/types.js";
import {
    ActivationLayer,
    activation,
    BatchNormalization,
    batchNormalization,
    Dense,
    Dropout,
    dense,
    dropout,
    LAYER_TYPES,
    layerFromConfig,
    registerLayer,
} from "../src/layers/index.js";

function randomMatrix(rng: Random, rows: number, cols: number, lo = -1, hi = 1): Matrix {
    return new Matrix(rows, cols).map(() => rng.uniform(lo, hi));
}

/** Scalar objective sum(G ⊙ layer(X)) (+ regularization), whose gradient w.r.t. the output is G. */
function objective(layer: Layer, x: Matrix, g: Matrix, training: boolean): number {
    const y = layer.forward(x, training);
    let sum = 0;
    for (let i = 0; i < y.data.length; i++) sum += y.data[i] * g.data[i];
    const reg = (layer as { regularizationLoss?: () => number }).regularizationLoss;
    return sum + (reg ? reg.call(layer) : 0);
}

function assertGradClose(analytic: number, numeric: number, label: string, tol = 1e-6): void {
    const scale = Math.max(1, Math.abs(analytic), Math.abs(numeric));
    assert.ok(Math.abs(analytic - numeric) / scale < tol, `${label}: analytic ${analytic} vs numeric ${numeric}`);
}

/**
 * Checks dL/dX and dL/dθ from `backward` against central finite differences of `objective`.
 * Parameters that are not trainable (moving statistics) are skipped.
 */
function gradientCheck(layer: Layer, x: Matrix, training: boolean, rng: Random, h = 1e-6): void {
    const outputSize = layer.forward(x, training).cols;
    const g = randomMatrix(rng, x.rows, outputSize);
    objective(layer, x, g, training);
    const dx = layer.backward(g).clone();
    const grads = new Map<Parameter, Float64Array>(layer.parameters().map((p) => [p, p.grad.data.slice()]));

    const numeric = (values: Float64Array, i: number): number => {
        const original = values[i];
        values[i] = original + h;
        const plus = objective(layer, x, g, training);
        values[i] = original - h;
        const minus = objective(layer, x, g, training);
        values[i] = original;
        return (plus - minus) / (2 * h);
    };
    for (let i = 0; i < x.data.length; i++) assertGradClose(dx.data[i], numeric(x.data, i), `dX[${i}]`);
    for (const [param, grad] of grads) {
        if (!param.trainable) continue;
        for (let i = 0; i < grad.length; i++)
            assertGradClose(grad[i], numeric(param.value.data, i), `${param.name}[${i}]`);
    }
}

function built<T extends Layer>(layer: T, inputSize: number, seed = 1): T {
    if (layer.name === "") layer.name = "test";
    layer.build(inputSize, new Random(seed));
    return layer;
}

describe("Dense", () => {
    const rng = new Random(7);

    for (const act of ["linear", "tanh", "sigmoid", "softmax", "gelu", "elu"] as const) {
        it(`passes a finite-difference gradient check (${act})`, () => {
            const layer = built(new Dense({ units: 3, activation: act }), 5);
            gradientCheck(layer, randomMatrix(rng, 4, 5), true, rng);
        });
    }

    it("register-blocked kernels match naive products for every remainder shape (batch 0..17, sizes 1..17)", () => {
        const fuzz = new Random(11);
        const naive = (a: Matrix, b: Matrix, transposeA: boolean, transposeB: boolean): Matrix => {
            const m = transposeA ? a.cols : a.rows,
                k = transposeA ? a.rows : a.cols;
            const n = transposeB ? b.rows : b.cols;
            const out = new Matrix(m, n);
            for (let i = 0; i < m; i++)
                for (let j = 0; j < n; j++) {
                    let s = 0;
                    for (let p = 0; p < k; p++)
                        s += (transposeA ? a.get(p, i) : a.get(i, p)) * (transposeB ? b.get(j, p) : b.get(p, j));
                    out.set(i, j, s);
                }
            return out;
        };
        const close = (actual: Matrix, expected: Matrix, label: string) => {
            assert.deepEqual(actual.shape, expected.shape, label);
            for (let i = 0; i < actual.data.length; i++) {
                assert.ok(Math.abs(actual.data[i] - expected.data[i]) <= 1e-12, `${label}[${i}]`);
            }
        };
        for (let trial = 0; trial < 300; trial++) {
            const rows = fuzz.int(18),
                inputs = 1 + fuzz.int(17),
                units = 1 + fuzz.int(17);
            const useBias = trial % 3 !== 0;
            const layer = new Dense({ units, useBias, biasInitializer: { name: "randomNormal", stddev: 1 } });
            layer.build(inputs, fuzz.fork());
            const x = randomMatrix(fuzz, rows, inputs);
            const label = `rows ${rows}, inputs ${inputs}, units ${units}, bias ${useBias}`;
            const expected = naive(x, layer.kernel.value, false, false);
            if (useBias) expected.map((v, _r, c) => v + (layer.bias as Parameter).value.data[c], expected);
            close(layer.forward(x, true), expected, `forward ${label}`);
            const g = randomMatrix(fuzz, rows, units);
            layer.kernel.grad.fill(99);
            const dx = layer.backward(g);
            close(layer.kernel.grad, naive(x, g, true, false), `dW ${label}`);
            close(dx, naive(g, layer.kernel.value, false, true), `dx ${label}`);
            if (useBias) {
                const db = new Matrix(1, units);
                for (let i = 0; i < g.data.length; i++) db.data[i % units] += g.data[i];
                close((layer.bias as Parameter).grad, db, `db ${label}`);
            }
        }
    });

    it("passes a gradient check with L1 + L2 kernel regularization", () => {
        const layer = built(new Dense({ units: 4, activation: "tanh", kernelRegularizer: { l1: 0.03, l2: 0.05 } }), 3);
        gradientCheck(layer, randomMatrix(rng, 5, 3), true, rng);
    });

    it("passes a gradient check without bias, for batch sizes 1 and 3 (odd-row kernels)", () => {
        for (const rows of [1, 3]) {
            const layer = built(new Dense({ units: 9, activation: "swish", useBias: false }), 6);
            gradientCheck(layer, randomMatrix(rng, rows, 6), true, rng);
        }
    });

    it("computes x · kernel + bias exactly like a naive loop", () => {
        const layer = built(new Dense({ units: 11, biasInitializer: { name: "constant", value: 0.5 } }), 7);
        const x = randomMatrix(rng, 5, 7);
        const y = layer.forward(x, false);
        const w = layer.kernel.value;
        for (let i = 0; i < 5; i++) {
            for (let j = 0; j < 11; j++) {
                let s = 0;
                for (let p = 0; p < 7; p++) s += x.get(i, p) * w.get(p, j);
                assert.equal(y.get(i, j), s + 0.5);
            }
        }
    });

    it("reports its regularization loss (l1·Σ|w| + l2·Σw²)", () => {
        const layer = built(new Dense({ units: 2, kernelRegularizer: { l1: 0.1, l2: 0.2 } }), 2);
        layer.kernel.value.data.set([1, -2, 3, -4]);
        assert.ok(Math.abs(layer.regularizationLoss() - (0.1 * 10 + 0.2 * 30)) < 1e-12);
        assert.equal(built(new Dense({ units: 2 }), 2).regularizationLoss(), 0);
    });

    it("names parameters after the layer, even when renamed after build", () => {
        const layer = new Dense({ units: 2 });
        layer.build(3, new Random(1));
        layer.name = "dense_9";
        assert.deepEqual(
            layer.parameters().map((p) => p.name),
            ["dense_9/kernel", "dense_9/bias"],
        );
        assert.deepEqual(layer.kernel.value.shape, [3, 2]);
        assert.deepEqual(layer.bias?.value.shape, [1, 2]);
        assert.equal(layer.bias?.regularize, false);
    });

    it("uses glorotUniform kernels and zero biases by default", () => {
        const layer = built(new Dense({ units: 50 }), 50);
        const limit = Math.sqrt(6 / 100);
        assert.ok(layer.kernel.value.data.every((v) => Math.abs(v) <= limit));
        assert.ok(layer.bias?.value.data.every((v) => v === 0));
    });

    it("reuses output buffers per batch size", () => {
        const layer = built(new Dense({ units: 3, activation: "relu" }), 2);
        const a = layer.forward(randomMatrix(rng, 4, 2), false);
        const b = layer.forward(randomMatrix(rng, 4, 2), false);
        const c = layer.forward(randomMatrix(rng, 2, 2), false);
        assert.equal(a, b);
        assert.notEqual(a, c);
        assert.equal(layer.forward(randomMatrix(rng, 4, 2), false), a);
    });

    it("round-trips its config", () => {
        const layer = new Dense({
            name: "out",
            units: 3,
            activation: { name: "leakyRelu", alpha: 0.2 },
            useBias: false,
            kernelInitializer: "heNormal",
            kernelRegularizer: { l2: 0.01 },
        });
        const config = layer.getConfig();
        assert.deepEqual(config, {
            type: "dense",
            name: "out",
            units: 3,
            activation: { name: "leakyRelu", alpha: 0.2 },
            useBias: false,
            kernelInitializer: { name: "heNormal" },
            biasInitializer: { name: "zeros" },
            kernelRegularizer: { l1: 0, l2: 0.01 },
        });
        assert.deepEqual(layerFromConfig(config).getConfig(), config);
    });

    it("validates options with actionable messages", () => {
        assert.throws(() => new Dense({ units: 0 }), /"units" must be a positive integer, got 0/);
        assert.throws(() => new Dense({ units: 2, activation: "relux" as never }), /Unknown activation "relux"/);
        assert.throws(
            () => new Dense({ units: 2, kernelRegularizer: { l2: -1 } }),
            /"l2" must be a finite number >= 0/,
        );
        assert.throws(() => new Dense({ units: 2, bogus: 1 } as never), /unknown option "bogus"/);
        assert.throws(() => new Dense({ units: 2, name: "a/b" }), /must not contain "\/"/);
    });

    it("rejects wrong input sizes, unbuilt use and backward before forward", () => {
        assert.throws(() => new Dense({ units: 2, name: "d" }).forward(new Matrix(1, 3), false), /is not built/);
        const layer = built(new Dense({ units: 2, name: "d" }), 3);
        assert.throws(() => layer.backward(new Matrix(1, 2)), /backward\(\) called before forward\(\)/);
        assert.throws(
            () => layer.forward(new Matrix(1, 4), false),
            (e: unknown) =>
                e instanceof ShapeError && /Dense layer "d" expected input with 3 features, got 4/.test(e.message),
        );
        layer.forward(new Matrix(2, 3), true);
        assert.throws(() => layer.backward(new Matrix(3, 2)), /gradient is \[3, 2\], expected \[2, 2\]/);
        assert.throws(() => layer.build(4, new Random(1)), /already built for 3 input features/);
        assert.doesNotThrow(() => layer.build(3, new Random(1)));
    });
});

describe("BatchNormalization", () => {
    const rng = new Random(11);

    function randomized(options = {}): BatchNormalization {
        const layer = built(new BatchNormalization(options), 4);
        for (const p of layer.parameters()) if (p.trainable) p.value.map(() => rng.uniform(0.5, 1.5), p.value);
        return layer;
    }

    it("passes a gradient check in training mode (batch statistics)", () => {
        gradientCheck(randomized(), randomMatrix(rng, 6, 4, -2, 3), true, rng);
    });

    it("passes a gradient check without scale/center and in inference mode", () => {
        gradientCheck(randomized({ scale: false, center: false }), randomMatrix(rng, 5, 4), true, rng);
        const layer = randomized();
        layer.parameters()[2].value.data.set([0.1, -0.2, 0.3, 0]);
        layer.parameters()[3].value.data.set([2, 0.5, 1, 3]);
        gradientCheck(layer, randomMatrix(rng, 3, 4), false, rng);
    });

    it("normalizes each feature to zero mean and unit variance while training", () => {
        const layer = built(new BatchNormalization({ epsilon: 1e-9 }), 3);
        const y = layer.forward(randomMatrix(rng, 64, 3, 5, 9), true);
        for (let c = 0; c < 3; c++) {
            let mean = 0;
            let sq = 0;
            for (let r = 0; r < 64; r++) mean += y.get(r, c) / 64;
            for (let r = 0; r < 64; r++) sq += (y.get(r, c) - mean) ** 2 / 64;
            assert.ok(Math.abs(mean) < 1e-12 && Math.abs(sq - 1) < 1e-6, `feature ${c}: mean ${mean}, var ${sq}`);
        }
    });

    it("keeps a single-sample training batch out of the moving statistics", () => {
        const layer = built(new BatchNormalization({ momentum: 0.9 }), 2);
        layer.forward(
            Matrix.fromArray([
                [1, 10],
                [3, 20],
            ]),
            true,
        );
        const [, , movingMean, movingVariance] = layer.parameters();
        const meanBefore = movingMean.value.data.slice();
        const varianceBefore = movingVariance.value.data.slice();
        const y = layer.forward(Matrix.fromArray([[100, -100]]), true);
        assert.deepEqual(movingMean.value.data, meanBefore);
        assert.deepEqual(movingVariance.value.data, varianceBefore);
        assert.ok(y.data.every(Number.isFinite));
    });

    it("updates moving statistics with momentum and uses them at inference", () => {
        const layer = built(new BatchNormalization({ momentum: 0.9 }), 2);
        layer.name = "bn";
        const x = Matrix.fromArray([
            [1, 10],
            [3, 20],
        ]);
        layer.forward(x, true);
        const [gamma, beta, movingMean, movingVariance] = layer.parameters();
        assert.deepEqual(
            layer.parameters().map((p) => [p.name, p.trainable]),
            [
                ["bn/gamma", true],
                ["bn/beta", true],
                ["bn/movingMean", false],
                ["bn/movingVariance", false],
            ],
        );
        assert.ok(Math.abs(movingMean.value.data[0] - 0.2) < 1e-12 && Math.abs(movingMean.value.data[1] - 1.5) < 1e-12);
        assert.ok(Math.abs(movingVariance.value.data[0] - (0.9 + 0.1 * 1)) < 1e-12);
        assert.ok(Math.abs(movingVariance.value.data[1] - (0.9 + 0.1 * 25)) < 1e-12);

        gamma.value.data.set([2, 3]);
        beta.value.data.set([0.5, -1]);
        const y = layer.forward(Matrix.fromArray([[4, 5]]), false);
        const expected0 = (2 * (4 - 0.2)) / Math.sqrt(1 + 1e-3) + 0.5;
        const expected1 = (3 * (5 - 1.5)) / Math.sqrt(0.9 + 2.5 + 1e-3) - 1;
        assert.ok(Math.abs(y.get(0, 0) - expected0) < 1e-12 && Math.abs(y.get(0, 1) - expected1) < 1e-12);
    });

    it("round-trips its config and validates options", () => {
        const config = new BatchNormalization({ name: "bn", momentum: 0.9, epsilon: 1e-5, center: false }).getConfig();
        assert.deepEqual(config, {
            type: "batchNormalization",
            name: "bn",
            momentum: 0.9,
            epsilon: 1e-5,
            center: false,
            scale: true,
        });
        assert.deepEqual(layerFromConfig(config).getConfig(), config);
        assert.throws(() => batchNormalization({ momentum: 1 }), /"momentum" must be a number in \[0, 1\)/);
        assert.throws(() => batchNormalization({ epsilon: 0 }), /"epsilon" must be a finite number > 0/);
    });
});

describe("ActivationLayer", () => {
    const rng = new Random(3);

    for (const act of ["relu", "softmax", "gelu", { name: "leakyRelu", alpha: 0.3 }] as const) {
        it(`passes a gradient check (${typeof act === "string" ? act : act.name})`, () => {
            gradientCheck(built(new ActivationLayer({ activation: act }), 5), randomMatrix(rng, 3, 5), true, rng);
        });
    }

    it("applies the activation and round-trips its config", () => {
        const layer = built(activation("relu"), 2);
        assert.deepEqual(layer.forward(Matrix.fromArray([[-1, 2]]), false).toArray(), [[0, 2]]);
        assert.equal(layer.outputSize, 2);
        assert.deepEqual(layer.parameters(), []);
        const config = activation({ name: "elu", alpha: 0.5 }, { name: "act" }).getConfig();
        assert.deepEqual(config, { type: "activation", name: "act", activation: { name: "elu", alpha: 0.5 } });
        assert.deepEqual(layerFromConfig(config).getConfig(), config);
    });
});

describe("Dropout", () => {
    it("drops about `rate` of the inputs and rescales survivors by 1 / (1 - rate)", () => {
        const layer = built(new Dropout({ rate: 0.3 }), 50, 5);
        const x = Matrix.filled(400, 50, 1);
        const y = layer.forward(x, true);
        let zeros = 0;
        let sum = 0;
        for (const v of y.data) {
            if (v === 0) zeros++;
            else assert.ok(Math.abs(v - 1 / 0.7) < 1e-12);
            sum += v;
        }
        const n = y.data.length;
        assert.ok(Math.abs(zeros / n - 0.3) < 0.01, `dropped fraction ${zeros / n}`);
        assert.ok(Math.abs(sum / n - 1) < 0.02, `mean ${sum / n}`);
    });

    it("is the identity at inference and when rate is 0", () => {
        const x = new Matrix(3, 4).map((_, r, c) => r - c);
        assert.equal(built(dropout(0.5), 4).forward(x, false), x);
        assert.equal(built(dropout(0), 4).forward(x, true), x);
    });

    it("backpropagates through the same mask", () => {
        const layer = built(dropout(0.5), 6, 9);
        const x = Matrix.filled(5, 6, 2);
        const y = layer.forward(x, true).clone();
        const g = layer.backward(Matrix.filled(5, 6, 3));
        for (let i = 0; i < y.data.length; i++) assert.equal(g.data[i], y.data[i] === 0 ? 0 : 6);
    });

    it("draws masks from the layer's seeded Random", () => {
        const masks = [1, 1, 2].map((seed) =>
            built(dropout(0.5), 8, seed)
                .forward(Matrix.filled(4, 8, 1), true)
                .clone(),
        );
        assert.deepEqual(masks[0].data, masks[1].data);
        assert.notDeepEqual(masks[0].data, masks[2].data);
    });

    it("validates the rate", () => {
        assert.throws(() => dropout(1), /"rate" must be a number in \[0, 1\), got 1/);
        assert.throws(() => new Dropout({} as never), /"rate" is required/);
        assert.deepEqual(dropout(0.25, { name: "drop" }).getConfig(), { type: "dropout", name: "drop", rate: 0.25 });
    });
});

describe("layer factories and registry", () => {
    it("dense() accepts an activation or options as its second argument", () => {
        assert.equal(dense(4).activation.name, "linear");
        assert.equal(dense(4, "relu").activation.name, "relu");
        assert.equal(dense(4, { name: "leakyRelu", alpha: 0.2 }).activation.getConfig().alpha, 0.2);
        const named = dense(4, { name: "output", activation: "sigmoid", useBias: false });
        assert.equal(named.name, "output");
        assert.equal(named.activation.name, "sigmoid");
        assert.equal(named.useBias, false);
        assert.equal(dense(4, { name: "head" }).name, "head");
        assert.throws(() => dense(4, { units: 3 } as never), /pass "units" as the first argument only/);
    });

    it("lists built-in types and rebuilds layers from configs", () => {
        assert.deepEqual([...LAYER_TYPES], ["dense", "dropout", "batchNormalization", "activation"]);
        const layer = layerFromConfig({ type: "dropout", name: "d", rate: 0.1 });
        assert.ok(layer instanceof Dropout);
        assert.equal(layer.name, "d");
    });

    it("explains unknown types and config keys", () => {
        assert.throws(
            () => layerFromConfig({ type: "conv2d", name: "c" }),
            (e: unknown) =>
                e instanceof ValidationError && /Unknown layer type "conv2d". Known types: dense/.test(e.message),
        );
        assert.throws(() => layerFromConfig({ type: "dense", name: "d", units: 2, unit: 3 }), /unknown key "unit"/);
    });

    it("registers custom layer types", () => {
        registerLayer("identityForTest", (config) => {
            const layer = activation("linear");
            layer.name = config.name;
            return layer;
        });
        assert.equal(layerFromConfig({ type: "identityForTest", name: "x" }).name, "x");
        assert.throws(() => registerLayer("dense", () => dense(1)), /built-in layer type/);
    });
});
