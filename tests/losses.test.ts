import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { LOSS_NAMES, getLoss } from "../src/losses.js";
import { getActivation } from "../src/activations.js";
import { Matrix } from "../src/core/matrix.js";
import { Random } from "../src/core/random.js";
import { ShapeError, ValidationError } from "../src/core/errors.js";
import type { Loss, LossIdentifier } from "../src/core/types.js";

function randomMatrix(rng: Random, rows: number, cols: number, lo: number, hi: number): Matrix {
    const m = new Matrix(rows, cols);
    for (let i = 0; i < m.data.length; i++) m.data[i] = rng.uniform(lo, hi);
    return m;
}

/** Rows of positive values that each sum to 1. */
function randomDistribution(rng: Random, rows: number, cols: number): Matrix {
    const m = randomMatrix(rng, rows, cols, 0.05, 1);
    for (let r = 0; r < rows; r++) {
        const row = m.row(r);
        const sum = row.reduce((s, v) => s + v, 0);
        for (let c = 0; c < cols; c++) row[c] /= sum;
    }
    return m;
}

function randomIndices(rng: Random, rows: number, classes: number): Matrix {
    const m = new Matrix(rows, 1);
    for (let r = 0; r < rows; r++) m.data[r] = rng.int(classes);
    return m;
}

function oneHot(indices: Matrix, classes: number): Matrix {
    const m = new Matrix(indices.rows, classes);
    for (let r = 0; r < indices.rows; r++) m.set(r, indices.data[r], 1);
    return m;
}

function assertClose(actual: number, expected: number, tol = 1e-12, label = ""): void {
    const scale = Math.max(1, Math.abs(expected));
    assert.ok(
        Math.abs(actual - expected) <= tol * scale,
        `${label} expected ${expected}, got ${actual} (diff ${Math.abs(actual - expected)})`,
    );
}

function assertAllClose(actual: Matrix, expected: Matrix, tol = 1e-12, label = ""): void {
    assert.deepEqual(actual.shape, expected.shape, `${label} shape`);
    for (let i = 0; i < actual.data.length; i++) assertClose(actual.data[i], expected.data[i], tol, `${label}[${i}]`);
}

function finiteDifferenceCheck(loss: Loss, prediction: Matrix, target: Matrix, label: string): void {
    const eps = 1e-5;
    const analytic = loss.gradient(prediction, target);
    assert.deepEqual(analytic.shape, prediction.shape);
    for (let i = 0; i < prediction.data.length; i++) {
        const plus = prediction.clone();
        const minus = prediction.clone();
        plus.data[i] += eps;
        minus.data[i] -= eps;
        const numeric = (loss.compute(plus, target) - loss.compute(minus, target)) / (2 * eps);
        assertClose(analytic.data[i], numeric, 1e-7, `${label} dp[${i}]`);
    }
}

describe("losses: registry", () => {
    it("exposes every loss name and resolves each one", () => {
        assert.equal(LOSS_NAMES.length, 6);
        for (const name of LOSS_NAMES) {
            const loss = getLoss(name);
            assert.equal(loss.name, name);
            assert.equal(loss.getConfig().name, name);
        }
    });

    it("resolves aliases to canonical names", () => {
        const aliases: Record<string, string> = {
            mse: "meanSquaredError",
            mae: "meanAbsoluteError",
            bce: "binaryCrossentropy",
            cce: "categoricalCrossentropy",
            scce: "sparseCategoricalCrossentropy",
        };
        for (const [alias, name] of Object.entries(aliases)) {
            assert.equal(getLoss(alias as LossIdentifier).name, name);
            assert.equal(getLoss({ name: alias } as never).name, name, "aliases also work inside configs");
        }
    });

    it("returns an existing instance as-is", () => {
        const loss = getLoss("mse");
        assert.equal(getLoss(loss), loss);
    });

    it("round-trips configs", () => {
        for (const id of [...LOSS_NAMES, { name: "huber", delta: 2.5 } as const]) {
            const original = getLoss(id);
            const restored = getLoss(JSON.parse(JSON.stringify(original.getConfig())));
            assert.deepEqual(restored.getConfig(), original.getConfig());
        }
        assert.deepEqual(getLoss("huber").getConfig(), { name: "huber", delta: 1 });
        assert.deepEqual(getLoss({ name: "huber", delta: 2.5 }).getConfig(), { name: "huber", delta: 2.5 });
    });

    it("rejects unknown names and bad parameters", () => {
        assert.throws(() => getLoss("hinge" as never), (err: unknown) => {
            assert.ok(err instanceof ValidationError);
            for (const name of LOSS_NAMES) assert.ok(err.message.includes(name), `message lists ${name}`);
            return true;
        });
        assert.throws(() => getLoss({ name: "nope" } as never), ValidationError);
        assert.throws(() => getLoss(undefined as never), ValidationError);
        assert.throws(() => getLoss({ name: "huber", delta: 0 }), ValidationError);
        assert.throws(() => getLoss({ name: "huber", delta: -1 }), ValidationError);
        assert.throws(() => getLoss({ name: "huber", delta: "1" }), ValidationError);
        assert.throws(() => getLoss({ name: "meanSquaredError", delta: 1 }), ValidationError);
    });
});

describe("losses: known values", () => {
    const p = Matrix.fromArray([[1, 2], [3, 4]]);
    const y = Matrix.fromArray([[1, 1], [1, 1]]);

    it("meanSquaredError", () => {
        assertClose(getLoss("mse").compute(p, y), (0 + 1 + 4 + 9) / 4);
        assert.equal(getLoss("mse").compute(y, y), 0);
    });

    it("meanAbsoluteError", () => {
        assertClose(getLoss("mae").compute(p, y), (0 + 1 + 2 + 3) / 4);
    });

    it("huber", () => {
        // errors 0, 1, 2, 3 with delta 1: 0, 0.5, 1.5, 2.5
        assertClose(getLoss("huber").compute(p, y), 4.5 / 4);
        // delta 2: 0, 0.5, 2, 4
        assertClose(getLoss({ name: "huber", delta: 2 }).compute(p, y), 6.5 / 4);
    });

    it("binaryCrossentropy averages over units and batch", () => {
        const loss = getLoss("bce");
        assertClose(
            loss.compute(Matrix.fromArray([[0.9], [0.2]]), Matrix.fromArray([[1], [0]])),
            -(Math.log(0.9) + Math.log(0.8)) / 2,
        );
        const two = loss.compute(Matrix.fromArray([[0.9, 0.3]]), Matrix.fromArray([[1, 0]]));
        assertClose(two, -(Math.log(0.9) + Math.log(0.7)) / 2);
    });

    it("binaryCrossentropy clamps probabilities to [1e-7, 1 - 1e-7]", () => {
        const loss = getLoss("bce");
        const pred = Matrix.fromArray([[0], [1]]);
        const target = Matrix.fromArray([[1], [0]]);
        assertClose(loss.compute(pred, target), -Math.log(1e-7), 1e-9);
        for (const v of loss.gradient(pred, target).data) assert.ok(Number.isFinite(v) && v !== 0);
    });

    it("categoricalCrossentropy sums over classes and averages over batch", () => {
        const pred = Matrix.fromArray([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]);
        const target = Matrix.fromArray([[1, 0, 0], [0, 0, 1]]);
        assertClose(getLoss("cce").compute(pred, target), -(Math.log(0.7) + Math.log(0.8)) / 2);
        const soft = Matrix.fromArray([[0.5, 0.5, 0], [0, 0.25, 0.75]]);
        assertClose(
            getLoss("cce").compute(pred, soft),
            -(0.5 * Math.log(0.7) + 0.5 * Math.log(0.2) + 0.25 * Math.log(0.1) + 0.75 * Math.log(0.8)) / 2,
        );
        assert.ok(Number.isFinite(getLoss("cce").compute(Matrix.fromArray([[0, 1]]), Matrix.fromArray([[1, 0]]))));
    });

    it("sparseCategoricalCrossentropy matches categorical with one-hot targets", () => {
        const pred = Matrix.fromArray([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]);
        const indices = Matrix.fromArray([[0], [2]]);
        assertClose(getLoss("scce").compute(pred, indices), -(Math.log(0.7) + Math.log(0.8)) / 2);
        assertClose(getLoss("scce").compute(pred, indices), getLoss("cce").compute(pred, oneHot(indices, 3)));
        assertAllClose(
            getLoss("scce").gradient(pred, indices),
            getLoss("cce").gradient(pred, oneHot(indices, 3)),
        );
    });
});

describe("losses: gradients (central finite differences)", () => {
    const rng = new Random(2024);

    it("meanSquaredError", () => {
        finiteDifferenceCheck(getLoss("mse"), randomMatrix(rng, 4, 3, -2, 2), randomMatrix(rng, 4, 3, -2, 2), "mse");
    });

    it("meanAbsoluteError (away from the kink)", () => {
        const target = randomMatrix(rng, 4, 3, -2, 2);
        const pred = target.clone();
        for (let i = 0; i < pred.data.length; i++) pred.data[i] += (i % 2 === 0 ? 1 : -1) * rng.uniform(0.1, 1);
        finiteDifferenceCheck(getLoss("mae"), pred, target, "mae");
    });

    for (const delta of [1, 0.5, 2]) {
        it(`huber (delta ${delta}), both regimes`, () => {
            const loss = getLoss({ name: "huber", delta });
            const target = randomMatrix(rng, 5, 4, -1, 1);
            const pred = target.clone();
            for (let i = 0; i < pred.data.length; i++) {
                // Alternate quadratic (|e| < delta) and linear (|e| > delta) regimes, away from |e| = delta.
                const magnitude = i % 2 === 0 ? rng.uniform(0.05, 0.9) * delta : rng.uniform(1.1, 3) * delta;
                pred.data[i] += (i % 3 === 0 ? -1 : 1) * magnitude;
            }
            finiteDifferenceCheck(loss, pred, target, `huber(${delta})`);
        });
    }

    it("binaryCrossentropy (hard and soft targets)", () => {
        const pred = randomMatrix(rng, 4, 3, 0.05, 0.95);
        const hard = new Matrix(4, 3);
        for (let i = 0; i < hard.data.length; i++) hard.data[i] = rng.int(2);
        finiteDifferenceCheck(getLoss("bce"), pred, hard, "bce hard");
        finiteDifferenceCheck(getLoss("bce"), pred, randomMatrix(rng, 4, 3, 0, 1), "bce soft");
    });

    it("categoricalCrossentropy (one-hot and probability targets)", () => {
        const pred = randomMatrix(rng, 4, 5, 0.05, 0.95);
        finiteDifferenceCheck(getLoss("cce"), pred, oneHot(randomIndices(rng, 4, 5), 5), "cce one-hot");
        finiteDifferenceCheck(getLoss("cce"), pred, randomDistribution(rng, 4, 5), "cce soft");
    });

    it("sparseCategoricalCrossentropy", () => {
        const pred = randomMatrix(rng, 4, 5, 0.05, 0.95);
        finiteDifferenceCheck(getLoss("scce"), pred, randomIndices(rng, 4, 5), "scce");
    });
});

describe("losses: fused gradients", () => {
    const rng = new Random(77);

    function checkFused(activationName: "sigmoid" | "softmax", loss: Loss, z: Matrix, target: Matrix): void {
        const activation = getActivation(activationName);
        const p = activation.forward(z);
        const composed = activation.backward(z, p, loss.gradient(p, target));
        const fused = loss.fusedGradient?.(activationName, p, target);
        assert.ok(fused, `${loss.name} has a fused form for ${activationName}`);
        assertAllClose(fused, composed, 1e-9, `${activationName}+${loss.name}`);

        const out = new Matrix(p.rows, p.cols);
        assert.equal(loss.fusedGradient?.(activationName, p, target, out), out);
        assertAllClose(out, fused, 0);
    }

    it("sigmoid + binaryCrossentropy = (p - y) / (batch·units)", () => {
        const z = randomMatrix(rng, 6, 3, -3, 3);
        const y = new Matrix(6, 3);
        for (let i = 0; i < y.data.length; i++) y.data[i] = rng.int(2);
        checkFused("sigmoid", getLoss("bce"), z, y);
        checkFused("sigmoid", getLoss("bce"), z, randomMatrix(rng, 6, 3, 0, 1));

        const p = getActivation("sigmoid").forward(z);
        const fused = getLoss("bce").fusedGradient?.("sigmoid", p, y);
        assert.ok(fused);
        for (let i = 0; i < p.data.length; i++) assertClose(fused.data[i], (p.data[i] - y.data[i]) / 18);
    });

    it("softmax + categoricalCrossentropy = (p - y) / batch", () => {
        const z = randomMatrix(rng, 5, 4, -3, 3);
        checkFused("softmax", getLoss("cce"), z, oneHot(randomIndices(rng, 5, 4), 4));
        const soft = randomDistribution(rng, 5, 4);
        checkFused("softmax", getLoss("cce"), z, soft);

        const p = getActivation("softmax").forward(z);
        const fused = getLoss("cce").fusedGradient?.("softmax", p, soft);
        assert.ok(fused);
        for (let i = 0; i < p.data.length; i++) assertClose(fused.data[i], (p.data[i] - soft.data[i]) / 5);
    });

    it("softmax + categoricalCrossentropy stays exact when target rows do not sum to 1", () => {
        const z = randomMatrix(rng, 5, 4, -3, 3);
        // Unnormalized soft targets, multi-hot rows, an all-zero row and a row summing to 2.
        checkFused("softmax", getLoss("cce"), z, randomMatrix(rng, 5, 4, 0, 1));
        const multiHot = Matrix.fromArray([
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [2, 0, 0, 0],
            [0.2, 0, 0, 0.3],
        ]);
        checkFused("softmax", getLoss("cce"), z, multiHot);
        // A 1-unit softmax is constant (always 1), so its gradient must be exactly 0.
        const one = getActivation("softmax").forward(randomMatrix(rng, 3, 1, -2, 2));
        const fused = getLoss("cce").fusedGradient?.("softmax", one, Matrix.fromArray([[1], [0], [0.5]]));
        assert.deepEqual(Array.from(fused?.data ?? []), [0, 0, 0]);
    });

    it("softmax + sparseCategoricalCrossentropy = (p - onehot(y)) / batch", () => {
        const z = randomMatrix(rng, 5, 4, -3, 3);
        const indices = randomIndices(rng, 5, 4);
        checkFused("softmax", getLoss("scce"), z, indices);

        const p = getActivation("softmax").forward(z);
        const fused = getLoss("scce").fusedGradient?.("softmax", p, indices);
        const expected = oneHot(indices, 4);
        assert.ok(fused);
        for (let i = 0; i < p.data.length; i++) assertClose(fused.data[i], (p.data[i] - expected.data[i]) / 5);
    });

    it("fused gradients may write in place over the prediction", () => {
        const p = getActivation("softmax").forward(randomMatrix(rng, 3, 4, -2, 2));
        const indices = randomIndices(rng, 3, 4);
        const expected = getLoss("scce").fusedGradient?.("softmax", p, indices);
        assert.ok(expected);
        const buffer = p.clone();
        getLoss("scce").fusedGradient?.("softmax", buffer, indices, buffer);
        assertAllClose(buffer, expected, 0);
    });

    it("returns null for pairs without a closed form", () => {
        const p = Matrix.fromArray([[0.3, 0.7]]);
        const y = Matrix.fromArray([[0, 1]]);
        assert.equal(getLoss("mse").fusedGradient?.("linear", p, y), null);
        assert.equal(getLoss("mse").fusedGradient?.("softmax", p, y), null);
        assert.equal(getLoss("mae").fusedGradient?.("sigmoid", p, y), null);
        assert.equal(getLoss("huber").fusedGradient?.("sigmoid", p, y), null);
        assert.equal(getLoss("bce").fusedGradient?.("softmax", p, y), null);
        assert.equal(getLoss("bce").fusedGradient?.("tanh", p, y), null);
        assert.equal(getLoss("cce").fusedGradient?.("sigmoid", p, y), null);
        assert.equal(getLoss("scce").fusedGradient?.("sigmoid", p, Matrix.fromArray([[1]])), null);
    });
});

describe("losses: buffers and validation", () => {
    it("writes gradients into a provided buffer", () => {
        const rng = new Random(5);
        const pred = randomMatrix(rng, 3, 4, 0.1, 0.9);
        const dense = randomDistribution(rng, 3, 4);
        const sparse = randomIndices(rng, 3, 4);
        for (const name of LOSS_NAMES) {
            const loss = getLoss(name);
            const target = name === "sparseCategoricalCrossentropy" ? sparse : dense;
            const expected = loss.gradient(pred, target);
            const out = Matrix.filled(3, 4, 123);
            assert.equal(loss.gradient(pred, target, out), out);
            assertAllClose(out, expected, 0, name);
            assert.throws(() => loss.gradient(pred, target, new Matrix(4, 3)), ShapeError, name);
        }
    });

    it("throws ShapeError on mismatched shapes", () => {
        const pred = new Matrix(3, 2).fill(0.5);
        for (const name of LOSS_NAMES) {
            const loss = getLoss(name);
            assert.throws(() => loss.compute(pred, new Matrix(2, 2)), ShapeError, name);
            assert.throws(() => loss.gradient(pred, new Matrix(3, 3)), ShapeError, name);
            assert.throws(() => loss.compute(new Matrix(0, 2), new Matrix(0, 2)), ShapeError, name);
        }
        assert.throws(() => getLoss("bce").fusedGradient?.("sigmoid", pred, new Matrix(3, 1)), ShapeError);
        assert.throws(() => getLoss("cce").fusedGradient?.("softmax", pred, new Matrix(3, 1)), ShapeError);
        assert.throws(() => getLoss("scce").fusedGradient?.("softmax", pred, new Matrix(3, 2)), ShapeError);
        assert.throws(() => getLoss("scce").compute(pred, new Matrix(3, 2)), ShapeError);
    });

    it("validates sparse class indices", () => {
        const scce = getLoss("scce");
        const pred = new Matrix(2, 3).fill(1 / 3);
        assert.throws(() => scce.compute(pred, Matrix.fromArray([[0], [3]])), ValidationError);
        assert.throws(() => scce.compute(pred, Matrix.fromArray([[-1], [0]])), ValidationError);
        assert.throws(() => scce.gradient(pred, Matrix.fromArray([[0.5], [1]])), ValidationError);
        assert.throws(() => scce.fusedGradient?.("softmax", pred, Matrix.fromArray([[0], [Number.NaN]])), ValidationError);
        assertClose(scce.compute(pred, Matrix.fromArray([[0], [2]])), Math.log(3));
    });
});
