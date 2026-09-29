import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { METRIC_NAMES, getMetric } from "../src/metrics.js";
import { Matrix, sliceRows } from "../src/core/matrix.js";
import { Random } from "../src/core/random.js";
import { ShapeError, ValidationError } from "../src/core/errors.js";
import type { Metric, MetricIdentifier } from "../src/core/types.js";

function assertClose(actual: number, expected: number, tol = 1e-12, label = ""): void {
    assert.ok(Math.abs(actual - expected) <= tol, `${label} expected ${expected}, got ${actual}`);
}

function randomMatrix(rng: Random, rows: number, cols: number, lo: number, hi: number): Matrix {
    const m = new Matrix(rows, cols);
    for (let i = 0; i < m.data.length; i++) m.data[i] = rng.uniform(lo, hi);
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

describe("metrics: registry", () => {
    it("exposes every metric name and resolves each one", () => {
        assert.equal(METRIC_NAMES.length, 7);
        for (const name of METRIC_NAMES) assert.equal(getMetric(name).name, name);
    });

    it("resolves aliases to canonical names", () => {
        assert.equal(getMetric("mse").name, "meanSquaredError");
        assert.equal(getMetric("mae").name, "meanAbsoluteError");
        assert.equal(getMetric("rmse").name, "rootMeanSquaredError");
    });

    it("returns an existing instance as-is", () => {
        const custom: Metric = { name: "accuracy", compute: () => 1 };
        assert.equal(getMetric(custom), custom);
        const builtin = getMetric("accuracy");
        assert.equal(getMetric(builtin), builtin);
    });

    it("rejects unknown names and lists the valid ones", () => {
        assert.throws(() => getMetric("f1" as MetricIdentifier), (err: unknown) => {
            assert.ok(err instanceof ValidationError);
            for (const name of METRIC_NAMES) assert.ok(err.message.includes(name), `message lists ${name}`);
            return true;
        });
        assert.throws(() => getMetric(null as never), ValidationError);
        assert.throws(() => getMetric({ name: "accuracy" } as never), ValidationError);
    });
});

describe("metrics: accuracy", () => {
    const binaryPred = Matrix.fromArray([[0.9], [0.4], [0.6], [0.2]]);
    const binaryTarget = Matrix.fromArray([[1], [0], [0], [0]]);
    const multiPred = Matrix.fromArray([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]);
    const sparseTarget = Matrix.fromArray([[1], [1], [1]]);

    it("binaryAccuracy thresholds at 0.5", () => {
        assertClose(getMetric("binaryAccuracy").compute(binaryPred, binaryTarget), 0.75);
        // Element-wise over multiple outputs.
        const pred = Matrix.fromArray([[0.9, 0.1], [0.51, 0.49]]);
        const target = Matrix.fromArray([[1, 1], [1, 0]]);
        assertClose(getMetric("binaryAccuracy").compute(pred, target), 0.75);
        // 0.5 itself is class 0.
        assertClose(getMetric("binaryAccuracy").compute(Matrix.fromArray([[0.5]]), Matrix.fromArray([[0]])), 1);
    });

    it("categoricalAccuracy compares argmax", () => {
        const target = oneHot(sparseTarget, 2);
        assertClose(getMetric("categoricalAccuracy").compute(multiPred, target), 2 / 3);
        // Probability targets use their argmax too.
        const soft = Matrix.fromArray([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]]);
        const pred = Matrix.fromArray([[0, 1, 0], [0, 0, 1]]);
        assertClose(getMetric("categoricalAccuracy").compute(pred, soft), 0.5);
    });

    it("sparseCategoricalAccuracy compares argmax against indices", () => {
        assertClose(getMetric("sparseCategoricalAccuracy").compute(multiPred, sparseTarget), 2 / 3);
    });

    it("accuracy auto-selects the variant from the shapes", () => {
        const accuracy = getMetric("accuracy");
        assertClose(accuracy.compute(binaryPred, binaryTarget), 0.75, 0, "binary");
        assertClose(accuracy.compute(multiPred, sparseTarget), 2 / 3, 0, "sparse");
        assertClose(accuracy.compute(multiPred, oneHot(sparseTarget, 2)), 2 / 3, 0, "categorical");
    });

    it("random guessing scores about 1/k", () => {
        const rng = new Random(31);
        const n = 20000;
        const binary = getMetric("accuracy").compute(randomMatrix(rng, n, 1, 0, 1), randomIndices(rng, n, 2));
        assert.ok(Math.abs(binary - 0.5) < 0.02, `binary ${binary}`);
        const four = getMetric("accuracy").compute(randomMatrix(rng, n, 4, 0, 1), randomIndices(rng, n, 4));
        assert.ok(Math.abs(four - 0.25) < 0.02, `4-class ${four}`);
    });

    it("validates shapes and indices", () => {
        assert.throws(() => getMetric("binaryAccuracy").compute(binaryPred, new Matrix(3, 1)), ShapeError);
        assert.throws(() => getMetric("categoricalAccuracy").compute(multiPred, sparseTarget), ShapeError);
        assert.throws(() => getMetric("sparseCategoricalAccuracy").compute(multiPred, new Matrix(3, 2)), ShapeError);
        assert.throws(() => getMetric("accuracy").compute(binaryPred, new Matrix(4, 2)), ShapeError);
        assert.throws(() => getMetric("accuracy").compute(new Matrix(0, 2), new Matrix(0, 1)), ShapeError);
        assert.throws(
            () => getMetric("sparseCategoricalAccuracy").compute(multiPred, Matrix.fromArray([[0], [2], [1]])),
            ValidationError,
        );
        assert.throws(
            () => getMetric("accuracy").compute(multiPred, Matrix.fromArray([[0], [0.5], [1]])),
            ValidationError,
        );
    });
});

describe("metrics: regression", () => {
    const p = Matrix.fromArray([[1, 2], [3, 4]]);
    const y = Matrix.fromArray([[1, 1], [1, 1]]);

    it("meanSquaredError, meanAbsoluteError, rootMeanSquaredError", () => {
        assertClose(getMetric("meanSquaredError").compute(p, y), 3.5);
        assertClose(getMetric("mse").compute(p, y), 3.5);
        assertClose(getMetric("meanAbsoluteError").compute(p, y), 1.5);
        assertClose(getMetric("mae").compute(p, y), 1.5);
        assertClose(getMetric("rootMeanSquaredError").compute(p, y), Math.sqrt(3.5));
        assertClose(getMetric("rmse").compute(y, y), 0);
    });

    it("rmse is the square root of mse on random data", () => {
        const rng = new Random(8);
        const pred = randomMatrix(rng, 50, 3, -1, 1);
        const target = randomMatrix(rng, 50, 3, -1, 1);
        assertClose(getMetric("rmse").compute(pred, target) ** 2, getMetric("mse").compute(pred, target), 1e-12);
        // E[(U1 - U2)^2] for U ~ U(-1, 1) is 2/3.
        const big = getMetric("mse").compute(randomMatrix(rng, 5000, 4, -1, 1), randomMatrix(rng, 5000, 4, -1, 1));
        assert.ok(Math.abs(big - 2 / 3) < 0.03, `mse ${big}`);
    });

    it("throws ShapeError on mismatched shapes", () => {
        for (const name of ["meanSquaredError", "meanAbsoluteError", "rootMeanSquaredError"] as const) {
            assert.throws(() => getMetric(name).compute(p, new Matrix(2, 3)), ShapeError, name);
        }
    });
});

describe("metrics: batch decomposition", () => {
    it("size-weighted batch means equal the full-dataset value", () => {
        const rng = new Random(123);
        const n = 103;
        const pred = randomMatrix(rng, n, 3, 0, 1);
        const indices = randomIndices(rng, n, 3);
        const cases: [MetricIdentifier, Matrix][] = [
            ["accuracy", indices],
            ["categoricalAccuracy", oneHot(indices, 3)],
            ["binaryAccuracy", randomMatrix(rng, n, 3, 0, 1)],
            ["mse", randomMatrix(rng, n, 3, 0, 1)],
            ["mae", randomMatrix(rng, n, 3, 0, 1)],
        ];
        for (const [id, target] of cases) {
            const metric = getMetric(id);
            let weighted = 0;
            for (let start = 0; start < n; start += 16) {
                const end = Math.min(n, start + 16);
                weighted += metric.compute(sliceRows(pred, start, end), sliceRows(target, start, end)) * (end - start);
            }
            assertClose(weighted / n, metric.compute(pred, target), 1e-12, String(id));
        }
    });

    it("is deterministic for the same seed", () => {
        const run = (seed: number) => {
            const rng = new Random(seed);
            return getMetric("accuracy").compute(randomMatrix(rng, 500, 5, 0, 1), randomIndices(rng, 500, 5));
        };
        assert.equal(run(42), run(42));
    });
});
