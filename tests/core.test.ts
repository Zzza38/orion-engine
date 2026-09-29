import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { ShapeError, ValidationError } from "../src/core/errors.js";
import {
    add,
    addRowVector,
    argmaxRows,
    gatherRows,
    Matrix,
    matmul,
    matmulTransposeA,
    matmulTransposeB,
    multiply,
    scale,
    sliceRows,
    subtract,
    sumRows,
    transpose,
} from "../src/core/matrix.js";
import { Random } from "../src/core/random.js";

function naiveMatmul(a: number[][], b: number[][]): number[][] {
    return a.map((row) => b[0].map((_, j) => row.reduce((sum, v, k) => sum + v * b[k][j], 0)));
}

function randomMatrix(rng: Random, rows: number, cols: number): Matrix {
    return new Matrix(rows, cols).map(() => rng.uniform(-1, 1));
}

function assertClose(actual: number[][], expected: number[][], tolerance = 1e-12) {
    assert.equal(actual.length, expected.length);
    for (let r = 0; r < actual.length; r++) {
        assert.equal(actual[r].length, expected[r].length);
        for (let c = 0; c < actual[r].length; c++) {
            assert.ok(
                Math.abs(actual[r][c] - expected[r][c]) <= tolerance,
                `[${r}, ${c}]: ${actual[r][c]} vs ${expected[r][c]}`,
            );
        }
    }
}

describe("Matrix", () => {
    it("builds from nested arrays and round-trips", () => {
        const m = Matrix.fromArray([
            [1, 2, 3],
            [4, 5, 6],
        ]);
        assert.deepEqual(m.shape, [2, 3]);
        assert.equal(m.get(1, 2), 6);
        assert.deepEqual(m.toArray(), [
            [1, 2, 3],
            [4, 5, 6],
        ]);
    });

    it("treats a flat vector as a single row", () => {
        const m = Matrix.from([1, 2, 3]);
        assert.deepEqual(m.shape, [1, 3]);
    });

    it("returns the same instance from Matrix.from(matrix)", () => {
        const m = Matrix.zeros(2, 2);
        assert.equal(Matrix.from(m), m);
    });

    it("rejects ragged and non-numeric input", () => {
        assert.throws(() => Matrix.fromArray([[1, 2], [3]]), ShapeError);
        assert.throws(() => Matrix.fromArray([[1, Number.NaN]]), ValidationError);
        assert.throws(() => new Matrix(2, 2, [1, 2, 3]), ShapeError);
        assert.throws(() => new Matrix(-1, 2), ShapeError);
    });

    it("clone is independent", () => {
        const m = Matrix.fromArray([[1, 2]]);
        const c = m.clone();
        c.set(0, 0, 99);
        assert.equal(m.get(0, 0), 1);
    });

    it("row() is a live view", () => {
        const m = Matrix.fromArray([
            [1, 2],
            [3, 4],
        ]);
        m.row(1)[0] = 30;
        assert.equal(m.get(1, 0), 30);
    });
});

describe("matrix ops", () => {
    const rng = new Random(123);

    it("matmul variants agree with a naive implementation", () => {
        for (const [m, k, n] of [
            [1, 1, 1],
            [3, 4, 5],
            [7, 2, 9],
            [16, 16, 16],
        ]) {
            const a = randomMatrix(rng, m, k);
            const b = randomMatrix(rng, k, n);
            const expected = naiveMatmul(a.toArray(), b.toArray());
            assertClose(matmul(a, b).toArray(), expected);
            assertClose(matmulTransposeA(transpose(a), b).toArray(), expected);
            assertClose(matmulTransposeB(a, transpose(b)).toArray(), expected);
        }
    });

    it("matmul reuses an output buffer and overwrites stale values", () => {
        const a = Matrix.fromArray([[1, 2]]);
        const b = Matrix.fromArray([[3], [4]]);
        const out = Matrix.filled(1, 1, 1000);
        assert.equal(matmul(a, b, out), out);
        assert.equal(out.get(0, 0), 11);
    });

    it("rejects mismatched shapes and wrong output buffers", () => {
        assert.throws(() => matmul(Matrix.zeros(2, 3), Matrix.zeros(2, 3)), ShapeError);
        assert.throws(() => matmul(Matrix.zeros(2, 3), Matrix.zeros(3, 2), Matrix.zeros(3, 3)), ShapeError);
        assert.throws(() => add(Matrix.zeros(1, 2), Matrix.zeros(2, 1)), ShapeError);
    });

    it("element-wise ops", () => {
        const a = Matrix.fromArray([[1, 2, 3]]);
        const b = Matrix.fromArray([[4, 5, 6]]);
        assert.deepEqual(add(a, b).toArray(), [[5, 7, 9]]);
        assert.deepEqual(subtract(a, b).toArray(), [[-3, -3, -3]]);
        assert.deepEqual(multiply(a, b).toArray(), [[4, 10, 18]]);
        assert.deepEqual(scale(a, 2).toArray(), [[2, 4, 6]]);
    });

    it("broadcasts and reduces over the batch axis", () => {
        const a = Matrix.fromArray([
            [1, 2],
            [3, 4],
            [5, 6],
        ]);
        assert.deepEqual(addRowVector(a, Matrix.fromVector([10, 20])).toArray(), [
            [11, 22],
            [13, 24],
            [15, 26],
        ]);
        assert.deepEqual(sumRows(a).toArray(), [[9, 12]]);
        assert.throws(() => addRowVector(a, new Float64Array(3)), ShapeError);
    });

    it("gathers, slices, and argmaxes rows", () => {
        const a = Matrix.fromArray([
            [0, 9, 1],
            [7, 2, 3],
            [4, 5, 6],
        ]);
        assert.deepEqual(gatherRows(a, [2, 0]).toArray(), [
            [4, 5, 6],
            [0, 9, 1],
        ]);
        assert.deepEqual(sliceRows(a, 1, 10).toArray(), [
            [7, 2, 3],
            [4, 5, 6],
        ]);
        assert.deepEqual(Array.from(argmaxRows(a)), [1, 0, 2]);
        assert.throws(() => gatherRows(a, [3]), ShapeError);
    });
});

describe("Random", () => {
    it("is deterministic for a given seed", () => {
        const a = new Random(42);
        const b = new Random(42);
        for (let i = 0; i < 100; i++) assert.equal(a.next(), b.next());
    });

    it("differs across seeds", () => {
        assert.notEqual(new Random(1).next(), new Random(2).next());
    });

    it("produces uniform floats in [0, 1) with the right mean", () => {
        const rng = new Random(7);
        let sum = 0;
        for (let i = 0; i < 50_000; i++) {
            const v = rng.next();
            assert.ok(v >= 0 && v < 1);
            sum += v;
        }
        assert.ok(Math.abs(sum / 50_000 - 0.5) < 0.01);
    });

    it("produces standard normals", () => {
        const rng = new Random(9);
        const n = 50_000;
        let sum = 0;
        let sumSq = 0;
        for (let i = 0; i < n; i++) {
            const v = rng.normal();
            sum += v;
            sumSq += v * v;
        }
        assert.ok(Math.abs(sum / n) < 0.02);
        assert.ok(Math.abs(sumSq / n - 1) < 0.03);
    });

    it("truncatedNormal stays within two standard deviations", () => {
        const rng = new Random(11);
        for (let i = 0; i < 5_000; i++) assert.ok(Math.abs(rng.truncatedNormal(0, 3)) <= 6);
    });

    it("shuffle is a permutation", () => {
        const rng = new Random(5);
        const values = Array.from({ length: 50 }, (_, i) => i);
        const shuffled = rng.shuffle(values.slice());
        assert.notDeepEqual(shuffled, values);
        assert.deepEqual(
            shuffled.slice().sort((x, y) => x - y),
            values,
        );
    });

    it("fork yields an independent, deterministic stream", () => {
        const a = new Random(3).fork();
        const b = new Random(3).fork();
        assert.equal(a.next(), b.next());
    });
});
