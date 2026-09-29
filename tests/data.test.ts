import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { ShapeError, ValidationError } from "../src/core/errors.js";
import { Matrix } from "../src/core/matrix.js";
import { Random } from "../src/core/random.js";
import { argmax, MinMaxScaler, oneHot, StandardScaler, shuffleTogether, trainTestSplit } from "../src/data.js";

describe("oneHot / argmax", () => {
    it("one-hot encodes labels, inferring or checking the class count", () => {
        assert.deepEqual(oneHot([0, 2, 1]), [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ]);
        assert.deepEqual(oneHot([1], 4), [[0, 1, 0, 0]]);
        assert.deepEqual(oneHot(Int32Array.of(0)), [[1]]);
        assert.throws(() => oneHot([0, 3], 3), /labels\[1\] = 3 is not an integer class index in \[0, 3\)/);
        assert.throws(() => oneHot([0.5]), /not an integer class index/);
        assert.throws(() => oneHot([0], 0), /numClasses must be a positive integer/);
    });

    it("finds the index of the largest value of a vector, rows, or a Matrix", () => {
        assert.equal(argmax([0.1, 0.7, 0.2]), 1);
        assert.equal(argmax([3, 3]), 0);
        assert.deepEqual(
            argmax([
                [1, 2],
                [4, 3],
            ]),
            [1, 0],
        );
        assert.deepEqual(
            argmax(
                Matrix.fromArray([
                    [0, 0, 5],
                    [9, 0, 1],
                ]),
            ),
            [2, 0],
        );
        assert.deepEqual(argmax(oneHot([2, 0, 1])), [2, 0, 1]);
        assert.throws(() => argmax([]), /input is empty/);
    });
});

describe("trainTestSplit / shuffleTogether", () => {
    const x = Array.from({ length: 10 }, (_, i) => [i, i * 10]);
    const y = Array.from({ length: 10 }, (_, i) => i);

    it("splits with ceil(testSize·n) test samples, keeping x and y paired", () => {
        const { xTrain, xTest, yTrain, yTest } = trainTestSplit(x, y, { testSize: 0.25, seed: 1 });
        assert.equal(xTest.length, 3);
        assert.equal(xTrain.length, 7);
        xTrain.forEach((row, i) => {
            assert.equal(row[0], yTrain[i]);
        });
        xTest.forEach((row, i) => {
            assert.equal(row[0], yTest[i]);
        });
        assert.deepEqual(
            [...yTrain, ...yTest].sort((a, b) => a - b),
            y,
        );
        assert.notEqual(xTrain[0], x[yTrain[0]], "rows are copied");
    });

    it("is deterministic with a seed, and takes the last samples without shuffling", () => {
        assert.deepEqual(trainTestSplit(x, y, { seed: 3 }), trainTestSplit(x, y, { seed: 3 }));
        const ordered = trainTestSplit(x, y, { shuffle: false, testSize: 2 });
        assert.deepEqual(ordered.yTest, [8, 9]);
        assert.deepEqual(ordered.yTrain, [0, 1, 2, 3, 4, 5, 6, 7]);
    });

    it("supports Matrix inputs", () => {
        const { xTrain, xTest } = trainTestSplit(Matrix.fromArray(x), y, { seed: 2 });
        assert.ok(xTrain instanceof Matrix && xTest instanceof Matrix);
        assert.deepEqual([xTrain.rows, xTest.rows], [8, 2]);
    });

    it("validates its inputs", () => {
        assert.throws(() => trainTestSplit(x, y.slice(1)), ShapeError);
        assert.throws(
            () => trainTestSplit(x, y, { testSize: 1.5 }),
            /"testSize" must be a fraction in \(0, 1\) or a positive integer/,
        );
        assert.throws(() => trainTestSplit(x, y, { testSize: 10 }), /leaves no training samples/);
        assert.throws(() => trainTestSplit(x, y, { seed: 0.5 }), /seed must be an integer/);
    });

    it("shuffles x and y with one permutation", () => {
        const [xs, ys] = shuffleTogether(x, y, 42);
        xs.forEach((row, i) => {
            assert.equal(row[0], ys[i]);
        });
        assert.notDeepEqual(ys, y);
        assert.deepEqual(shuffleTogether(x, y, new Random(42)), [xs, ys]);
        const [xm] = shuffleTogether(Matrix.fromArray(x), y, 42);
        assert.deepEqual(xm.toArray(), xs);
    });
});

describe("StandardScaler", () => {
    const data = [
        [1, 10, 5],
        [2, 20, 5],
        [3, 60, 5],
    ];

    it("standardizes features to zero mean and unit (population) variance", () => {
        const scaler = new StandardScaler();
        const scaled = scaler.fitTransform(data);
        for (let c = 0; c < 2; c++) {
            const column = scaled.map((row) => row[c]);
            const mean = column.reduce((s, v) => s + v, 0) / 3;
            const variance = column.reduce((s, v) => s + (v - mean) ** 2, 0) / 3;
            assert.ok(Math.abs(mean) < 1e-12 && Math.abs(variance - 1) < 1e-12);
        }
        assert.deepEqual(
            scaled.map((row) => row[2]),
            [0, 0, 0],
            "constant features are only centered",
        );
        assert.deepEqual(scaler.mean, [2, 30, 5]);
        assert.equal(scaler.std[2], 1);
    });

    it("inverts, mirrors the input kind, and round-trips through JSON", () => {
        const scaler = new StandardScaler().fit(data);
        const back = scaler.inverseTransform(scaler.transform(data));
        back.flat().forEach((v, i) => {
            assert.ok(Math.abs(v - data.flat()[i]) < 1e-12);
        });
        assert.ok(scaler.transform(Matrix.fromArray(data)) instanceof Matrix);
        const one = scaler.transform([2, 30, 5]);
        assert.deepEqual(one, [0, 0, 0]);
        const restored = StandardScaler.fromJSON(JSON.parse(JSON.stringify(scaler)));
        assert.deepEqual(restored.transform(data), scaler.transform(data));
    });

    it("explains misuse", () => {
        assert.throws(() => new StandardScaler().transform(data), /call fit\(x\) first/);
        assert.throws(() => new StandardScaler().fit(data).transform([[1, 2]]), /fitted on 3 features, got 2/);
        assert.throws(() => new StandardScaler().fit([[1, Number.POSITIVE_INFINITY]]), /not a finite number/);
        assert.throws(() => StandardScaler.fromJSON({ type: "minMaxScaler" } as never), ValidationError);
    });
});

describe("MinMaxScaler", () => {
    const data = [
        [0, 10, 3],
        [5, 20, 3],
        [10, 30, 3],
    ];

    it("rescales each feature to the feature range", () => {
        const scaler = new MinMaxScaler({ featureRange: [-1, 1] });
        assert.deepEqual(scaler.fitTransform(data), [
            [-1, -1, -1],
            [0, 0, -1],
            [1, 1, -1],
        ]);
        assert.deepEqual(new MinMaxScaler().fitTransform(data)[1], [0.5, 0.5, 0]);
        assert.deepEqual(scaler.dataMin, [0, 10, 3]);
        assert.deepEqual(scaler.dataMax, [10, 30, 3]);
    });

    it("inverts and round-trips through JSON", () => {
        const scaler = new MinMaxScaler().fit(data);
        assert.deepEqual(scaler.inverseTransform(scaler.transform(data)), data);
        const restored = MinMaxScaler.fromJSON(JSON.parse(JSON.stringify(scaler)));
        assert.deepEqual(restored.transform([[2.5, 15, 3]]), [[0.25, 0.25, 0]]);
        assert.throws(
            () => new MinMaxScaler({ featureRange: [1, 1] }),
            /"featureRange" must be \[min, max\] with min < max/,
        );
    });
});
