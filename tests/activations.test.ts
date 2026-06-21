import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { Activation, ActivationDerivative, Loss } from "../src/index.js";

describe("Activation", () => {
    it("sigmoid approaches 0 and 1", () => {
        assert.ok(Activation.sigmoid(-20) < 0.001);
        assert.ok(Activation.sigmoid(20) > 0.999);
    });

    it("relu zeroes negative values", () => {
        assert.equal(Activation.relu(-1), 0);
        assert.equal(Activation.relu(2), 2);
    });

    it("softmax sums to 1", () => {
        const result = Activation.softmax([1, 2, 3]) as number[];
        const sum = result.reduce((a, b) => a + b, 0);
        assert.ok(Math.abs(sum - 1) < 1e-9);
    });
});

describe("ActivationDerivative", () => {
    it("linear derivative is 1", () => {
        assert.equal(ActivationDerivative.linear(5), 1);
    });

    it("sigmoid derivative at 0 is 0.25", () => {
        assert.ok(Math.abs(ActivationDerivative.sigmoid(0) - 0.25) < 1e-9);
    });
});

describe("Loss", () => {
    it("mse is zero for perfect predictions", () => {
        assert.equal(Loss.mse([1, 0], [1, 0]), 0);
    });

    it("mse gradient is zero for perfect predictions", () => {
        const grad = Loss.gradient("mse", [1, 0], [1, 0]);
        assert.deepEqual(grad, [0, 0]);
    });
});
