import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { NeuralNetwork } from "../src/index.js";

const XOR_INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]];
const XOR_TARGETS = [[0], [1], [1], [0]];
const TOLERANCE = 0.1;
const MAX_ATTEMPTS = 10;

function trainXor(): NeuralNetwork {
    const network = new NeuralNetwork();
    network.addLayer(2, "linear");
    network.addLayer(4, "sigmoid");
    network.addLayer(1, "sigmoid");
    network.train(XOR_INPUTS, XOR_TARGETS, 10000, 0.3, "crossEntropy");
    return network;
}

function xorErrors(network: NeuralNetwork): number[] {
    return XOR_INPUTS.map((input, i) =>
        Math.abs(network.runNetwork(input)[0] - XOR_TARGETS[i][0])
    );
}

describe("XOR", () => {
    it("learns all four patterns within tolerance", () => {
        let passed = false;

        for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            const network = trainXor();
            const errors = xorErrors(network);
            if (errors.every(err => err < TOLERANCE)) {
                passed = true;
                break;
            }
        }

        assert.ok(passed, `XOR did not converge within ${MAX_ATTEMPTS} attempts`);
    });

    it("backpropagate returns finite loss", () => {
        const network = new NeuralNetwork();
        network.addLayer(2, "linear");
        network.addLayer(2, "sigmoid");
        network.addLayer(1, "sigmoid");

        const loss = network.backpropagate([0, 1], [1], 0.1, "crossEntropy");
        assert.ok(Number.isFinite(loss));
    });
});
