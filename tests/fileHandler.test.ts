import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, it } from "node:test";
import {
    loadNetwork,
    NeuralNetwork,
    writeNetwork,
    writeNetworkToFile,
    loadNetworkFromFile,
} from "../src/index.js";

const DOC_EXAMPLE = `2:relu:2:swish
0.71:-0.2:0.19|-1.82:0.95:0.97`;

describe("fileHandler", () => {
    it("loads the documented .onn example", () => {
        const network = loadNetwork(DOC_EXAMPLE);
        assert.equal(network.layers.length, 2);
        assert.equal(network.layers[0].neurons.length, 2);
        assert.equal(network.layers[1].neurons.length, 2);
        assert.equal(network.layers[0].activation, "relu");
        assert.equal(network.layers[1].activation, "swish");
    });

    it("roundtrips serialize and deserialize", () => {
        const original = new NeuralNetwork(true);
        original.addLayer(2, "linear");
        original.addLayer(2, "sigmoid");
        original.addLayer(1, "sigmoid");
        original.loadWeightsAndBiases(1, [[0.5, -0.5], [1, 0]], [0.1, -0.1]);
        original.loadWeightsAndBiases(2, [[0.25, 0.75]], [0.5]);

        const input = [1, 0];
        const before = original.runNetwork(input);

        const loaded = loadNetwork(writeNetwork(original));
        const after = loaded.runNetwork(input);

        assert.deepEqual(after, before);
    });

    it("roundtrips through the filesystem", () => {
        const dir = fs.mkdtempSync(path.join(os.tmpdir(), "orion-test-"));
        const filePath = path.join(dir, "model.onn");

        try {
            const network = new NeuralNetwork(true);
            network.addLayer(2, "linear");
            network.addLayer(1, "sigmoid");
            network.loadWeightsAndBiases(1, [[0.3, 0.7]], [0.2]);

            const input = [0.5, 0.5];
            const before = network.runNetwork(input);

            assert.equal(writeNetworkToFile(network, filePath), true);
            const loaded = loadNetworkFromFile(filePath);
            const after = loaded.runNetwork(input);

            assert.deepEqual(after, before);
        } finally {
            fs.rmSync(dir, { recursive: true, force: true });
        }
    });
});
