import path from "node:path";
import { fileURLToPath } from "node:url";
import {
    loadNetwork,
    loadNetworkFromFile,
    NeuralNetwork,
    writeNetwork,
    writeNetworkToFile,
} from "../src/index.js";

const modelPath = path.join(path.dirname(fileURLToPath(import.meta.url)), "xor.onn");

const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
const targets = [[0], [1], [1], [0]];

const network = new NeuralNetwork();
network.addLayer(2, "linear");
network.addLayer(4, "relu");
network.addLayer(1, "sigmoid");

const loss = network.train(inputs, targets, 10000, 0.3, "crossEntropy");
console.log("final avg loss:", loss.toFixed(4));

for (let i = 0; i < inputs.length; i++) {
    const pred = network.runNetwork(inputs[i])[0];
    console.log(`${inputs[i]} -> ${pred.toFixed(4)} (target ${targets[i]})`);
}

writeNetworkToFile(network, modelPath);
console.log("\nsaved to", modelPath);

const serialized = writeNetwork(network);
const loaded = loadNetwork(serialized);
const fromFile = loadNetworkFromFile(modelPath);

for (const [label, model] of [["memory", loaded], ["file", fromFile]] as const) {
    console.log(`\nloaded (${label}):`);
    for (const input of inputs) {
        const pred = model.runNetwork(input)[0];
        console.log(`  ${input} -> ${pred.toFixed(4)}`);
    }
}
