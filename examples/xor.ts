/**
 * Learn XOR, then save and reload the model.
 *
 *   pnpm examples
 */
import { tmpdir } from "node:os";
import { join } from "node:path";
import { dense, loadModel, Sequential, saveModel } from "../src/node.js";

const x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];
const y = [0, 1, 1, 0];

const model = new Sequential({ inputSize: 2, seed: 42, name: "xor", layers: [dense(8, "tanh"), dense(1, "sigmoid")] });
model.compile({ loss: "bce", optimizer: { name: "adam", learningRate: 0.05 }, metrics: ["accuracy"] });
console.log(model.summary());

const history = model.fit(x, y, { epochs: 300, batchSize: 4, verbose: 50 });
console.log(`\nfinal loss ${history.last("loss")?.toFixed(4)}, accuracy ${history.last("accuracy")}`);

for (const [input, output] of x.map((row) => [row, model.predict(row)] as const)) {
    console.log(`${input.join(" xor ")} -> ${output[0].toFixed(3)}`);
}

const path = join(tmpdir(), "orion-examples", "xor.onn");
await saveModel(model, path);
const restored = await loadModel(path);
console.log(`\nsaved to ${path}; reloaded model predicts [1, 0] -> ${restored.predict([1, 0])[0].toFixed(3)}`);
