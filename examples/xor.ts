/**
 * XOR: the smallest problem a linear model cannot solve, learned by a 2-8-1 network.
 *
 * Demonstrates the core workflow: build a Sequential model, compile it with a loss and an
 * optimizer, fit it with progress logging, and predict.
 *
 *   npx tsx examples/xor.ts
 *
 * Expected output (seeded, so it is the same on every run): the model summary (33 parameters),
 * a progress line every 100 epochs with the loss falling to about 0.0015 and accuracy 1.0000,
 * then one line per input, e.g. `1 xor 0 = 0.998 → 1`, with every rounded prediction correct.
 */
import { dense, Sequential } from "../src/index.js";

// The four XOR cases. Targets for a single sigmoid output can be a flat array: one value per sample.
const x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];
const y = [0, 1, 1, 0];

// Two inputs → 8 tanh units → 1 sigmoid unit (a probability). The seed makes the run reproducible.
const model = new Sequential({ inputSize: 2, seed: 42, name: "xor", layers: [dense(8, "tanh"), dense(1, "sigmoid")] });

// Binary cross-entropy ("bce") is the natural loss for a sigmoid output.
model.compile({ loss: "bce", optimizer: { name: "adam", learningRate: 0.05 }, metrics: ["accuracy"] });
console.log(model.summary());
console.log();

// All four samples fit in one batch; `verbose: 100` logs every 100th epoch.
model.fit(x, y, { epochs: 300, batchSize: 4, verbose: 100 });
console.log();

// predict() mirrors its input: number[][] in, number[][] out (one row per sample).
const predictions = model.predict(x);
for (const [i, [a, b]] of x.entries()) {
    const p = predictions[i][0];
    console.log(`${a} xor ${b} = ${p.toFixed(3)} → ${Math.round(p)}`);
}
