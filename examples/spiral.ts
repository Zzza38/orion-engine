/**
 * Spiral: separate three interleaved spiral arms, a classic problem that needs a non-linear
 * decision boundary.
 *
 * Demonstrates a deeper network with BatchNormalization, Dropout and He initialization, a cosine
 * learning-rate schedule, a custom onEpochEnd logger, and an ASCII map of the learned decision
 * regions.
 *
 *   npx tsx examples/spiral.ts
 *
 * Expected output (seeded): a line every 25 epochs where the (noisy) training loss jumps around
 * 0.1–0.5 while valLoss falls steadily from about 0.16 to 0.10, valAccuracy climbs from 0.93 to
 * 0.98, and the learning rate follows the cosine curve from 0.02 down to 0.0005; then a test
 * accuracy of 97.8%, and a map where the three regions (·, +, #) spiral around the centre, with
 * nearly every test point (A, B, C) inside its own class's region.
 */
import type { Logs } from "../src/index.js";
import {
    activation,
    argmax,
    batchNormalization,
    cosineDecay,
    dense,
    dropout,
    learningRateScheduler,
    Random,
    Sequential,
    trainTestSplit,
} from "../src/index.js";

// Three arms of 120 points each, with a little angular noise.
const CLASSES = 3;
const rng = new Random(11);
const x: number[][] = [];
const y: number[] = [];
for (let c = 0; c < CLASSES; c++) {
    for (let i = 0; i < 120; i++) {
        const radius = i / 120;
        const angle = (c * 2 * Math.PI) / CLASSES + radius * 5 + rng.normal(0, 0.15);
        x.push([radius * Math.sin(angle), radius * Math.cos(angle)]);
        y.push(c);
    }
}
const { xTrain, xTest, yTrain, yTest } = trainTestSplit(x, y, { testSize: 0.25, seed: 5 });

// A block per hidden layer: Dense (no activation, no bias) → BatchNormalization → ReLU. Batch norm
// re-centres every feature, so the Dense bias would be redundant; He initialization suits ReLU.
const block = (units: number) => [
    dense(units, { useBias: false, kernelInitializer: "heNormal" }),
    batchNormalization(),
    activation("relu"),
];
const model = new Sequential({
    inputSize: 2,
    seed: 8,
    name: "spiral",
    layers: [...block(64), dropout(0.1), ...block(64), dense(CLASSES, "softmax")],
});
model.compile({ loss: "scce", optimizer: { name: "adam", learningRate: 0.02 }, metrics: ["accuracy"] });

// Cosine decay: start at 0.02 and glide down to 0.0005 by the last epoch.
const EPOCHS = 150;
const schedule = cosineDecay({ initial: 0.02, epochs: EPOCHS, minimum: 0.0005 });

// The training loss is measured in training mode (dropout on, per-batch statistics), so it is noisy;
// the validation numbers are computed in inference mode and show the real trend.
const logEvery25 = (epoch: number, logs: Logs) => {
    if ((epoch + 1) % 25 !== 0) return;
    console.log(
        `epoch ${String(epoch + 1).padStart(3)}  loss ${logs.loss.toFixed(4)}  valLoss ${logs.valLoss.toFixed(4)}  ` +
            `valAccuracy ${logs.valAccuracy.toFixed(3)}  learningRate ${logs.learningRate.toFixed(5)}`,
    );
};
model.fit(xTrain, yTrain, {
    epochs: EPOCHS,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: [learningRateScheduler(schedule)],
    onEpochEnd: logEvery25,
});

// evaluate() and predict() run in inference mode: dropout is off and batch norm uses its moving averages.
const { accuracy } = model.evaluate(xTest, yTest);
console.log(`\nTest accuracy: ${(accuracy * 100).toFixed(1)}% on ${yTest.length} unseen points\n`);

// Decision map: classify every cell of a grid in one predict() call, then overlay the test points.
const COLS = 64;
const ROWS = 28;
const toPoint = (col: number, row: number) => [-1.1 + (2.2 * col) / (COLS - 1), 1.1 - (2.2 * row) / (ROWS - 1)];
const cells: number[][] = [];
for (let row = 0; row < ROWS; row++) for (let col = 0; col < COLS; col++) cells.push(toPoint(col, row));
const regions = argmax(model.predict(cells));
const canvas = Array.from({ length: ROWS }, (_, row) =>
    Array.from({ length: COLS }, (_, col) => "·+#"[regions[row * COLS + col]]),
);
for (const [i, [px, py]] of xTest.entries()) {
    const col = Math.round(((px + 1.1) / 2.2) * (COLS - 1));
    const row = Math.round(((1.1 - py) / 2.2) * (ROWS - 1));
    canvas[row][col] = "ABC"[yTest[i]];
}
console.log(canvas.map((line) => `  ${line.join("")}`).join("\n"));
console.log("\n  regions: · class 0, + class 1, # class 2   test points: A, B, C (their true class)");
