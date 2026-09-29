/**
 * Regression: fit a noisy sine wave with a small MLP and plot the result in the terminal.
 *
 * Demonstrates a linear output with mean squared error, the "mae" / "rmse" metrics, a seeded
 * train/test split, reduceLROnPlateau, and predicting a whole grid at once.
 *
 *   npx tsx examples/regression.ts
 *
 * Expected output (seeded): progress every 100 epochs with the loss settling near 0.009, the final
 * learning rate (0.0001, the floor), then test MSE ≈ 0.0125 and RMSE ≈ 0.11, close to the noise
 * level σ = 0.1 (the model learned the signal, not the noise), an RMSE of about 0.02 against the
 * noise-free sine, and a 64×15 ASCII plot where the model's curve (●) runs through the middle of
 * the noisy samples (·).
 */
import { dense, Random, reduceLROnPlateau, Sequential, trainTestSplit } from "../src/index.js";

// 300 samples of y = sin(x) + Gaussian noise (σ = 0.1) on [-π, π], from a seeded generator.
const NOISE = 0.1;
const rng = new Random(2024);
const x: number[][] = [];
const y: number[] = [];
for (let i = 0; i < 300; i++) {
    const v = rng.uniform(-Math.PI, Math.PI);
    x.push([v]); // one feature per sample: each sample is a row
    y.push(Math.sin(v) + rng.normal(0, NOISE));
}
const { xTrain, xTest, yTrain, yTest } = trainTestSplit(x, y, { testSize: 0.2, seed: 1 });

// 1 input → 32 tanh → 32 tanh → 1 linear output (a Dense layer is linear unless told otherwise).
const model = new Sequential({ inputSize: 1, seed: 3, layers: [dense(32, "tanh"), dense(32, "tanh"), dense(1)] });
model.compile({ loss: "mse", optimizer: { name: "adam", learningRate: 0.01 }, metrics: ["mae", "rmse"] });
// Halve the learning rate whenever the training loss stalls for 20 epochs: Adam at 0.01 gets close
// quickly, and the smaller steps then settle the curve instead of jittering around it.
const history = model.fit(xTrain, yTrain, {
    epochs: 400,
    batchSize: 32,
    verbose: 100,
    callbacks: [reduceLROnPlateau({ monitor: "loss", factor: 0.5, patience: 20, minLearningRate: 1e-4 })],
});
console.log(`Final learning rate: ${history.last("learningRate")}`);

// Metrics are reported under their canonical names: "mae" → meanAbsoluteError, "rmse" → rootMeanSquaredError.
const test = model.evaluate(xTest, yTest);
console.log(
    `\nTest MSE ${test.loss.toFixed(4)}, RMSE ${test.rootMeanSquaredError.toFixed(4)}, ` +
        `MAE ${test.meanAbsoluteError.toFixed(4)} (noise σ = ${NOISE})`,
);

// How close is the model to the true, noise-free function?
const WIDTH = 64;
const grid = Array.from({ length: WIDTH }, (_, i) => [-Math.PI + (2 * Math.PI * i) / (WIDTH - 1)]);
const curve = model.predict(grid).map((row) => row[0]);
const trueRmse = Math.sqrt(curve.reduce((sum, p, i) => sum + (p - Math.sin(grid[i][0])) ** 2, 0) / WIDTH);
console.log(`RMSE against the noise-free sin(x): ${trueRmse.toFixed(4)}\n`);

// ASCII plot: noisy training samples as "·", the model's prediction as "●".
const HEIGHT = 15;
const Y_MIN = -1.4;
const Y_MAX = 1.4;
const canvas = Array.from({ length: HEIGHT }, () => new Array<string>(WIDTH).fill(" "));
const plot = (xValue: number, yValue: number, mark: string) => {
    const col = Math.round(((xValue + Math.PI) / (2 * Math.PI)) * (WIDTH - 1));
    const row = Math.round(((Y_MAX - yValue) / (Y_MAX - Y_MIN)) * (HEIGHT - 1));
    if (row >= 0 && row < HEIGHT && col >= 0 && col < WIDTH) canvas[row][col] = mark;
};
for (const [i, [v]] of xTrain.entries()) plot(v, yTrain[i], "·");
for (const [i, [v]] of grid.entries()) plot(v, curve[i], "●");
for (const [r, line] of canvas.entries()) {
    const label =
        r === 0 ? Y_MAX.toFixed(1) : r === HEIGHT - 1 ? Y_MIN.toFixed(1) : r === (HEIGHT - 1) / 2 ? "0.0" : "";
    console.log(`${label.padStart(5)} │${line.join("")}`);
}
console.log(`      └${"─".repeat(WIDTH)}`);
console.log(`       -π${"π".padStart(WIDTH - 2)}`);
