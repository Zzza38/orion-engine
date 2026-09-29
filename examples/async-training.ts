/**
 * Async training: train without freezing the event loop, report progress from an async callback,
 * and cancel a long run with an AbortSignal.
 *
 * `fitAsync` runs the same training loop as `fit`, but yields to the event loop every ~16 ms and
 * awaits callbacks that return Promises, so timers, network requests and (in a browser) rendering
 * and clicks keep working while the model trains. In a page you would wire an AbortController to a
 * Stop button; here a timeout plays that role.
 *
 *   npx tsx examples/async-training.ts
 *
 * Expected output: five progress lines (epochs 4, 8, …, 20) with valLoss falling from about 0.07
 * to about 0.02, then "Training for up to 2 s…", a progress line every 20 epochs, and "Stopped by
 * the timeout after N epochs" (N depends on your machine; a few dozen is typical) and an MSE below
 * part 1's final loss, because the model keeps everything it learned before the abort. The timer-tick count keeps growing
 * during training, which shows the event loop was never blocked (a synchronous fit() would leave
 * it at 0).
 */
import type { Logs } from "../src/index.js";
import { dense, Random, Sequential } from "../src/index.js";

// A wavy 2-D surface, sampled 4,000 times: enough work that training takes a few seconds.
const rng = new Random(7);
const x = Array.from({ length: 4000 }, () => [rng.uniform(-2, 2), rng.uniform(-2, 2)]);
const y = x.map(([a, b]) => Math.sin(2 * a) * Math.cos(2 * b));

const model = new Sequential({ inputSize: 2, seed: 1, layers: [dense(64, "tanh"), dense(64, "tanh"), dense(1)] });
model.compile({ loss: "mse", optimizer: { name: "adam", learningRate: 0.005 } });

// This timer only fires if training gives the event loop a chance to run.
let ticks = 0;
const timer = setInterval(() => ticks++, 50);
const format = (epoch: number, logs: Logs) =>
    `epoch ${String(epoch + 1).padStart(3)}  loss ${logs.loss.toFixed(4)}  valLoss ${logs.valLoss.toFixed(4)}  ` +
    `(timer ticks so far: ${ticks})`;

// Part 1: an async progress callback. fitAsync awaits the returned Promise before the next epoch,
// so this is the place to update a UI, post a message to the main thread, or append to a log file.
await model.fitAsync(x, y, {
    epochs: 20,
    batchSize: 64,
    validationSplit: 0.1,
    onEpochEnd: async (epoch, logs) => {
        if ((epoch + 1) % 4 === 0) console.log(format(epoch, logs));
        await new Promise((resolve) => setTimeout(resolve, 0)); // stand-in for real async work
    },
});

// Part 2: cancellation. Ask for far more epochs than we are willing to wait for, and abort after
// 2 seconds. fitAsync rejects with the signal's reason (a "TimeoutError" DOMException here) at the
// next batch; the weights keep everything learned up to that point.
console.log("\nTraining for up to 2 s…");
let completed = 0;
try {
    await model.fitAsync(x, y, {
        epochs: 100_000,
        batchSize: 64,
        validationSplit: 0.1,
        signal: AbortSignal.timeout(2000),
        onEpochEnd: (epoch, logs) => {
            completed = epoch + 1;
            if (completed % 20 === 0) console.log(format(epoch, logs));
        },
    });
} catch (error) {
    if (!(error instanceof Error && error.name === "TimeoutError")) throw error;
    const { loss } = model.evaluate(x, y);
    console.log(`Stopped by the timeout after ${completed} epochs; MSE on all data is now ${loss.toFixed(4)}.`);
} finally {
    clearInterval(timer);
}
console.log(`The 50 ms timer ticked ${ticks} times while training ran.`);
