import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { Callback, Logs } from "../src/callbacks.js";
import {
    earlyStopping,
    formatEpoch,
    History,
    learningRateScheduler,
    progressLogger,
    reduceLROnPlateau,
} from "../src/callbacks.js";
import { ValidationError } from "../src/core/errors.js";
import { dense } from "../src/layers/index.js";
import { Sequential } from "../src/model.js";
import { stepDecay } from "../src/schedules.js";

const X = Array.from({ length: 16 }, (_, i) => [i / 16, 1 - i / 16]);
const Y = X.map(([a, b]) => a - b);

function model(learningRate = 0.01): Sequential {
    const m = new Sequential({ inputSize: 2, seed: 1, layers: [dense(4, "tanh"), dense(1)] });
    m.compile({ loss: "mse", optimizer: { name: "sgd", learningRate }, metrics: ["mae"] });
    return m;
}

/** A callback that overwrites `key` in the epoch logs with scripted values (runs before the others). */
function script(key: string, values: number[]): Callback {
    return {
        onEpochEnd: (epoch: number, logs: Logs) => {
            logs[key] = values[epoch];
        },
    };
}

function captureWarnings<T>(fn: () => T): { result: T; warnings: string[] } {
    const warnings: string[] = [];
    const original = console.warn;
    console.warn = (message: string) => void warnings.push(message);
    try {
        return { result: fn(), warnings };
    } finally {
        console.warn = original;
    }
}

describe("History", () => {
    it("records epochs and aligns keys that appear or disappear", () => {
        const history = new History();
        history.append(0, { loss: 3 });
        history.append(1, { loss: 2, valLoss: 2.5 });
        history.append(2, { valLoss: 2.2 });
        assert.deepEqual(history.epochs, [0, 1, 2]);
        assert.equal(history.length, 3);
        assert.deepEqual(history.history.loss, [3, 2, Number.NaN]);
        assert.deepEqual(history.history.valLoss, [Number.NaN, 2.5, 2.2]);
        assert.equal(history.last("valLoss"), 2.2);
        assert.equal(history.last("nope"), undefined);
        assert.deepEqual(history.toJSON(), {
            epochs: [0, 1, 2],
            history: { loss: [3, 2, Number.NaN], valLoss: [Number.NaN, 2.5, 2.2] },
        });
    });

    it("finds the best epoch (min for losses, max for accuracy, or explicit)", () => {
        const history = new History();
        [
            { loss: 1, accuracy: 0.5 },
            { loss: 0.4, accuracy: 0.9 },
            { loss: 0.4, accuracy: 0.8 },
            { loss: 0.6, accuracy: 0.9 },
        ].forEach((logs, epoch) => {
            history.append(epoch, logs);
        });
        assert.deepEqual(history.best("loss"), { epoch: 1, value: 0.4 });
        assert.deepEqual(history.best("accuracy"), { epoch: 1, value: 0.9 });
        assert.deepEqual(history.best("loss", "max"), { epoch: 0, value: 1 });
        assert.equal(history.best("missing"), undefined);
    });
});

describe("earlyStopping", () => {
    it("stops after `patience` epochs without improvement larger than minDelta", () => {
        const stopper = earlyStopping({ monitor: "loss", patience: 3, minDelta: 0.05 });
        const history = model().fit(X, Y, {
            epochs: 20,
            callbacks: [script("loss", [1, 0.9, 0.88, 0.87, 0.86, 0.5]), stopper],
        });
        // 0.88, 0.87, 0.86 improve on 0.9 by less than 0.05 -> stop at epoch 4.
        assert.equal(stopper.stoppedEpoch, 4);
        assert.equal(stopper.bestEpoch, 1);
        assert.equal(history.length, 5);
    });

    it("maximizes accuracy-like keys in auto mode and honours an explicit mode", () => {
        const auto = earlyStopping({ monitor: "acc", patience: 1 });
        model().fit(X, Y, { epochs: 10, callbacks: [script("acc", [0.5, 0.6, 0.55, 0.7]), auto] });
        assert.deepEqual([auto.stoppedEpoch, auto.bestEpoch, auto.bestValue], [2, 1, 0.6]);
        const max = earlyStopping({ monitor: "loss", mode: "max", patience: 1 });
        model().fit(X, Y, { epochs: 10, callbacks: [script("loss", [1, 2, 1.5]), max] });
        assert.deepEqual([max.stoppedEpoch, max.bestEpoch], [2, 1]);
    });

    it("runs to completion without stopping and resets between fits", () => {
        const stopper = earlyStopping({ monitor: "loss", patience: 2 });
        const m = model();
        m.fit(X, Y, { epochs: 3, callbacks: [script("loss", [1, 2, 3]), stopper] });
        assert.equal(stopper.stoppedEpoch, 2);
        const history = m.fit(X, Y, { epochs: 3, callbacks: [script("loss", [3, 2, 1]), stopper] });
        assert.equal(stopper.stoppedEpoch, null);
        assert.equal(stopper.bestEpoch, 2);
        assert.equal(history.length, 3);
    });

    it("falls back from valLoss to loss with one warning when there is no validation data", () => {
        const stopper = earlyStopping({ patience: 1 });
        const { warnings } = captureWarnings(() =>
            model().fit(X, Y, { epochs: 5, callbacks: [script("loss", [3, 2, 2.5, 1]), stopper] }),
        );
        assert.equal(warnings.length, 1);
        assert.match(
            warnings[0],
            /"valLoss" is not available because fit\(\) has no validation data; monitoring "loss" instead/,
        );
        assert.equal(stopper.stoppedEpoch, 2);
    });

    it("rejects monitored keys that are never logged, listing the available ones", () => {
        assert.throws(
            () =>
                model().fit(X, Y, {
                    epochs: 2,
                    validationSplit: 0.25,
                    callbacks: [earlyStopping({ monitor: "val_loss" })],
                }),
            (e: unknown) =>
                e instanceof ValidationError &&
                /monitored value "val_loss" is not in the epoch logs. Available: loss, meanAbsoluteError, valLoss/.test(
                    e.message,
                ),
        );
        assert.throws(() => earlyStopping({ patience: -1 }), /"patience" must be a non-negative integer/);
        assert.throws(() => earlyStopping({ mode: "up" as never }), /"mode" must be "min", "max" or "auto"/);
        assert.throws(() => earlyStopping({ patient: 3 } as never), /unknown option "patient"/);
    });
});

describe("learningRateScheduler", () => {
    it("sets the learning rate at the start of each epoch and logs it", () => {
        const history = model().fit(X, Y, {
            epochs: 5,
            callbacks: [learningRateScheduler(stepDecay({ initial: 0.1, factor: 0.5, every: 2 }))],
        });
        assert.deepEqual(history.history.learningRate, [0.1, 0.1, 0.05, 0.05, 0.025]);
    });

    it("passes the current rate to the schedule and validates what it returns", () => {
        const history = model(0.2).fit(X, Y, { epochs: 3, callbacks: [learningRateScheduler((_epoch, lr) => lr / 2)] });
        assert.deepEqual(history.history.learningRate, [0.1, 0.05, 0.025]);
        assert.throws(
            () =>
                model().fit(X, Y, {
                    epochs: 2,
                    callbacks: [learningRateScheduler((epoch) => (epoch === 1 ? Number.NaN : 0.1))],
                }),
            /schedule returned NaN for epoch 1; it must return a finite number >= 0/,
        );
        assert.throws(() => learningRateScheduler(0.1 as never), /expected a function/);
    });
});

describe("reduceLROnPlateau", () => {
    it("multiplies the rate by `factor` after `patience` flat epochs, with cooldown and a floor", () => {
        const m = model(1);
        const rates: number[] = [];
        m.fit(X, Y, {
            epochs: 9,
            callbacks: [
                script("loss", [1, 1, 1, 1, 1, 1, 1, 1, 1]),
                reduceLROnPlateau({ monitor: "loss", factor: 0.5, patience: 2, cooldown: 1, minLearningRate: 0.2 }),
                { onEpochEnd: (_e, _logs, ctx) => void rates.push(ctx.optimizer.learningRate) },
            ],
        });
        // wait 1, 2 -> reduce (0.5) at epoch 2; cooldown epoch 3; wait 1, 2 -> 0.25 at epoch 5; cooldown 6; -> floor 0.2 at 8.
        assert.deepEqual(rates, [1, 1, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.2]);
    });

    it("resets its patience when the monitored value improves", () => {
        const m = model(1);
        m.fit(X, Y, {
            epochs: 6,
            callbacks: [script("loss", [5, 4, 4, 3, 3, 2]), reduceLROnPlateau({ monitor: "loss", patience: 2 })],
        });
        assert.equal(m.optimizer?.learningRate, 1);
        assert.throws(() => reduceLROnPlateau({ factor: 1 }), /"factor" must be a number in \(0, 1\)/);
    });
});

describe("progressLogger", () => {
    it("logs every N epochs and always the last", () => {
        const lines: string[] = [];
        model().fit(X, Y, { epochs: 5, callbacks: [progressLogger({ every: 2, log: (line) => lines.push(line) })] });
        assert.equal(lines.length, 3);
        assert.match(lines[0], /^Epoch 2\/5 - loss: \d+\.\d{4} - meanAbsoluteError: \d+\.\d{4} - [\d.]+ms$/);
        assert.match(lines[2], /^Epoch 5\/5 /);
    });

    it("formats epoch lines", () => {
        assert.equal(
            formatEpoch(3, 10, {
                loss: 0.123456,
                accuracy: 1,
                valLoss: 0.00001234,
                learningRate: 0.1,
                durationMs: 12.6,
            }),
            "Epoch  3/10 - loss: 0.1235 - accuracy: 1.0000 - valLoss: 1.234e-5 - 13ms",
        );
        assert.equal(formatEpoch(1, 1, { loss: Number.NaN, durationMs: 0.25 }), "Epoch 1/1 - loss: NaN - 0.3ms");
        assert.throws(() => progressLogger({ every: 0 }), /"every" must be a positive integer/);
    });
});
