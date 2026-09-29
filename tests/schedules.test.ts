import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { ValidationError } from "../src/core/errors.js";
import type { LearningRateSchedule } from "../src/schedules.js";
import {
    constantSchedule,
    cosineDecay,
    exponentialDecay,
    linearWarmup,
    piecewiseConstant,
    stepDecay,
} from "../src/schedules.js";

function assertValues(
    schedule: LearningRateSchedule,
    expected: [epoch: number, lr: number][],
    tolerance = 1e-12,
): void {
    for (const [epoch, lr] of expected) {
        const actual = schedule(epoch);
        assert.ok(Math.abs(actual - lr) <= tolerance, `epoch ${epoch}: expected ${lr}, got ${actual}`);
    }
}

describe("constantSchedule", () => {
    it("returns the same rate for every epoch", () => {
        const schedule = constantSchedule(0.1);
        assert.equal(schedule(0), 0.1);
        assert.equal(schedule(1000), 0.1);
        assert.equal(schedule(2.5), 0.1);
    });

    it("validates its input", () => {
        assert.throws(() => constantSchedule(-1), ValidationError);
        assert.throws(() => constantSchedule(Number.NaN), ValidationError);
        assert.throws(() => constantSchedule(Number.POSITIVE_INFINITY), ValidationError);
    });
});

describe("stepDecay", () => {
    it("multiplies by factor every `every` epochs", () => {
        const schedule = stepDecay({ initial: 0.1, factor: 0.5, every: 10 });
        assertValues(schedule, [
            [0, 0.1],
            [9, 0.1],
            [9.99, 0.1],
            [10, 0.05],
            [19, 0.05],
            [20, 0.025],
            [35, 0.0125],
        ]);
        assert.equal(schedule(0), 0.1);
    });

    it("factor 1 is constant", () => {
        assertValues(stepDecay({ initial: 0.3, factor: 1, every: 1 }), [
            [0, 0.3],
            [50, 0.3],
        ]);
    });

    it("validates its options", () => {
        assert.throws(() => stepDecay({ initial: -0.1, factor: 0.5, every: 10 }), ValidationError);
        assert.throws(() => stepDecay({ initial: 0.1, factor: 0, every: 10 }), ValidationError);
        assert.throws(() => stepDecay({ initial: 0.1, factor: 1.5, every: 10 }), ValidationError);
        assert.throws(() => stepDecay({ initial: 0.1, factor: 0.5, every: 0 }), ValidationError);
        assert.throws(() => stepDecay({ initial: 0.1, factor: 0.5, every: 2.5 }), ValidationError);
        assert.throws(() => stepDecay({ initial: 0.1, factor: 0.5 } as never), ValidationError);
        assert.throws(() => stepDecay(null as never), ValidationError);
    });
});

describe("exponentialDecay", () => {
    it("decays smoothly by `rate` per epoch", () => {
        assertValues(exponentialDecay({ initial: 1, rate: 0.5 }), [
            [0, 1],
            [1, 0.5],
            [3, 0.125],
            [0.5, Math.SQRT1_2],
        ]);
    });

    it("stretches the decay over `every` epochs", () => {
        assertValues(exponentialDecay({ initial: 0.2, rate: 0.1, every: 4 }), [
            [0, 0.2],
            [2, 0.2 * Math.sqrt(0.1)],
            [4, 0.02],
            [8, 0.002],
        ]);
    });

    it("validates its options", () => {
        assert.throws(() => exponentialDecay({ initial: 1, rate: 0 }), ValidationError);
        assert.throws(() => exponentialDecay({ initial: 1, rate: 1.1 }), ValidationError);
        assert.throws(() => exponentialDecay({ initial: 1, rate: 0.5, every: 0 }), ValidationError);
        assert.throws(() => exponentialDecay({ initial: 1, rate: 0.5, every: -2 }), ValidationError);
        assert.throws(() => exponentialDecay({ initial: Number.NaN, rate: 0.5 }), ValidationError);
    });
});

describe("cosineDecay", () => {
    it("anneals from initial to 0 over `epochs`, then stays at 0", () => {
        assertValues(cosineDecay({ initial: 1, epochs: 10 }), [
            [0, 1],
            [5, 0.5],
            [2.5, 0.5 * (1 + Math.SQRT1_2)],
            [7.5, 0.5 * (1 - Math.SQRT1_2)],
            [10, 0],
            [25, 0],
        ]);
    });

    it("respects `minimum`", () => {
        const schedule = cosineDecay({ initial: 1, epochs: 4, minimum: 0.1 });
        assertValues(schedule, [
            [0, 1],
            [2, 0.55],
            [4, 0.1],
            [100, 0.1],
        ]);
        assert.equal(schedule(4), 0.1);
    });

    it("is monotonically non-increasing", () => {
        const schedule = cosineDecay({ initial: 0.01, epochs: 50, minimum: 1e-4 });
        for (let e = 1; e <= 60; e++) assert.ok(schedule(e) <= schedule(e - 1), `epoch ${e}`);
    });

    it("validates its options", () => {
        assert.throws(() => cosineDecay({ initial: 1, epochs: 0 }), ValidationError);
        assert.throws(() => cosineDecay({ initial: 1, epochs: 2.5 }), ValidationError);
        assert.throws(() => cosineDecay({ initial: 1, epochs: 10, minimum: 2 }), ValidationError);
        assert.throws(() => cosineDecay({ initial: 1, epochs: 10, minimum: -0.1 }), ValidationError);
    });
});

describe("linearWarmup", () => {
    it("ramps from 0 to the wrapped schedule's initial value", () => {
        assertValues(linearWarmup(constantSchedule(1), { epochs: 4 }), [
            [0, 0],
            [1, 0.25],
            [2, 0.5],
            [3, 0.75],
            [4, 1],
            [100, 1],
        ]);
    });

    it("starts from `from`", () => {
        assertValues(linearWarmup(constantSchedule(1), { epochs: 4, from: 0.2 }), [
            [0, 0.2],
            [2, 0.6],
            [4, 1],
        ]);
    });

    it("runs the wrapped schedule shifted by the warm-up length", () => {
        const schedule = linearWarmup(stepDecay({ initial: 1, factor: 0.5, every: 2 }), { epochs: 3 });
        assertValues(schedule, [
            [0, 0],
            [1.5, 0.5],
            [3, 1],
            [4, 1],
            [5, 0.5],
            [7, 0.25],
        ]);
    });

    it("composes with cosineDecay", () => {
        const schedule = linearWarmup(cosineDecay({ initial: 0.01, epochs: 10 }), { epochs: 5 });
        assertValues(schedule, [
            [0, 0],
            [5, 0.01],
            [10, 0.005],
            [15, 0],
        ]);
    });

    it("epochs = 0 passes the wrapped schedule through", () => {
        const inner = exponentialDecay({ initial: 1, rate: 0.5 });
        const schedule = linearWarmup(inner, { epochs: 0 });
        for (const e of [0, 1, 2.5, 7]) assert.equal(schedule(e), inner(e));
    });

    it("validates its arguments", () => {
        assert.throws(() => linearWarmup(0.1 as never, { epochs: 3 }), ValidationError);
        assert.throws(() => linearWarmup(constantSchedule(1), { epochs: -1 }), ValidationError);
        assert.throws(() => linearWarmup(constantSchedule(1), { epochs: 1.5 }), ValidationError);
        assert.throws(() => linearWarmup(constantSchedule(1), { epochs: 3, from: -0.1 }), ValidationError);
        assert.throws(() => linearWarmup(constantSchedule(1), undefined as never), ValidationError);
    });
});

describe("piecewiseConstant", () => {
    it("switches value at each boundary epoch", () => {
        const schedule = piecewiseConstant({ boundaries: [10, 20], values: [0.1, 0.01, 0.001] });
        assertValues(schedule, [
            [0, 0.1],
            [9, 0.1],
            [9.5, 0.1],
            [10, 0.01],
            [19, 0.01],
            [20, 0.001],
            [1000, 0.001],
        ]);
    });

    it("works with no boundaries", () => {
        assertValues(piecewiseConstant({ boundaries: [], values: [0.3] }), [
            [0, 0.3],
            [99, 0.3],
        ]);
    });

    it("copies its inputs", () => {
        const boundaries = [5];
        const values = [1, 2];
        const schedule = piecewiseConstant({ boundaries, values });
        boundaries[0] = 100;
        values[1] = 50;
        assert.equal(schedule(6), 2);
    });

    it("validates its options", () => {
        assert.throws(() => piecewiseConstant({ boundaries: [10], values: [0.1] }), ValidationError);
        assert.throws(() => piecewiseConstant({ boundaries: [10], values: [0.1, 0.2, 0.3] }), ValidationError);
        assert.throws(() => piecewiseConstant({ boundaries: [10, 10], values: [1, 2, 3] }), ValidationError);
        assert.throws(() => piecewiseConstant({ boundaries: [20, 10], values: [1, 2, 3] }), ValidationError);
        assert.throws(() => piecewiseConstant({ boundaries: [-1], values: [1, 2] }), ValidationError);
        assert.throws(() => piecewiseConstant({ boundaries: [Number.NaN], values: [1, 2] }), ValidationError);
        assert.throws(() => piecewiseConstant({ boundaries: [5], values: [1, -2] }), ValidationError);
        assert.throws(() => piecewiseConstant({ boundaries: 5, values: [1, 2] } as never), ValidationError);
    });
});

describe("epoch validation", () => {
    it("every schedule rejects negative or non-finite epochs", () => {
        const schedules: LearningRateSchedule[] = [
            constantSchedule(0.1),
            stepDecay({ initial: 0.1, factor: 0.5, every: 2 }),
            exponentialDecay({ initial: 0.1, rate: 0.5 }),
            cosineDecay({ initial: 0.1, epochs: 5 }),
            linearWarmup(constantSchedule(0.1), { epochs: 2 }),
            piecewiseConstant({ boundaries: [1], values: [0.1, 0.01] }),
        ];
        for (const schedule of schedules) {
            for (const bad of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
                assert.throws(() => schedule(bad), ValidationError);
            }
        }
    });

    it("error messages identify the schedule and the offending value", () => {
        assert.throws(
            () => stepDecay({ initial: 0.1, factor: 2, every: 1 }),
            /stepDecay: "factor" must be a number in \(0, 1\], got 2/,
        );
        assert.throws(
            () => cosineDecay({ initial: 1, epochs: 5 })(-3),
            /cosineDecay: epoch must be a finite number >= 0, got -3/,
        );
    });
});
