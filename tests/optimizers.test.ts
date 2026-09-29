import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { ShapeError, ValidationError } from "../src/core/errors.js";
import { Matrix } from "../src/core/matrix.js";
import type { Optimizer, OptimizerConfig, Parameter } from "../src/core/types.js";
import {
    Adagrad,
    Adam,
    AdamW,
    OPTIMIZER_NAMES,
    RMSprop,
    SGD,
    getOptimizer,
} from "../src/optimizers.js";

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

interface ParamOptions {
    name?: string;
    trainable?: boolean;
    regularize?: boolean;
    rows?: number;
}

function makeParam(values: readonly number[], options: ParamOptions = {}): Parameter {
    const rows = options.rows ?? 1;
    const cols = values.length / rows;
    return {
        name: options.name ?? "w",
        value: new Matrix(rows, cols, Float64Array.from(values)),
        grad: new Matrix(rows, cols),
        trainable: options.trainable ?? true,
        regularize: options.regularize ?? true,
    };
}

function assertClose(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 1e-12, message = ""): void {
    assert.equal(actual.length, expected.length, `${message} length`);
    for (let i = 0; i < expected.length; i++) {
        const diff = Math.abs(actual[i] - expected[i]);
        const scale = Math.max(1, Math.abs(expected[i]));
        assert.ok(
            diff <= tolerance * scale,
            `${message} [${i}]: expected ${expected[i]}, got ${actual[i]} (diff ${diff})`,
        );
    }
}

/** Scalar reference update: mutates `w` in place given gradient `g`. Holds its own state. */
type Reference = (w: number[], g: number[]) => void;

function sgdReference(o: { lr: number; momentum?: number; nesterov?: boolean; weightDecay?: number }): Reference {
    const mu = o.momentum ?? 0;
    const wd = o.weightDecay ?? 0;
    let v: number[] | undefined;
    return (w, g) => {
        v ??= w.map(() => 0);
        for (let i = 0; i < w.length; i++) {
            const gi = g[i] + wd * w[i];
            if (mu === 0) {
                w[i] -= o.lr * gi;
            } else {
                v[i] = mu * v[i] + gi;
                w[i] -= o.lr * (o.nesterov ? gi + mu * v[i] : v[i]);
            }
        }
    };
}

function adamReference(o: {
    lr: number;
    beta1?: number;
    beta2?: number;
    epsilon?: number;
    amsgrad?: boolean;
    weightDecay?: number;
}): Reference {
    const b1 = o.beta1 ?? 0.9;
    const b2 = o.beta2 ?? 0.999;
    const eps = o.epsilon ?? 1e-7;
    const wd = o.weightDecay ?? 0;
    let m: number[] | undefined;
    let v: number[] | undefined;
    let vMax: number[] | undefined;
    let t = 0;
    return (w, g) => {
        t++;
        m ??= w.map(() => 0);
        v ??= w.map(() => 0);
        vMax ??= w.map(() => 0);
        for (let i = 0; i < w.length; i++) {
            w[i] -= o.lr * wd * w[i]; // decoupled decay (AdamW only)
            m[i] = b1 * m[i] + (1 - b1) * g[i];
            v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2;
            vMax[i] = Math.max(vMax[i], v[i]);
            const mHat = m[i] / (1 - b1 ** t);
            const vHat = (o.amsgrad ? vMax[i] : v[i]) / (1 - b2 ** t);
            w[i] -= (o.lr * mHat) / (Math.sqrt(vHat) + eps);
        }
    };
}

function rmspropReference(o: { lr: number; rho?: number; momentum?: number; epsilon?: number; centered?: boolean }): Reference {
    const rho = o.rho ?? 0.9;
    const mu = o.momentum ?? 0;
    const eps = o.epsilon ?? 1e-7;
    let s: number[] | undefined;
    let mean: number[] | undefined;
    let buf: number[] | undefined;
    return (w, g) => {
        s ??= w.map(() => 0);
        mean ??= w.map(() => 0);
        buf ??= w.map(() => 0);
        for (let i = 0; i < w.length; i++) {
            s[i] = rho * s[i] + (1 - rho) * g[i] ** 2;
            mean[i] = rho * mean[i] + (1 - rho) * g[i];
            const denom = Math.sqrt(o.centered ? s[i] - mean[i] ** 2 : s[i]) + eps;
            if (mu > 0) {
                buf[i] = mu * buf[i] + g[i] / denom;
                w[i] -= o.lr * buf[i];
            } else {
                w[i] -= (o.lr * g[i]) / denom;
            }
        }
    };
}

function adagradReference(o: { lr: number; initialAccumulatorValue?: number; epsilon?: number }): Reference {
    const init = o.initialAccumulatorValue ?? 0.1;
    const eps = o.epsilon ?? 1e-7;
    let a: number[] | undefined;
    return (w, g) => {
        a ??= w.map(() => init);
        for (let i = 0; i < w.length; i++) {
            a[i] += g[i] ** 2;
            w[i] -= (o.lr * g[i]) / (Math.sqrt(a[i]) + eps);
        }
    };
}

/** A gradient that depends on the weights and the step, with mixed signs, so moments evolve non-trivially. */
function wobblyGradient(w: ArrayLike<number>, t: number): number[] {
    const target = [1, -2, 0.5, 3, -0.25, 0];
    return Array.from(w, (wi, i) => 2 * (wi - target[i % target.length]) + 0.7 * Math.sin(1.3 * t + i));
}

/** Runs the optimizer and the reference side by side for `steps` steps, comparing after each. */
function checkAgainstReference(optimizer: Optimizer, reference: Reference, steps = 30, rows = 2): void {
    const initial = [0.5, -1.5, 2, 0.1, -0.3, 4];
    const param = makeParam(initial, { rows });
    const w = initial.slice();
    for (let t = 1; t <= steps; t++) {
        const g = wobblyGradient(w, t);
        param.grad.data.set(wobblyGradient(param.value.data, t));
        optimizer.step([param]);
        reference(w, g);
        assertClose(param.value.data, w, 1e-10, `${optimizer.name} step ${t}`);
    }
    assert.equal(optimizer.iterations, steps);
}

/** Minimizes f(w) = Σ(w − c)², returning the final max |w − c|. */
function minimizeQuadratic(optimizer: Optimizer, steps: number): number {
    const c = [3, -2, 0.5, 1];
    const param = makeParam([0, 0, 0, 0]);
    for (let s = 0; s < steps; s++) {
        for (let i = 0; i < c.length; i++) param.grad.data[i] = 2 * (param.value.data[i] - c[i]);
        optimizer.step([param]);
    }
    return Math.max(...c.map((ci, i) => Math.abs(param.value.data[i] - ci)));
}

// ---------------------------------------------------------------------------------------------
// Hand-computed updates
// ---------------------------------------------------------------------------------------------

describe("single-step updates (hand-computed)", () => {
    it("SGD: w ← w − η·g", () => {
        const p = makeParam([1, 2]);
        p.grad.data.set([0.5, -1]);
        new SGD({ learningRate: 0.1 }).step([p]);
        assertClose(p.value.data, [0.95, 2.1]);
    });

    it("SGD momentum: second step uses v = μ·v + g", () => {
        const p = makeParam([1]);
        const opt = new SGD({ learningRate: 0.1, momentum: 0.9 });
        p.grad.data[0] = 1;
        opt.step([p]); // v = 1, w = 1 − 0.1 = 0.9
        assertClose(p.value.data, [0.9]);
        opt.step([p]); // v = 0.9 + 1 = 1.9, w = 0.9 − 0.19 = 0.71
        assertClose(p.value.data, [0.71]);
    });

    it("SGD Nesterov: w ← w − η·(g + μ·v)", () => {
        const p = makeParam([1]);
        const opt = new SGD({ learningRate: 0.1, momentum: 0.9, nesterov: true });
        p.grad.data[0] = 1;
        opt.step([p]); // v = 1, w = 1 − 0.1·(1 + 0.9) = 0.81
        assertClose(p.value.data, [0.81]);
        opt.step([p]); // v = 1.9, w = 0.81 − 0.1·(1 + 1.71) = 0.539
        assertClose(p.value.data, [0.539]);
    });

    it("SGD weight decay adds λ·w to the gradient", () => {
        const p = makeParam([2]);
        p.grad.data[0] = 1;
        new SGD({ learningRate: 0.1, weightDecay: 0.5 }).step([p]); // g = 1 + 1 = 2, w = 2 − 0.2
        assertClose(p.value.data, [1.8]);
    });

    it("Adam: first step moves by ≈ η·sign(g)", () => {
        const p = makeParam([1, 1]);
        p.grad.data.set([0.5, -2]);
        new Adam({ learningRate: 0.1 }).step([p]);
        // m̂ = g, v̂ = g², so Δ = η·g / (|g| + ε)
        assertClose(p.value.data, [1 - (0.1 * 0.5) / (0.5 + 1e-7), 1 + (0.1 * 2) / (2 + 1e-7)]);
    });

    it("Adam: second step matches bias-corrected moments", () => {
        const p = makeParam([0]);
        const opt = new Adam({ learningRate: 0.01, beta1: 0.5, beta2: 0.75, epsilon: 1e-8 });
        p.grad.data[0] = 1;
        opt.step([p]);
        p.grad.data[0] = 3;
        opt.step([p]);
        // step 1: m = 0.5, v = 0.25, m̂ = 1, v̂ = 1 → w = −0.01·1/(1+1e-8)
        // step 2: m = 0.25 + 1.5 = 1.75, v = 0.1875 + 2.25 = 2.4375
        //         m̂ = 1.75/0.75, v̂ = 2.4375/0.4375
        const w1 = -0.01 / (1 + 1e-8);
        const w2 = w1 - (0.01 * (1.75 / 0.75)) / (Math.sqrt(2.4375 / 0.4375) + 1e-8);
        assertClose(p.value.data, [w2]);
    });

    it("AdamW: decoupled decay shrinks w by (1 − η·λ) on top of the Adam step", () => {
        const p = makeParam([2]);
        p.grad.data[0] = 0.5;
        new AdamW({ learningRate: 0.1, weightDecay: 0.2 }).step([p]);
        assertClose(p.value.data, [2 * (1 - 0.02) - (0.1 * 0.5) / (0.5 + 1e-7)]);
    });

    it("RMSprop: first step divides by √((1 − ρ)·g²)", () => {
        const p = makeParam([1]);
        p.grad.data[0] = 0.5;
        new RMSprop({ learningRate: 0.01 }).step([p]);
        assertClose(p.value.data, [1 - (0.01 * 0.5) / (Math.sqrt(0.1 * 0.25) + 1e-7)]);
    });

    it("RMSprop centered: first step divides by √(s − ḡ²)", () => {
        const p = makeParam([1]);
        p.grad.data[0] = 0.5;
        new RMSprop({ learningRate: 0.01, centered: true }).step([p]);
        // s = 0.025, ḡ = 0.05, s − ḡ² = 0.0225 → √ = 0.15
        assertClose(p.value.data, [1 - (0.01 * 0.5) / (0.15 + 1e-7)]);
    });

    it("Adagrad: first step divides by √(a₀ + g²)", () => {
        const p = makeParam([1]);
        p.grad.data[0] = 0.5;
        new Adagrad({ learningRate: 0.01 }).step([p]);
        assertClose(p.value.data, [1 - (0.01 * 0.5) / (Math.sqrt(0.35) + 1e-7)]);
    });
});

// ---------------------------------------------------------------------------------------------
// Multi-step reference comparisons
// ---------------------------------------------------------------------------------------------

describe("multi-step updates match scalar reference implementations", () => {
    it("SGD (plain, momentum, Nesterov, weight decay)", () => {
        checkAgainstReference(new SGD({ learningRate: 0.05 }), sgdReference({ lr: 0.05 }));
        checkAgainstReference(new SGD({ learningRate: 0.05, momentum: 0.8 }), sgdReference({ lr: 0.05, momentum: 0.8 }));
        checkAgainstReference(
            new SGD({ learningRate: 0.05, momentum: 0.8, nesterov: true, weightDecay: 0.1 }),
            sgdReference({ lr: 0.05, momentum: 0.8, nesterov: true, weightDecay: 0.1 }),
        );
    });

    it("Adam (default and AMSGrad)", () => {
        checkAgainstReference(new Adam({ learningRate: 0.05 }), adamReference({ lr: 0.05 }), 60);
        checkAgainstReference(
            new Adam({ learningRate: 0.05, beta1: 0.7, beta2: 0.95, epsilon: 1e-6, amsgrad: true }),
            adamReference({ lr: 0.05, beta1: 0.7, beta2: 0.95, epsilon: 1e-6, amsgrad: true }),
            60,
        );
    });

    it("AdamW", () => {
        checkAgainstReference(new AdamW({ learningRate: 0.05 }), adamReference({ lr: 0.05, weightDecay: 0.01 }), 60);
        checkAgainstReference(
            new AdamW({ learningRate: 0.02, weightDecay: 0.3, amsgrad: true }),
            adamReference({ lr: 0.02, weightDecay: 0.3, amsgrad: true }),
            60,
        );
    });

    it("RMSprop (plain, centered, momentum)", () => {
        checkAgainstReference(new RMSprop({ learningRate: 0.01 }), rmspropReference({ lr: 0.01 }));
        checkAgainstReference(
            new RMSprop({ learningRate: 0.01, rho: 0.8, centered: true }),
            rmspropReference({ lr: 0.01, rho: 0.8, centered: true }),
        );
        checkAgainstReference(
            new RMSprop({ learningRate: 0.01, momentum: 0.5, centered: true, epsilon: 1e-5 }),
            rmspropReference({ lr: 0.01, momentum: 0.5, centered: true, epsilon: 1e-5 }),
        );
    });

    it("Adagrad", () => {
        checkAgainstReference(new Adagrad({ learningRate: 0.1 }), adagradReference({ lr: 0.1 }));
        checkAgainstReference(
            new Adagrad({ learningRate: 0.1, initialAccumulatorValue: 0, epsilon: 1e-3 }),
            adagradReference({ lr: 0.1, initialAccumulatorValue: 0, epsilon: 1e-3 }),
        );
    });

    it("keeps independent state per parameter", () => {
        const opt = new Adam({ learningRate: 0.1 });
        const a = makeParam([1]);
        const b = makeParam([1]);
        const refA = adamReference({ lr: 0.1 });
        const refB = adamReference({ lr: 0.1 });
        const wa = [1];
        const wb = [1];
        for (let t = 1; t <= 5; t++) {
            a.grad.data[0] = t;
            b.grad.data[0] = -2 * t;
            opt.step([a, b]);
            refA(wa, [t]);
            refB(wb, [-2 * t]);
        }
        assertClose(a.value.data, wa);
        assertClose(b.value.data, wb);
    });
});

// ---------------------------------------------------------------------------------------------
// Convergence
// ---------------------------------------------------------------------------------------------

describe("convergence", () => {
    const cases: [string, () => Optimizer, number, number][] = [
        ["SGD", () => new SGD({ learningRate: 0.1 }), 200, 1e-8],
        ["SGD momentum", () => new SGD({ learningRate: 0.05, momentum: 0.9 }), 500, 1e-6],
        ["SGD Nesterov", () => new SGD({ learningRate: 0.05, momentum: 0.9, nesterov: true }), 500, 1e-6],
        ["Adam", () => new Adam({ learningRate: 0.05 }), 2000, 1e-6],
        ["Adam AMSGrad", () => new Adam({ learningRate: 0.05, amsgrad: true }), 2000, 1e-6],
        ["AdamW (no decay)", () => new AdamW({ learningRate: 0.05, weightDecay: 0 }), 2000, 1e-6],
        ["RMSprop", () => new RMSprop({ learningRate: 0.01 }), 2000, 1e-6],
        ["RMSprop centered+momentum", () => new RMSprop({ learningRate: 0.005, centered: true, momentum: 0.5 }), 2000, 1e-2],
        ["Adagrad", () => new Adagrad({ learningRate: 0.5 }), 1000, 1e-6],
    ];
    for (const [label, create, steps, tolerance] of cases) {
        it(`${label} minimizes Σ(w − c)²`, () => {
            const error = minimizeQuadratic(create(), steps);
            assert.ok(error < tolerance, `${label}: max |w − c| = ${error}`);
        });
    }

    it("AdamW with decay converges near (slightly shrunk toward 0) the minimum", () => {
        const error = minimizeQuadratic(new AdamW({ learningRate: 0.05, weightDecay: 0.01 }), 2000);
        assert.ok(error > 1e-4 && error < 0.02, `max |w − c| = ${error}`);
    });

    it("Adam solves the Rosenbrock function", () => {
        const p = makeParam([-1.2, 1]);
        const opt = new Adam({ learningRate: 0.02 });
        for (let s = 0; s < 5000; s++) {
            const [x, y] = p.value.data;
            p.grad.data[0] = -2 * (1 - x) - 400 * x * (y - x * x);
            p.grad.data[1] = 200 * (y - x * x);
            opt.step([p]);
        }
        assertClose(p.value.data, [1, 1], 1e-4, "rosenbrock");
    });

    it("Adam handles an ill-conditioned quadratic (condition number 1e4)", () => {
        const p = makeParam([1, 1]);
        const opt = new Adam({ learningRate: 0.05 });
        for (let s = 0; s < 2000; s++) {
            p.grad.data[0] = 0.1 * p.value.data[0];
            p.grad.data[1] = 1000 * p.value.data[1];
            opt.step([p]);
        }
        assertClose(p.value.data, [0, 0], 1e-4, "ill-conditioned");
    });
});

// ---------------------------------------------------------------------------------------------
// Parameter flags
// ---------------------------------------------------------------------------------------------

function allOptimizers(): Optimizer[] {
    return [
        new SGD({ learningRate: 0.1, momentum: 0.9 }),
        new Adam({ learningRate: 0.1 }),
        new AdamW({ learningRate: 0.1 }),
        new RMSprop({ learningRate: 0.1 }),
        new Adagrad({ learningRate: 0.1 }),
    ];
}

describe("parameter flags", () => {
    it("trainable=false parameters are left untouched", () => {
        for (const opt of allOptimizers()) {
            const live = makeParam([1, 2]);
            const frozen = makeParam([1, 2], { trainable: false });
            live.grad.data.set([1, -1]);
            frozen.grad.data.set([1, -1]);
            opt.step([live, frozen]);
            opt.step([live, frozen]);
            assert.deepEqual(Array.from(frozen.value.data), [1, 2], opt.name);
            assert.notDeepEqual(Array.from(live.value.data), [1, 2], opt.name);
            assert.equal(opt.iterations, 2);
        }
    });

    it("frozen parameters are not shape-checked and can be unfrozen later", () => {
        const opt = new SGD({ learningRate: 0.1 });
        const p = makeParam([1]);
        p.trainable = false;
        opt.step([p]);
        assert.equal(p.value.data[0], 1);
        p.trainable = true;
        p.grad.data[0] = 1;
        opt.step([p]);
        assertClose(p.value.data, [0.9]);
    });

    it("SGD weight decay skips regularize=false parameters", () => {
        for (const momentum of [0, 0.9]) {
            const opt = new SGD({ learningRate: 0.1, momentum, weightDecay: 0.5 });
            const kernel = makeParam([2, -4]);
            const bias = makeParam([2, -4], { regularize: false });
            opt.step([kernel, bias]); // zero gradients: only decay acts
            assertClose(kernel.value.data, [2 * 0.95, -4 * 0.95]);
            assert.deepEqual(Array.from(bias.value.data), [2, -4]);
        }
    });

    it("AdamW decoupled decay skips regularize=false parameters", () => {
        const opt = new AdamW({ learningRate: 0.1, weightDecay: 0.2 });
        const kernel = makeParam([2, -4]);
        const bias = makeParam([2, -4], { regularize: false });
        opt.step([kernel, bias]); // zero gradients: Adam step is 0, only decay acts
        assertClose(kernel.value.data, [2 * 0.98, -4 * 0.98]);
        assert.deepEqual(Array.from(bias.value.data), [2, -4]);
    });

    it("Adam applies no weight decay", () => {
        const p = makeParam([2, -4]);
        new Adam({ learningRate: 0.1 }).step([p]);
        assert.deepEqual(Array.from(p.value.data), [2, -4]);
    });
});

// ---------------------------------------------------------------------------------------------
// Gradient clipping
// ---------------------------------------------------------------------------------------------

describe("gradient clipping", () => {
    it("clipNorm rescales each parameter's gradient independently and leaves grad untouched", () => {
        const opt = new SGD({ learningRate: 1, clipNorm: 1 });
        const big = makeParam([0, 0]);
        const small = makeParam([0, 0]);
        big.grad.data.set([3, 4]); // norm 5 → [0.6, 0.8]
        small.grad.data.set([0.3, 0.4]); // norm 0.5 → unchanged
        opt.step([big, small]);
        assertClose(big.value.data, [-0.6, -0.8]);
        assertClose(small.value.data, [-0.3, -0.4]);
        assert.deepEqual(Array.from(big.grad.data), [3, 4]);
        assert.deepEqual(Array.from(small.grad.data), [0.3, 0.4]);
    });

    it("clipValue clamps each element and leaves grad untouched", () => {
        const p = makeParam([0, 0, 0]);
        p.grad.data.set([3, -0.1, -7]);
        new SGD({ learningRate: 1, clipValue: 0.5 }).step([p]);
        assertClose(p.value.data, [-0.5, 0.1, 0.5]);
        assert.deepEqual(Array.from(p.grad.data), [3, -0.1, -7]);
    });

    it("applies clipNorm before clipValue", () => {
        const p = makeParam([0, 0]);
        p.grad.data.set([3, 4]);
        new SGD({ learningRate: 1, clipNorm: 1, clipValue: 0.7 }).step([p]);
        assertClose(p.value.data, [-0.6, -0.7]);
    });

    it("weight decay is added after clipping", () => {
        const p = makeParam([1]);
        p.grad.data[0] = 10;
        new SGD({ learningRate: 0.1, clipValue: 1, weightDecay: 0.5 }).step([p]);
        assertClose(p.value.data, [1 - 0.1 * (1 + 0.5)]);
    });

    it("adaptive optimizers see the clipped gradient", () => {
        const raw = [[5, -0.2, 12], [-3, 0.4, 0.1], [0.05, -9, 2]];
        const clipped = new Adam({ learningRate: 0.1, clipNorm: 1.5, clipValue: 0.9 });
        const plain = new Adam({ learningRate: 0.1 });
        const a = makeParam([1, 2, 3]);
        const b = makeParam([1, 2, 3]);
        for (const g of raw) {
            a.grad.data.set(g);
            const norm = Math.hypot(...g);
            const factor = norm > 1.5 ? 1.5 / norm : 1;
            b.grad.data.set(g.map((x) => Math.max(-0.9, Math.min(0.9, x * factor))));
            clipped.step([a]);
            plain.step([b]);
            assert.deepEqual(Array.from(a.grad.data), g);
        }
        assertClose(a.value.data, b.value.data);
    });
});

// ---------------------------------------------------------------------------------------------
// State, learning rate, shapes
// ---------------------------------------------------------------------------------------------

describe("optimizer state", () => {
    it("iterations counts step calls (even with no parameters)", () => {
        const opt = new Adam();
        assert.equal(opt.iterations, 0);
        opt.step([]);
        opt.step([makeParam([1])]);
        assert.equal(opt.iterations, 2);
    });

    it("reset() clears moments and the iteration count", () => {
        const configs = [
            () => new SGD({ learningRate: 0.1, momentum: 0.9 }),
            () => new Adam({ learningRate: 0.1, amsgrad: true }),
            () => new AdamW({ learningRate: 0.1 }),
            () => new RMSprop({ learningRate: 0.1, centered: true, momentum: 0.5 }),
            () => new Adagrad({ learningRate: 0.1 }),
        ];
        for (const create of configs) {
            const used = create();
            const p = makeParam([1, -1]);
            for (let t = 0; t < 5; t++) {
                p.grad.data.set([t + 1, -2 * t]);
                used.step([p]);
            }
            used.reset();
            assert.equal(used.iterations, 0, used.name);

            const q = makeParam([1, -1]);
            const r = makeParam([1, -1]);
            q.grad.data.set([0.3, -0.7]);
            r.grad.data.set([0.3, -0.7]);
            // `p` also had its state cleared: stepping it now must behave like a fresh first step.
            p.value.data.set([1, -1]);
            p.grad.data.set([0.3, -0.7]);
            used.step([p, q]);
            create().step([r]);
            assertClose(q.value.data, r.value.data, 1e-15, used.name);
            assertClose(p.value.data, r.value.data, 1e-15, used.name);
            assert.equal(used.iterations, 1);
        }
    });

    it("learningRate can be changed between steps", () => {
        const opt = new SGD({ learningRate: 0.1 });
        const p = makeParam([1]);
        p.grad.data[0] = 1;
        opt.step([p]);
        opt.learningRate = 0.5;
        opt.step([p]);
        assertClose(p.value.data, [1 - 0.1 - 0.5]);
        assert.equal(opt.getConfig().learningRate, 0.5);
    });

    it("learningRate setter rejects invalid values", () => {
        const opt = new Adam();
        for (const bad of [-0.1, Number.NaN, Number.POSITIVE_INFINITY]) {
            assert.throws(() => {
                opt.learningRate = bad;
            }, ValidationError);
        }
        assert.equal(opt.learningRate, 0.001);
    });

    it("throws ShapeError for a value/grad shape mismatch without mutating anything", () => {
        const opt = new SGD({ learningRate: 0.1 });
        const good = makeParam([1, 2]);
        good.grad.data.set([1, 1]);
        const bad: Parameter = { ...makeParam([1, 2]), name: "bad", grad: new Matrix(2, 1) };
        assert.throws(() => opt.step([good, bad]), (error: unknown) => {
            assert.ok(error instanceof ShapeError);
            assert.match(error.message, /bad/);
            return true;
        });
        assert.deepEqual(Array.from(good.value.data), [1, 2]);
        assert.equal(opt.iterations, 0);
    });
});

// ---------------------------------------------------------------------------------------------
// Configs & registry
// ---------------------------------------------------------------------------------------------

describe("configs", () => {
    it("defaults match the documented values", () => {
        assert.deepEqual(new SGD().getConfig(), {
            name: "sgd",
            learningRate: 0.01,
            momentum: 0,
            nesterov: false,
            weightDecay: 0,
        });
        assert.deepEqual(new Adam().getConfig(), {
            name: "adam",
            learningRate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-7,
            amsgrad: false,
        });
        assert.deepEqual(new AdamW().getConfig(), {
            name: "adamw",
            learningRate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-7,
            amsgrad: false,
            weightDecay: 0.01,
        });
        assert.deepEqual(new RMSprop().getConfig(), {
            name: "rmsprop",
            learningRate: 0.001,
            rho: 0.9,
            momentum: 0,
            epsilon: 1e-7,
            centered: false,
        });
        assert.deepEqual(new Adagrad().getConfig(), {
            name: "adagrad",
            learningRate: 0.01,
            initialAccumulatorValue: 0.1,
            epsilon: 1e-7,
        });
    });

    it("round-trips through JSON and getOptimizer with identical behaviour", () => {
        const originals: Optimizer[] = [
            new SGD({ learningRate: 0.03, momentum: 0.7, nesterov: true, weightDecay: 1e-4, clipNorm: 2 }),
            new Adam({ learningRate: 0.02, beta1: 0.8, beta2: 0.99, epsilon: 1e-8, amsgrad: true, clipValue: 0.5 }),
            new AdamW({ learningRate: 0.01, weightDecay: 0.05, clipNorm: 1, clipValue: 3 }),
            new RMSprop({ learningRate: 0.004, rho: 0.95, momentum: 0.3, epsilon: 1e-6, centered: true }),
            new Adagrad({ learningRate: 0.2, initialAccumulatorValue: 0.5, epsilon: 1e-5 }),
        ];
        for (const original of originals) {
            const config = original.getConfig();
            const json = JSON.parse(JSON.stringify(config)) as OptimizerConfig;
            assert.deepEqual(json, config, original.name);
            const copy = getOptimizer(json);
            assert.notEqual(copy, original);
            assert.equal(copy.constructor, original.constructor);
            assert.deepEqual(copy.getConfig(), config);

            const a = makeParam([0.5, -1, 2, 3], { rows: 2 });
            const b = makeParam([0.5, -1, 2, 3], { rows: 2 });
            for (let t = 1; t <= 10; t++) {
                const g = wobblyGradient(a.value.data, t);
                a.grad.data.set(g);
                b.grad.data.set(g);
                original.step([a]);
                copy.step([b]);
            }
            assert.deepEqual(Array.from(a.value.data), Array.from(b.value.data), original.name);
        }
    });

    it("omits unset clipping options from the config", () => {
        const config = new Adam().getConfig();
        assert.equal("clipNorm" in config, false);
        assert.equal("clipValue" in config, false);
    });

    it("rejects invalid hyperparameters", () => {
        const invalid: [string, () => unknown][] = [
            ["negative learning rate", () => new SGD({ learningRate: -0.1 })],
            ["NaN learning rate", () => new Adam({ learningRate: Number.NaN })],
            ["infinite learning rate", () => new Adam({ learningRate: Number.POSITIVE_INFINITY })],
            ["momentum = 1", () => new SGD({ momentum: 1 })],
            ["negative momentum", () => new SGD({ momentum: -0.5 })],
            ["negative weight decay", () => new SGD({ weightDecay: -1 })],
            ["non-boolean nesterov", () => new SGD({ nesterov: 1 as unknown as boolean })],
            ["beta1 = 1", () => new Adam({ beta1: 1 })],
            ["beta2 < 0", () => new Adam({ beta2: -0.1 })],
            ["epsilon = 0", () => new Adam({ epsilon: 0 })],
            ["string epsilon", () => new Adam({ epsilon: "1e-7" as unknown as number })],
            ["AdamW negative decay", () => new AdamW({ weightDecay: -0.01 })],
            ["rho = 1.5", () => new RMSprop({ rho: 1.5 })],
            ["RMSprop momentum = 1", () => new RMSprop({ momentum: 1 })],
            ["negative accumulator", () => new Adagrad({ initialAccumulatorValue: -0.1 })],
            ["clipNorm = 0", () => new SGD({ clipNorm: 0 })],
            ["negative clipValue", () => new Adam({ clipValue: -1 })],
            ["unknown option", () => new Adam({ learningrate: 0.1 } as never)],
            ["weightDecay on Adam", () => new Adam({ weightDecay: 0.1 } as never)],
            ["non-object options", () => new SGD(0.1 as never)],
            ["config with bad value", () => getOptimizer({ name: "rmsprop", rho: -1 })],
        ];
        for (const [label, create] of invalid) {
            assert.throws(create, ValidationError, label);
        }
    });

    it("error messages name the optimizer and option", () => {
        assert.throws(() => new Adam({ beta1: 1 }), /Adam: "beta1" must be a number in \[0, 1\), got 1/);
        assert.throws(() => new SGD({ lr: 1 } as never), /SGD: unknown option "lr". Valid options: learningRate/);
    });
});

describe("getOptimizer", () => {
    it("exports the list of names", () => {
        assert.deepEqual([...OPTIMIZER_NAMES], ["sgd", "adam", "adamw", "rmsprop", "adagrad"]);
    });

    it("builds each optimizer by name with defaults", () => {
        const classes = { sgd: SGD, adam: Adam, adamw: AdamW, rmsprop: RMSprop, adagrad: Adagrad };
        for (const name of OPTIMIZER_NAMES) {
            const opt = getOptimizer(name);
            assert.equal(opt.name, name);
            assert.equal(opt.constructor, classes[name]);
            assert.deepEqual(opt.getConfig(), new classes[name]().getConfig());
        }
    });

    it("accepts names case-insensitively", () => {
        assert.ok(getOptimizer("Adam" as never) instanceof Adam);
        assert.equal(getOptimizer("RMSProp" as never).name, "rmsprop");
    });

    it("builds from a partial config", () => {
        const opt = getOptimizer({ name: "sgd", learningRate: 0.5, momentum: 0.9 });
        assert.ok(opt instanceof SGD);
        assert.equal(opt.learningRate, 0.5);
        assert.equal(opt.momentum, 0.9);
        assert.equal(opt.nesterov, false);
    });

    it("treats null config values as unset", () => {
        const opt = getOptimizer(JSON.parse('{"name":"adam","clipNorm":null,"beta1":null}') as OptimizerConfig);
        assert.deepEqual(opt.getConfig(), new Adam().getConfig());
    });

    it("returns instances as-is", () => {
        const opt = new Adagrad();
        assert.equal(getOptimizer(opt), opt);
        const custom: Optimizer = {
            name: "sgd",
            learningRate: 1,
            iterations: 0,
            step() {},
            reset() {},
            getConfig: () => ({ name: "sgd" }),
        };
        assert.equal(getOptimizer(custom), custom);
    });

    it("rejects unknown names, listing the valid ones", () => {
        assert.throws(() => getOptimizer("adadelta" as never), (error: unknown) => {
            assert.ok(error instanceof ValidationError);
            for (const name of OPTIMIZER_NAMES) assert.ok(error.message.includes(name), error.message);
            assert.match(error.message, /adadelta/);
            return true;
        });
        assert.throws(() => getOptimizer({ name: "nope" } as never), ValidationError);
        assert.throws(() => getOptimizer({} as never), ValidationError);
        assert.throws(() => getOptimizer(42 as never), ValidationError);
        assert.throws(() => getOptimizer(null as never), ValidationError);
    });
});
