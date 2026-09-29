import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { INITIALIZER_NAMES, getInitializer } from "../src/initializers.js";
import { Matrix } from "../src/core/matrix.js";
import { Random } from "../src/core/random.js";
import { ValidationError } from "../src/core/errors.js";
import type { Initializer, InitializerIdentifier } from "../src/core/types.js";

/** Standard deviation of a unit normal truncated to [-2, 2]. */
const TRUNCATION_CORRECTION = 0.87962566103423978;

function stats(data: Float64Array): { mean: number; variance: number; min: number; max: number } {
    let sum = 0;
    let min = Infinity;
    let max = -Infinity;
    for (const v of data) {
        sum += v;
        if (v < min) min = v;
        if (v > max) max = v;
    }
    const mean = sum / data.length;
    let sq = 0;
    for (const v of data) sq += (v - mean) ** 2;
    return { mean, variance: sq / (data.length - 1), min, max };
}

function sample(id: InitializerIdentifier, rows: number, cols: number, seed = 1): Matrix {
    const m = new Matrix(rows, cols);
    getInitializer(id).initialize(m, rows, cols, new Random(seed));
    return m;
}

/** Asserts the sample variance is within `rel` of the target, and the mean within 4 standard errors of `mean`. */
function assertMoments(m: Matrix, mean: number, variance: number, rel = 0.05, label = ""): void {
    const s = stats(m.data);
    const stderr = Math.sqrt(variance / m.data.length);
    assert.ok(Math.abs(s.mean - mean) < 4 * stderr + 1e-12, `${label} mean ${s.mean}, expected ${mean}`);
    assert.ok(Math.abs(s.variance / variance - 1) < rel, `${label} variance ${s.variance}, expected ${variance}`);
}

describe("initializers: registry", () => {
    it("exposes every initializer name and resolves each one", () => {
        assert.equal(INITIALIZER_NAMES.length, 11);
        for (const name of INITIALIZER_NAMES) {
            const init = getInitializer(name);
            assert.equal(init.name, name);
            assert.equal(init.getConfig().name, name);
        }
    });

    it("returns an existing instance as-is", () => {
        const init = getInitializer("heNormal");
        assert.equal(getInitializer(init), init);
        const custom: Initializer = {
            name: "zeros",
            initialize: (target) => void target.fill(7),
            getConfig: () => ({ name: "zeros" }),
        };
        assert.equal(getInitializer(custom), custom);
    });

    it("round-trips configs", () => {
        const ids: InitializerIdentifier[] = [
            ...INITIALIZER_NAMES,
            { name: "constant", value: 0.25 },
            { name: "randomUniform", minval: -1, maxval: 2 },
            { name: "randomNormal", mean: 3, stddev: 0.5 },
        ];
        for (const id of ids) {
            const original = getInitializer(id);
            const restored = getInitializer(JSON.parse(JSON.stringify(original.getConfig())));
            assert.deepEqual(restored.getConfig(), original.getConfig());
            const a = new Matrix(8, 9);
            const b = new Matrix(8, 9);
            original.initialize(a, 8, 9, new Random(5));
            restored.initialize(b, 8, 9, new Random(5));
            assert.deepEqual(Array.from(a.data), Array.from(b.data), JSON.stringify(id));
        }
    });

    it("reports Keras defaults in configs", () => {
        assert.deepEqual(getInitializer("constant").getConfig(), { name: "constant", value: 0 });
        assert.deepEqual(getInitializer("randomUniform").getConfig(), { name: "randomUniform", minval: -0.05, maxval: 0.05 });
        assert.deepEqual(getInitializer("randomNormal").getConfig(), { name: "randomNormal", mean: 0, stddev: 0.05 });
        assert.deepEqual(getInitializer("glorotUniform").getConfig(), { name: "glorotUniform" });
    });

    it("rejects unknown names and bad parameters", () => {
        assert.throws(() => getInitializer("orthogonal" as never), (err: unknown) => {
            assert.ok(err instanceof ValidationError);
            for (const name of INITIALIZER_NAMES) assert.ok(err.message.includes(name), `message lists ${name}`);
            return true;
        });
        assert.throws(() => getInitializer({ name: "nope" } as never), ValidationError);
        assert.throws(() => getInitializer(3 as never), ValidationError);
        assert.throws(() => getInitializer({ name: "constant", value: "1" }), ValidationError);
        assert.throws(() => getInitializer({ name: "constant", value: Number.NaN }), ValidationError);
        assert.throws(() => getInitializer({ name: "randomUniform", minval: 1, maxval: 0 }), ValidationError);
        assert.throws(() => getInitializer({ name: "randomNormal", stddev: -1 }), ValidationError);
        assert.throws(() => getInitializer({ name: "glorotUniform", seed: 1 }), /seed/);
    });

    it("rejects invalid fans", () => {
        for (const name of INITIALIZER_NAMES) {
            const init = getInitializer(name);
            assert.throws(() => init.initialize(new Matrix(2, 2), -1, 2, new Random(1)), ValidationError, name);
            assert.throws(() => init.initialize(new Matrix(2, 2), 2, Number.NaN, new Random(1)), ValidationError, name);
        }
    });
});

describe("initializers: constants", () => {
    it("zeros, ones and constant fill every element", () => {
        const m = Matrix.filled(3, 4, 9);
        getInitializer("zeros").initialize(m, 3, 4, new Random(1));
        assert.ok(m.data.every((v) => v === 0));
        getInitializer("ones").initialize(m, 3, 4, new Random(1));
        assert.ok(m.data.every((v) => v === 1));
        getInitializer({ name: "constant", value: -2.5 }).initialize(m, 3, 4, new Random(1));
        assert.ok(m.data.every((v) => v === -2.5));
        getInitializer("constant").initialize(m, 3, 4, new Random(1));
        assert.ok(m.data.every((v) => v === 0));
    });
});

describe("initializers: distributions", () => {
    const rows = 100;
    const cols = 200;

    it("randomUniform", () => {
        const m = sample("randomUniform", rows, cols);
        const s = stats(m.data);
        assert.ok(s.min >= -0.05 && s.max < 0.05);
        assertMoments(m, 0, 0.1 ** 2 / 12, 0.05, "randomUniform default");

        const custom = sample({ name: "randomUniform", minval: 1, maxval: 3 }, rows, cols);
        const cs = stats(custom.data);
        assert.ok(cs.min >= 1 && cs.max < 3);
        assertMoments(custom, 2, 4 / 12, 0.05, "randomUniform [1, 3)");
    });

    it("randomNormal", () => {
        assertMoments(sample("randomNormal", rows, cols), 0, 0.05 ** 2, 0.05, "randomNormal default");
        assertMoments(sample({ name: "randomNormal", mean: -1, stddev: 2 }, rows, cols), -1, 4, 0.05, "randomNormal custom");
    });

    const scaling: [InitializerIdentifier, number, "uniform" | "normal"][] = [
        ["glorotUniform", 2 / (rows + cols), "uniform"],
        ["glorotNormal", 2 / (rows + cols), "normal"],
        ["heUniform", 2 / rows, "uniform"],
        ["heNormal", 2 / rows, "normal"],
        ["lecunUniform", 1 / rows, "uniform"],
        ["lecunNormal", 1 / rows, "normal"],
    ];
    for (const [id, variance, kind] of scaling) {
        it(`${String(id)} has variance ${variance.toPrecision(4)} (Keras formula)`, () => {
            const m = sample(id, rows, cols, 17);
            assertMoments(m, 0, variance, 0.05, String(id));
            const s = stats(m.data);
            if (kind === "uniform") {
                const limit = Math.sqrt(3 * variance);
                assert.ok(s.min >= -limit && s.max < limit, `${String(id)} within ±${limit}`);
                assert.ok(s.max > 0.99 * limit && s.min < -0.99 * limit, `${String(id)} spans the range`);
            } else {
                const bound = (2 * Math.sqrt(variance)) / TRUNCATION_CORRECTION;
                assert.ok(s.min >= -bound && s.max <= bound, `${String(id)} truncated at 2σ`);
                assert.ok(s.max > 0.95 * bound, `${String(id)} reaches the truncation bound`);
            }
        });
    }

    it("floors the fan at 1 instead of dividing by zero", () => {
        for (const name of ["glorotUniform", "heNormal", "lecunUniform"] as const) {
            const m = new Matrix(2, 2);
            getInitializer(name).initialize(m, 0, 0, new Random(3));
            assert.ok(m.data.every((v) => Number.isFinite(v)), name);
        }
    });
});

describe("initializers: determinism", () => {
    it("same seed gives identical weights, different seeds differ", () => {
        for (const name of INITIALIZER_NAMES) {
            const a = sample(name, 20, 30, 42);
            const b = sample(name, 20, 30, 42);
            assert.deepEqual(Array.from(a.data), Array.from(b.data), name);
            if (!["zeros", "ones", "constant"].includes(name)) {
                const c = sample(name, 20, 30, 43);
                assert.notDeepEqual(Array.from(a.data), Array.from(c.data), name);
            }
        }
    });
});
