import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { ACTIVATION_NAMES, getActivation } from "../src/activations.js";
import { Matrix } from "../src/core/matrix.js";
import { Random } from "../src/core/random.js";
import { ShapeError, ValidationError } from "../src/core/errors.js";
import type { Activation, ActivationConfig, ActivationIdentifier } from "../src/core/types.js";

function randomMatrix(rng: Random, rows: number, cols: number, lo: number, hi: number): Matrix {
    const m = new Matrix(rows, cols);
    for (let i = 0; i < m.data.length; i++) m.data[i] = rng.uniform(lo, hi);
    return m;
}

function assertClose(actual: number, expected: number, tol = 1e-12, label = ""): void {
    const scale = Math.max(1, Math.abs(expected));
    assert.ok(
        Math.abs(actual - expected) <= tol * scale,
        `${label} expected ${expected}, got ${actual} (diff ${Math.abs(actual - expected)})`,
    );
}

function assertAllClose(actual: Matrix, expected: Matrix, tol = 1e-12, label = ""): void {
    assert.deepEqual(actual.shape, expected.shape, `${label} shape`);
    for (let i = 0; i < actual.data.length; i++) assertClose(actual.data[i], expected.data[i], tol, `${label}[${i}]`);
}

/** Evaluates a scalar activation on one value via a 1x1 matrix. */
function f(id: ActivationIdentifier, x: number): number {
    return getActivation(id).forward(Matrix.fromVector([x])).data[0];
}

/** Moves values away from kinks where the derivative is undefined (finite differences break there). */
function avoidKinks(m: Matrix, kinks: number[], margin = 1e-2): void {
    for (let i = 0; i < m.data.length; i++) {
        for (const k of kinks) {
            if (Math.abs(m.data[i] - k) < margin) m.data[i] = k + 2 * margin;
        }
    }
}

const KINKS: Record<string, number[]> = {
    relu: [0],
    relu6: [0, 6],
    leakyRelu: [0],
    elu: [0],
    hardSigmoid: [-2.5, 2.5],
};

/** Every activation, plus parameterized variants. */
const VARIANTS: ActivationIdentifier[] = [
    ...ACTIVATION_NAMES,
    { name: "leakyRelu", alpha: 0.2 },
    { name: "elu", alpha: 0.5 },
];

function variantLabel(id: ActivationIdentifier): string {
    return typeof id === "string" ? id : JSON.stringify(id);
}

describe("activations: registry", () => {
    it("exposes every activation name and resolves each one", () => {
        assert.equal(ACTIVATION_NAMES.length, 15);
        assert.equal(new Set(ACTIVATION_NAMES).size, 15);
        for (const name of ACTIVATION_NAMES) {
            const activation = getActivation(name);
            assert.equal(activation.name, name);
            assert.deepEqual(activation.getConfig().name, name);
        }
    });

    it("returns an existing instance as-is", () => {
        const relu = getActivation("relu");
        assert.equal(getActivation(relu), relu);
    });

    it("accepts user-defined activations", () => {
        const custom: Activation = {
            name: "linear",
            forward: (z) => z,
            backward: (_z, _a, g) => g,
            getConfig: () => ({ name: "linear" }),
        };
        assert.equal(getActivation(custom), custom);
    });

    it("rejects unknown names and lists the valid ones", () => {
        assert.throws(() => getActivation("nope" as never), (err: unknown) => {
            assert.ok(err instanceof ValidationError);
            assert.match(err.message, /nope/);
            for (const name of ACTIVATION_NAMES) assert.ok(err.message.includes(name), `message lists ${name}`);
            return true;
        });
        assert.throws(() => getActivation({ name: "bogus" } as never), ValidationError);
        assert.throws(() => getActivation(null as never), ValidationError);
        assert.throws(() => getActivation(42 as never), ValidationError);
    });

    it("rejects unknown and invalid parameters", () => {
        assert.throws(() => getActivation({ name: "relu", alpha: 0.1 }), ValidationError);
        assert.throws(() => getActivation({ name: "leakyRelu", alpa: 0.1 }), /alpa/);
        assert.throws(() => getActivation({ name: "leakyRelu", alpha: "0.1" }), ValidationError);
        assert.throws(() => getActivation({ name: "elu", alpha: Number.NaN }), ValidationError);
        assert.throws(() => getActivation({ name: "elu", alpha: Infinity }), ValidationError);
    });

    it("ignores undefined config values", () => {
        const a = getActivation({ name: "leakyRelu", alpha: undefined });
        assert.deepEqual(a.getConfig(), { name: "leakyRelu", alpha: 0.01 });
    });
});

describe("activations: known values", () => {
    it("linear", () => {
        assert.equal(f("linear", -3.5), -3.5);
        assert.equal(f("linear", 2), 2);
    });

    it("sigmoid", () => {
        assert.equal(f("sigmoid", 0), 0.5);
        assertClose(f("sigmoid", 2), 1 / (1 + Math.exp(-2)));
        assertClose(f("sigmoid", -2), 1 / (1 + Math.exp(2)));
    });

    it("sigmoid is stable for large |z|", () => {
        assert.equal(f("sigmoid", 1000), 1);
        assert.equal(f("sigmoid", -1000), 0);
        assertClose(f("sigmoid", -40), Math.exp(-40) / (1 + Math.exp(-40)), 1e-12);
        assert.ok(f("sigmoid", -40) > 0, "keeps precision in the lower tail");
    });

    it("tanh", () => {
        assertClose(f("tanh", 0.5), Math.tanh(0.5));
        assert.equal(f("tanh", 0), 0);
    });

    it("relu and relu6", () => {
        assert.equal(f("relu", -1), 0);
        assert.equal(f("relu", 2), 2);
        assert.equal(f("relu6", -1), 0);
        assert.equal(f("relu6", 3), 3);
        assert.equal(f("relu6", 7), 6);
    });

    it("leakyRelu defaults to alpha 0.01", () => {
        assertClose(f("leakyRelu", -2), -0.02);
        assert.equal(f("leakyRelu", 3), 3);
        assertClose(f({ name: "leakyRelu", alpha: 0.2 }, -2), -0.4);
    });

    it("elu defaults to alpha 1", () => {
        assertClose(f("elu", -1), Math.exp(-1) - 1);
        assert.equal(f("elu", 2), 2);
        assertClose(f({ name: "elu", alpha: 0.5 }, -1), 0.5 * (Math.exp(-1) - 1));
    });

    it("selu uses the standard constants", () => {
        const scale = 1.0507009873554805;
        const alpha = 1.6732632423543772;
        assertClose(f("selu", 1), scale);
        assertClose(f("selu", -1), scale * alpha * (Math.exp(-1) - 1));
        assert.equal(f("selu", 0), 0);
    });

    it("gelu (tanh approximation)", () => {
        const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
        assertClose(f("gelu", 1), gelu(1));
        assertClose(f("gelu", -0.7), gelu(-0.7));
        assertClose(f("gelu", 1), 0.8411919906082768, 1e-12);
    });

    it("swish", () => {
        assertClose(f("swish", 1), 1 / (1 + Math.exp(-1)));
        assert.equal(f("swish", 0), 0);
        assert.equal(Math.abs(f("swish", -1000)), 0);
    });

    it("swish derivative is σ(z)·(1 + z·(1 - σ(z)))", () => {
        const swish = getActivation("swish");
        const z = Matrix.fromVector([1, -2]);
        const d = swish.backward(z, swish.forward(z), Matrix.fromVector([1, 1]));
        const s = (x: number) => 1 / (1 + Math.exp(-x));
        assertClose(d.data[0], s(1) * (1 + 1 * (1 - s(1))));
        assertClose(d.data[1], s(-2) * (1 - 2 * (1 - s(-2))));
        assertClose(d.data[0], 0.9276705118714869, 1e-12);
    });

    it("mish", () => {
        assertClose(f("mish", 1), Math.tanh(Math.log1p(Math.E)));
        assertClose(f("mish", -1), -Math.tanh(Math.log1p(Math.exp(-1))));
        assert.equal(f("mish", 1000), 1000);
        assert.ok(Number.isFinite(f("mish", -1000)));
    });

    it("softplus is stable", () => {
        assertClose(f("softplus", 0), Math.LN2);
        assertClose(f("softplus", 1), Math.log(1 + Math.E));
        assert.equal(f("softplus", 1000), 1000);
        assert.equal(f("softplus", -1000), 0);
        assertClose(f("softplus", -30), Math.exp(-30), 1e-12);
    });

    it("softsign", () => {
        assert.equal(f("softsign", 1), 0.5);
        assert.equal(f("softsign", -3), -0.75);
    });

    it("hardSigmoid is clip(0.2z + 0.5, 0, 1)", () => {
        assert.equal(f("hardSigmoid", 0), 0.5);
        assertClose(f("hardSigmoid", 1), 0.7);
        assert.equal(f("hardSigmoid", 3), 1);
        assert.equal(f("hardSigmoid", -3), 0);
    });

    it("softmax is row-wise and sums to 1", () => {
        const softmax = getActivation("softmax");
        const a = softmax.forward(Matrix.fromArray([[1, 2, 3], [0, 0, 0]]));
        const expected = [0.09003057317038046, 0.24472847105479764, 0.6652409557748219];
        for (let i = 0; i < 3; i++) assertClose(a.get(0, i), expected[i]);
        for (let i = 0; i < 3; i++) assertClose(a.get(1, i), 1 / 3);
        for (let r = 0; r < 2; r++) assertClose(a.row(r).reduce((s, v) => s + v, 0), 1);
    });

    it("softmax is shift-invariant and stable for large inputs", () => {
        const softmax = getActivation("softmax");
        const big = softmax.forward(Matrix.fromArray([[1000, 1001, 1002], [-1000, -1001, -1002]]));
        const ref = softmax.forward(Matrix.fromArray([[0, 1, 2]]));
        for (let i = 0; i < 3; i++) assertClose(big.get(0, i), ref.get(0, i));
        for (const v of big.data) assert.ok(Number.isFinite(v));
        assertClose(big.get(1, 0), ref.get(0, 2));
    });
});

describe("activations: gradients (central finite differences)", () => {
    const eps = 1e-5;
    for (const id of VARIANTS) {
        const label = variantLabel(id);
        it(`backward matches finite differences for ${label}`, () => {
            const activation = getActivation(id);
            const rng = new Random(1234);
            const z = randomMatrix(rng, 4, 5, -4, 7.5);
            avoidKinks(z, KINKS[activation.name] ?? []);
            const g = randomMatrix(rng, 4, 5, -2, 2);
            const analytic = activation.backward(z, activation.forward(z), g);

            // L(z) = Σ g ⊙ f(z), so dL/dz is exactly the vector-Jacobian product backward computes.
            const objective = (m: Matrix): number => {
                const a = activation.forward(m);
                let sum = 0;
                for (let i = 0; i < a.data.length; i++) sum += a.data[i] * g.data[i];
                return sum;
            };
            for (let i = 0; i < z.data.length; i++) {
                const plus = z.clone();
                const minus = z.clone();
                plus.data[i] += eps;
                minus.data[i] -= eps;
                const numeric = (objective(plus) - objective(minus)) / (2 * eps);
                assertClose(analytic.data[i], numeric, 1e-6, `${label} dz[${i}]`);
            }
        });
    }

    it("softmax backward is the full Jacobian-vector product (not the diagonal)", () => {
        const softmax = getActivation("softmax");
        const z = Matrix.fromArray([[0.3, -1.2, 2.0, 0.5]]);
        const a = softmax.forward(z);
        const g = Matrix.fromArray([[0.7, -1.3, 0.2, 2.1]]);
        const d = softmax.backward(z, a, g);
        let dot = 0;
        for (let j = 0; j < 4; j++) dot += a.data[j] * g.data[j];
        for (let i = 0; i < 4; i++) {
            let jvp = 0;
            for (let j = 0; j < 4; j++) {
                const jac = i === j ? a.data[i] * (1 - a.data[i]) : -a.data[i] * a.data[j];
                jvp += jac * g.data[j];
            }
            assertClose(d.data[i], jvp);
            assertClose(d.data[i], a.data[i] * (g.data[i] - dot));
        }
        // Gradient of a softmax output sums to zero per row.
        assertClose(d.data.reduce((s, v) => s + v, 0), 0, 1e-12);
    });

    it("gradients stay finite for large |z|", () => {
        const z = Matrix.fromVector([-1000, -50, 50, 1000]);
        const g = Matrix.fromVector([1, 1, 1, 1]);
        for (const name of ACTIVATION_NAMES) {
            const activation = getActivation(name);
            const a = activation.forward(z);
            const d = activation.backward(z, a, g);
            for (const v of a.data) assert.ok(Number.isFinite(v), `${name} forward finite`);
            for (const v of d.data) assert.ok(Number.isFinite(v), `${name} backward finite`);
        }
    });
});

describe("activations: buffers and shapes", () => {
    it("writes into a provided output buffer and supports in-place use", () => {
        const rng = new Random(7);
        for (const name of ACTIVATION_NAMES) {
            const activation = getActivation(name);
            const z = randomMatrix(rng, 3, 4, -3, 3);
            const g = randomMatrix(rng, 3, 4, -1, 1);
            const expectedA = activation.forward(z);
            const expectedD = activation.backward(z, expectedA, g);

            const out = new Matrix(3, 4);
            assert.equal(activation.forward(z, out), out);
            assertAllClose(out, expectedA, 0, name);

            const dOut = new Matrix(3, 4);
            assert.equal(activation.backward(z, expectedA, g, dOut), dOut);
            assertAllClose(dOut, expectedD, 0, name);

            const inPlaceZ = z.clone();
            activation.forward(inPlaceZ, inPlaceZ);
            assertAllClose(inPlaceZ, expectedA, 0, `${name} in-place forward`);

            const inPlaceG = g.clone();
            activation.backward(z, expectedA, inPlaceG, inPlaceG);
            assertAllClose(inPlaceG, expectedD, 0, `${name} in-place backward`);
        }
    });

    it("allocates a fresh output when none is given", () => {
        const z = Matrix.fromVector([1, 2]);
        for (const name of ACTIVATION_NAMES) {
            const a = getActivation(name).forward(z);
            assert.notEqual(a, z, name);
            assert.notEqual(a.data, z.data, name);
        }
    });

    it("throws ShapeError for mismatched buffers", () => {
        const z = new Matrix(2, 3);
        for (const name of ACTIVATION_NAMES) {
            const activation = getActivation(name);
            assert.throws(() => activation.forward(z, new Matrix(3, 2)), ShapeError, name);
            assert.throws(() => activation.backward(z, new Matrix(2, 3), new Matrix(2, 2)), ShapeError, name);
            assert.throws(() => activation.backward(z, new Matrix(1, 3), new Matrix(2, 3)), ShapeError, name);
            assert.throws(
                () => activation.backward(z, new Matrix(2, 3), new Matrix(2, 3), new Matrix(2, 4)),
                ShapeError,
                name,
            );
        }
    });
});

describe("activations: config round-trip", () => {
    for (const id of VARIANTS) {
        const label = variantLabel(id);
        it(`getActivation(getConfig()) behaves identically for ${label}`, () => {
            const original = getActivation(id);
            const config: ActivationConfig = JSON.parse(JSON.stringify(original.getConfig()));
            const restored = getActivation(config);
            assert.equal(restored.name, original.name);
            assert.deepEqual(restored.getConfig(), original.getConfig());

            const rng = new Random(99);
            const z = randomMatrix(rng, 3, 4, -5, 5);
            const g = randomMatrix(rng, 3, 4, -1, 1);
            const a1 = original.forward(z);
            const a2 = restored.forward(z);
            assertAllClose(a2, a1, 0, label);
            assertAllClose(restored.backward(z, a2, g), original.backward(z, a1, g), 0, label);
        });
    }

    it("serializes parameters", () => {
        assert.deepEqual(getActivation("leakyRelu").getConfig(), { name: "leakyRelu", alpha: 0.01 });
        assert.deepEqual(getActivation("elu").getConfig(), { name: "elu", alpha: 1 });
        assert.deepEqual(getActivation({ name: "leakyRelu", alpha: 0.3 }).getConfig(), { name: "leakyRelu", alpha: 0.3 });
        assert.deepEqual(getActivation("relu").getConfig(), { name: "relu" });
    });
});
