import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { niceTicks } from "../docs/playground/chart.js";
import { generateCode } from "../docs/playground/codegen.js";
import type { PlaygroundConfig } from "../docs/playground/config.js";
import {
    classCount,
    DATASETS,
    defaultConfig,
    outputSpec,
    parameterCount,
    sanitizeConfig,
    taskKind,
} from "../docs/playground/config.js";
import {
    accuracyOf,
    DOMAIN,
    featureData,
    generateDataset,
    gridFeatureData,
    rSquaredOf,
} from "../docs/playground/datasets.js";
import { contourSegments } from "../docs/playground/heatmap.js";
import { decodeConfig, encodeConfig } from "../docs/playground/url-state.js";

function config(overrides: Partial<PlaygroundConfig> = {}): PlaygroundConfig {
    return { ...defaultConfig(), ...overrides };
}

describe("playground datasets", () => {
    it("generates every dataset deterministically from its seed", () => {
        for (const { id } of DATASETS) {
            const a = generateDataset(config({ dataset: id, noise: 20, dataSeed: 9 }));
            const b = generateDataset(config({ dataset: id, noise: 20, dataSeed: 9 }));
            assert.deepEqual(a, b, id);
            const c = generateDataset(config({ dataset: id, noise: 20, dataSeed: 10 }));
            assert.notDeepEqual(a, c, id);
        }
    });

    it("splits samples by the train ratio", () => {
        const data = generateDataset(config({ samples: 400, trainRatio: 70 }));
        assert.equal(data.train.length, 280);
        assert.equal(data.test.length, 120);
    });

    it("produces valid labels for each task", () => {
        for (const { id } of DATASETS) {
            for (const classes of [2, 3, 4]) {
                const cfg = config({ dataset: id, classes, noise: 30 });
                const { train, test } = generateDataset(cfg);
                const task = taskKind(cfg);
                for (const p of [...train, ...test]) {
                    assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y));
                    if (task === "regression") assert.ok(p.label >= -1 && p.label <= 1);
                    else assert.ok(Number.isInteger(p.label) && p.label >= 0 && p.label < classCount(cfg));
                }
            }
        }
    });

    it("computes scaled features and a top-to-bottom grid", () => {
        const data = featureData([{ x: DOMAIN, y: -DOMAIN / 2 }], ["x", "y", "x2", "xy", "sinx"]);
        assert.deepEqual(Array.from(data), [1, -0.5, 1, -0.5, Math.sin(DOMAIN)]);
        const grid = gridFeatureData(4, ["x", "y"]);
        assert.equal(grid.length, 4 * 4 * 2);
        assert.ok(grid[1] > 0, "row 0 is the top of the plot (positive y)");
        assert.ok(grid[0] < 0, "column 0 is the left of the plot (negative x)");
    });

    it("scores accuracy and R² from predictions", () => {
        const points = [
            { x: 0, y: 0, label: 1 },
            { x: 0, y: 0, label: 0 },
        ];
        assert.equal(accuracyOf([0.9, 0.2], 1, points), 1);
        assert.equal(accuracyOf([0.1, 0.9, 0.8, 0.2], 2, points), 1);
        assert.equal(accuracyOf([0.4, 0.6], 1, points), 0);
        const targets = [
            { x: 0, y: 0, label: -1 },
            { x: 0, y: 0, label: 1 },
        ];
        assert.equal(rSquaredOf([-1, 1], targets), 1);
        assert.equal(rSquaredOf([0, 0], targets), 0);
    });
});

describe("playground config", () => {
    it("derives output layer and loss from the task", () => {
        assert.deepEqual(outputSpec(config({ dataset: "circle" })), { units: 1, activation: "sigmoid", loss: "bce" });
        assert.deepEqual(outputSpec(config({ dataset: "blobs", classes: 3 })), {
            units: 3,
            activation: "softmax",
            loss: "scce",
        });
        assert.deepEqual(outputSpec(config({ dataset: "blobs", classes: 2 })).loss, "bce");
        assert.deepEqual(outputSpec(config({ dataset: "wave" })), { units: 1, activation: "linear", loss: "mse" });
    });

    it("counts parameters", () => {
        // 2 → 8 → 6 → 1: (2·8 + 8) + (8·6 + 6) + (6·1 + 1)
        assert.equal(parameterCount(config()), 24 + 54 + 7);
    });

    it("clamps and snaps out-of-range values", () => {
        const c = sanitizeConfig({
            ...defaultConfig(),
            noise: 400,
            samples: 7,
            layers: Array.from({ length: 9 }, () => ({ units: 99, activation: "nope" as "tanh", dropout: 0.9 })),
            learningRate: 0.02,
            batchSize: 20,
            features: [],
        });
        assert.equal(c.noise, 50);
        assert.equal(c.samples, 50);
        assert.equal(c.layers.length, 6);
        assert.deepEqual(c.layers[0], { units: 32, activation: "tanh", dropout: 0.5 });
        assert.equal(c.learningRate, 0.03);
        assert.equal(c.batchSize, 16);
        assert.deepEqual(c.features, ["x", "y"]);
    });
});

describe("playground URL state", () => {
    it("round-trips a configuration through the hash", () => {
        const original = config({
            dataset: "spiral",
            noise: 15,
            features: ["x", "y", "sinx"],
            layers: [
                { units: 12, activation: "relu", dropout: 0.1 },
                { units: 3, activation: "gelu", dropout: 0 },
            ],
            optimizer: "rmsprop",
            learningRate: 0.001,
            l2: 0.0001,
            seed: 1234,
            speed: "max",
        });
        const hash = encodeConfig(original);
        assert.match(hash, /layers=12:relu:0\.1,3:gelu/);
        assert.deepEqual(decodeConfig(`#${hash}`), sanitizeConfig(original));
    });

    it("ignores garbage and falls back to defaults", () => {
        assert.deepEqual(decodeConfig(""), defaultConfig());
        const c = decodeConfig("#data=nope&noise=abc&layers=x:y,,4&opt=zzz&%E0%A4%A=1&features=q");
        assert.equal(c.dataset, "circle");
        assert.equal(c.noise, defaultConfig().noise);
        assert.deepEqual(c.layers, [{ units: 4, activation: "tanh", dropout: 0 }]);
        assert.equal(c.optimizer, "adam");
        assert.deepEqual(c.features, ["x", "y"]);
    });
});

describe("playground code generation", () => {
    it("emits the architecture with the public API", () => {
        const code = generateCode(
            config({
                dataset: "blobs",
                classes: 4,
                l2: 0.001,
                layers: [{ units: 5, activation: "swish", dropout: 0.2 }],
                optimizer: "sgd",
                learningRate: 0.1,
            }),
            { epochs: 50 },
        );
        assert.match(code, /import \{ Sequential, dense, dropout \} from "@zzza38\/orion-engine";/);
        assert.match(
            code,
            /dense\(5, \{ activation: "swish", kernelRegularizer: \{ l2: 0\.001 \} \}\),\n\s+dropout\(0\.2\),/,
        );
        assert.match(code, /dense\(4, \{ activation: "softmax"/);
        assert.match(code, /loss: "scce"/);
        assert.match(code, /optimizer: \{ name: "sgd", learningRate: 0\.1 \}/);
        assert.match(code, /epochs: 50, batchSize: 16/);
    });

    it("omits dropout from the import and avoids exponent notation", () => {
        const code = generateCode(config({ learningRate: 0.0001 }));
        assert.match(code, /import \{ Sequential, dense \} from/);
        assert.match(code, /learningRate: 0\.0001 /);
        assert.doesNotMatch(code, /e-\d/);
    });
});

describe("playground rendering helpers", () => {
    it("finds a contour crossing between grid samples", () => {
        // 2×2 field: left column below the level, right column above → one vertical segment at x = 1.
        const segments = contourSegments([0, 1, 0, 1], 2, 0.5);
        assert.deepEqual(segments, [1, 0.5, 1, 1.5]);
        assert.deepEqual(contourSegments([1, 1, 1, 1], 2, 0.5), []);
    });

    it("picks round tick values", () => {
        assert.deepEqual(niceTicks(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1]);
        assert.deepEqual(niceTicks(0, 0.7, 4), [0, 0.2, 0.4, 0.6]);
        assert.deepEqual(niceTicks(0, 1000, 4), [0, 500, 1000]);
    });
});
