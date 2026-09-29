import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";
import * as orion from "../src/index.js";

describe("VERSION", () => {
    it("matches package.json", () => {
        const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
            version: string;
        };
        assert.equal(orion.VERSION, pkg.version);
    });
});

describe("public API", () => {
    it("exposes the main building blocks from the package entry", () => {
        for (const name of [
            "Sequential",
            "dense",
            "dropout",
            "batchNormalization",
            "activation",
            "layerFromConfig",
            "earlyStopping",
            "History",
            "oneHot",
            "StandardScaler",
            "serializeModel",
            "deserializeModel",
            "encodeArtifact",
            "decodeArtifact",
            "detectFormat",
            "validateArtifact",
            "getActivation",
            "ACTIVATION_NAMES",
            "Adam",
            "cosineDecay",
            "Matrix",
            "Random",
            "TrainingError",
        ]) {
            assert.ok(name in orion, `missing export ${name}`);
        }
        assert.equal("saveModel" in orion, false, "file helpers live in the /node entry");
    });
});
