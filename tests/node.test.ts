import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, describe, it } from "node:test";
import { dense, loadModel, loadModelSync, Sequential, saveModel, saveModelSync, VERSION } from "../src/node.js";

const dir = mkdtempSync(join(tmpdir(), "orion-node-test-"));
after(() => rmSync(dir, { recursive: true, force: true }));

function model(): Sequential {
    const m = new Sequential({ inputSize: 2, seed: 8, name: "saved", layers: [dense(4, "tanh"), dense(1, "sigmoid")] });
    m.compile({ loss: "bce", optimizer: { name: "adam", learningRate: 0.02 }, metrics: ["accuracy"] });
    m.fit(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        [0, 1, 1, 0],
        { epochs: 5, batchSize: 2 },
    );
    return m;
}

const SAMPLE = [0.25, 0.75];

describe("node entry point", () => {
    it("re-exports the main entry", () => {
        assert.equal(typeof Sequential, "function");
        assert.equal(VERSION, "0.1.0");
    });

    it("saves binary by default and JSON for .json paths, creating directories", async () => {
        const m = model();
        const binaryPath = join(dir, "nested", "deeper", "model.onn");
        const jsonPath = join(dir, "json", "model.onn.json");
        await saveModel(m, binaryPath);
        await saveModel(m, jsonPath);
        assert.deepEqual([...readFileSync(binaryPath).subarray(0, 6)], [0x89, 0x4f, 0x52, 0x49, 0x4f, 0x4e]);
        assert.equal(readFileSync(jsonPath, "utf8")[0], "{");

        const fromJson = await loadModel(jsonPath);
        assert.deepEqual(fromJson.predict(SAMPLE), m.predict(SAMPLE));
        assert.equal(fromJson.name, "saved");
        assert.equal(fromJson.compiled, true);
        const fromBinary = await loadModel(binaryPath);
        assert.ok(Math.abs(fromBinary.predict(SAMPLE)[0] - m.predict(SAMPLE)[0]) < 1e-6);
    });

    it("lets options.format override the extension and passes serializer options through", async () => {
        const m = model();
        const path = join(dir, "override.json");
        await saveModel(m, path, { format: "binary", precision: "float64" });
        assert.equal(readFileSync(path)[0], 0x89);
        assert.deepEqual((await loadModel(path)).predict(SAMPLE), m.predict(SAMPLE));
        const pretty = join(dir, "pretty.bin");
        await saveModel(m, pretty, { format: "json", pretty: true, metadata: { run: 1 } });
        assert.match(readFileSync(pretty, "utf8"), /"run": 1/);
    });

    it("has synchronous variants", () => {
        const m = model();
        const path = join(dir, "sync", "model.onn.json");
        saveModelSync(m, path);
        assert.deepEqual(loadModelSync(path, { seed: 3 }).predict(SAMPLE), m.predict(SAMPLE));
        assert.equal(loadModelSync(path, { seed: 3 }).seed, 3);
    });

    it("loads legacy .onn text files", async () => {
        const path = join(dir, "legacy.onn");
        writeFileSync(path, "2:relu:2:swish\n0.71:-0.2:0.19|-1.82:0.95:0.97");
        const legacy = await loadModel(path);
        assert.ok(Math.abs(legacy.predict([1, 0])[0] - 0.6398545523625034) < 1e-15);
    });

    it("reports bad paths and missing files", async () => {
        await assert.rejects(saveModel(model(), ""), /path must be a non-empty string/);
        await assert.rejects(loadModel(join(dir, "missing.onn")), { code: "ENOENT" });
        assert.throws(() => loadModelSync(join(dir, "missing.onn")), { code: "ENOENT" });
    });
});
