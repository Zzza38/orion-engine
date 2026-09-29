/**
 * Saving and loading models: files in Node, bytes/strings anywhere, and legacy 0.0.x imports.
 *
 * Demonstrates `saveModel` / `loadModel` from the Node entry point (binary float32, binary
 * float64 and JSON), the browser-safe `serializeModel` / `deserializeModel`, that reloaded models
 * come back compiled and can keep training, and importing a model written by orion-engine 0.0.x.
 *
 *   npx tsx examples/save-and-load.ts
 *
 * Expected output: the three file sizes (binary float32 ≈ 1.2 KB, binary float64 ≈ 1.3 KB,
 * pretty-printed JSON ≈ 2.1 KB; for a model this small the header dominates), "identical" for the
 * float64 and JSON copies and a tiny maximum difference (well below 1e-6) for the float32 one,
 * the reloaded model's name and compiled state with its loss dropping after 50 more epochs, the
 * in-memory round trip, and the legacy XOR model's predictions (0.000, 1.000, 1.000, 0.000)
 * before and after upgrading the file to the binary format.
 */
import { mkdtempSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { dense, deserializeModel, loadModel, Sequential, saveModel, serializeModel } from "../src/node.js";

const x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];
const y = [0, 1, 1, 0];

// Train a small model to save.
const model = new Sequential({ inputSize: 2, seed: 42, name: "xor", layers: [dense(8, "tanh"), dense(1, "sigmoid")] });
model.compile({ loss: "bce", optimizer: { name: "adam", learningRate: 0.05 }, metrics: ["accuracy"] });
model.fit(x, y, { epochs: 300, batchSize: 4 });
const expected = model.predict(x);

const dir = mkdtempSync(join(tmpdir(), "orion-save-and-load-"));
try {
    // --- Files (Node): the format follows the extension unless `format` is given -------------
    const files = {
        "binary, float32 (default)": join(dir, "xor.onn"),
        "binary, float64": join(dir, "xor-f64.onn"),
        "JSON (pretty)": join(dir, "xor.onn.json"),
    };
    await saveModel(model, files["binary, float32 (default)"], { metadata: { task: "xor" } });
    await saveModel(model, files["binary, float64"], { precision: "float64" });
    await saveModel(model, files["JSON (pretty)"], { pretty: true });

    console.log("Saved:");
    for (const [label, path] of Object.entries(files)) {
        console.log(`  ${label.padEnd(26)} ${String(statSync(path).size).padStart(5)} bytes  ${path}`);
    }

    // loadModel detects the format from the content, not the file name.
    console.log("\nReloaded predictions vs the original:");
    for (const [label, path] of Object.entries(files)) {
        const restored = await loadModel(path);
        const got = restored.predict(x);
        const maxDiff = Math.max(...got.map((row, i) => Math.abs(row[0] - expected[i][0])));
        console.log(
            `  ${label.padEnd(26)} ${maxDiff === 0 ? "identical" : `max difference ${maxDiff.toExponential(2)}`}`,
        );
    }

    // Models saved after compile() come back compiled (with a fresh optimizer), ready to train more.
    const restored = await loadModel(files["binary, float64"]);
    const before = restored.evaluate(x, y).loss;
    restored.fit(x, y, { epochs: 50, batchSize: 4 });
    console.log(
        `\nReloaded model: name "${restored.name}", compiled: ${restored.compiled}, ` +
            `loss ${before.toFixed(5)} → ${restored.evaluate(x, y).loss.toFixed(5)} after 50 more epochs`,
    );

    // --- Bytes and strings (browser-safe, from the main entry) -----------------------------------
    const bytes = serializeModel(model); // Uint8Array: send it over the network, store it in IndexedDB, …
    const json = serializeModel(model, { format: "json" }); // string: localStorage, a JSON API, …
    const fromBytes = deserializeModel(bytes);
    const fromJson = deserializeModel(json);
    console.log(
        `\nIn memory: ${bytes.byteLength} bytes / ${json.length} characters; ` +
            `predict([1, 0]) = ${fromBytes.predict([1, 0])[0].toFixed(6)} / ${fromJson.predict([1, 0])[0].toFixed(6)}`,
    );

    // --- Legacy 0.0.x models ---------------------------------------------------------------------
    // orion-engine 0.0.x wrote two lines of text: "size:activation" pairs (the first is the input
    // layer), then each neuron's weights and bias. This one is a 2-3-1 XOR network.
    const legacyOnn =
        "2:linear:3:tanh:1:sigmoid\n-2.212:-2.238:-0.051|3.532:-4.58:-1.408|-4.545:3.465:-1.361|-4.654:8.302:8.303:3.817";
    const legacy = deserializeModel(legacyOnn); // or: await loadModel("old-model.onn")
    const show = (m: Sequential) =>
        m
            .predict(x)
            .map((row) => row[0].toFixed(3))
            .join(", ");
    console.log(
        `\nLegacy model (${legacy.layers.length} dense layers, input size ${legacy.inputSize}): ${show(legacy)}`,
    );

    // Upgrade it: write the old text file, load it, and save it back in the current binary format.
    const legacyPath = join(dir, "legacy-xor.onn");
    writeFileSync(legacyPath, legacyOnn);
    await saveModel(await loadModel(legacyPath), legacyPath, { precision: "float64" });
    console.log(`Upgraded to binary (${statSync(legacyPath).size} bytes): ${show(await loadModel(legacyPath))}`);
} finally {
    rmSync(dir, { recursive: true, force: true });
}
