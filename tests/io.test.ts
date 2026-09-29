import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { SerializationError, ValidationError } from "../src/core/errors.js";
import type { LayerConfig, ModelArtifact, WeightEntry } from "../src/core/types.js";
import {
    crc32,
    decodeArtifact,
    decodeBinary,
    decodeJson,
    decodeLegacyOnn,
    detectFormat,
    encodeArtifact,
    encodeBinary,
    encodeJson,
    validateArtifact,
} from "../src/io/index.js";

// ---------------------------------------------------------------------------------------------
// Fixtures & helpers
// ---------------------------------------------------------------------------------------------

/** Values that break naive float formatting: repeating decimals, extremes, subnormals, -0. */
const TRICKY = [
    0.1, 1 / 3, Math.PI, -Math.E, 0.30000000000000004, 123456789.12345679, -2.5e-8, 1e21,
    1e-300, 5e-324, -1.7976931348623157e308, -0,
];

/** Values that fit float32 (no overflow), for float32 tests. */
const MODERATE = [0.1, 1 / 3, Math.PI, -Math.E, -2.5e-8, 12345.678, 1e-3, -0.75, 42, -1e10, 7e-5, 0];

function dense(name: string, units: number, activation: string): LayerConfig {
    return {
        type: "dense",
        name,
        units,
        activation: { name: activation },
        useBias: true,
        kernelInitializer: { name: "glorotUniform" },
        biasInitializer: { name: "zeros" },
    };
}

function sampleArtifact(values: readonly number[] = MODERATE): ModelArtifact {
    const pick = (n: number, shift: number): number[] =>
        Array.from({ length: n }, (_, i) => values[(i + shift) % values.length]);
    return {
        format: "orion-engine",
        formatVersion: 1,
        inputSize: 3,
        layers: [dense("dense_1", 4, "relu"), dense("dense_2", 2, "softmax")],
        weights: [
            { name: "dense_1/kernel", shape: [3, 4], data: Float64Array.from(pick(12, 0)) },
            { name: "dense_1/bias", shape: [1, 4], data: pick(4, 5) },
            { name: "dense_2/kernel", shape: [4, 2], data: Float64Array.from(pick(8, 3)) },
            { name: "dense_2/bias", shape: [1, 2], data: pick(2, 7) },
        ],
        training: {
            loss: { name: "categoricalCrossentropy" },
            optimizer: { name: "adam", learningRate: 0.001, clipNorm: 1 },
            metrics: ["accuracy"],
        },
        metadata: { author: "tests", tags: ["a", "b"], nested: { ok: true, nothing: null }, "odd key": 1 },
    };
}

/** Compares everything but weight data structurally, and weight data element by element. */
function assertArtifactsEqual(
    actual: ModelArtifact,
    expected: ModelArtifact,
    compare: (a: number, e: number, where: string) => void = assertSameNumber,
): void {
    assert.equal(actual.format, expected.format);
    assert.equal(actual.formatVersion, expected.formatVersion);
    assert.equal(actual.inputSize, expected.inputSize);
    assert.deepEqual(actual.layers, expected.layers);
    assert.deepEqual(actual.training, expected.training);
    assert.deepEqual(actual.metadata, expected.metadata);
    assert.equal(actual.weights.length, expected.weights.length);
    actual.weights.forEach((w, i) => {
        const e = expected.weights[i];
        assert.equal(w.name, e.name);
        assert.deepEqual(w.shape, e.shape);
        assert.ok(w.data instanceof Float64Array, `${w.name}: decoded data should be a Float64Array`);
        assert.equal(w.data.length, e.data.length);
        for (let k = 0; k < e.data.length; k++) compare(w.data[k], e.data[k], `${w.name}[${k}]`);
    });
}

function assertSameNumber(actual: number, expected: number, where: string): void {
    assert.ok(Object.is(actual, expected), `${where}: expected ${expected}, got ${actual}`);
}

function assertClose32(actual: number, expected: number, where: string): void {
    const tolerance = 1e-6 * Math.abs(expected);
    assert.ok(Math.abs(actual - expected) <= tolerance, `${where}: expected ~${expected}, got ${actual}`);
    assert.equal(actual, Math.fround(expected), `${where}: should be exactly the float32 rounding`);
}

function throwsSerialization(fn: () => unknown, message: RegExp): void {
    assert.throws(fn, (error: unknown) => {
        assert.ok(error instanceof SerializationError, `expected SerializationError, got ${String(error)}`);
        assert.match(error.message, message);
        return true;
    });
}

const utf8 = (text: string): Uint8Array => new TextEncoder().encode(text);

/**
 * Independent, spec-driven container writer used to feed decodeBinary hand-crafted (but correctly
 * checksummed) headers.
 */
function buildContainer(header: unknown, data: Uint8Array = new Uint8Array(0)): Uint8Array {
    const headerBytes = utf8(JSON.stringify(header));
    const dataStart = Math.ceil((16 + headerBytes.length) / 8) * 8;
    const out = new Uint8Array(dataStart + data.length + 4);
    out.set([0x89, 0x4f, 0x52, 0x49, 0x4f, 0x4e, 0x0d, 0x0a]);
    const view = new DataView(out.buffer);
    view.setUint16(8, 1, true);
    view.setUint32(12, headerBytes.length, true);
    out.set(headerBytes, 16);
    out.set(data, dataStart);
    view.setUint32(out.length - 4, crc32(out.subarray(0, out.length - 4)), true);
    return out;
}

function float64Bytes(values: number[]): Uint8Array {
    const out = new Uint8Array(values.length * 8);
    const view = new DataView(out.buffer);
    values.forEach((v, i) => view.setFloat64(i * 8, v, true));
    return out;
}

const DOC_EXAMPLE = "2:relu:2:swish\n0.71:-0.2:0.19|-1.82:0.95:0.97";

// ---------------------------------------------------------------------------------------------
// validateArtifact
// ---------------------------------------------------------------------------------------------

describe("validateArtifact", () => {
    it("accepts a valid artifact and returns the same reference", () => {
        const artifact = sampleArtifact();
        assert.equal(validateArtifact(artifact), artifact);
    });

    it("accepts empty layers/weights and omitted training/metadata", () => {
        const artifact: ModelArtifact = { format: "orion-engine", formatVersion: 1, inputSize: 1, layers: [], weights: [] };
        assert.equal(validateArtifact(artifact), artifact);
    });

    it("tolerates undefined-valued layer config properties", () => {
        const artifact = sampleArtifact();
        artifact.layers[0].extra = undefined;
        assert.doesNotThrow(() => validateArtifact(artifact));
    });

    const cases: [string, (a: Record<string, unknown> & ModelArtifact) => unknown, RegExp][] = [
        ["null", () => null, /artifact: expected an object, got null/],
        ["an array", () => [], /artifact: expected an object/],
        ["a string", () => "model", /artifact: expected an object/],
        ["a wrong format marker", (a) => ({ ...a, format: "onnx" }), /format: expected "orion-engine", got "onnx"/],
        ["a missing format marker", (a) => ({ ...a, format: undefined }), /format: expected "orion-engine", got undefined/],
        ["a newer formatVersion", (a) => ({ ...a, formatVersion: 2 }), /formatVersion 2 is newer.*upgrade/],
        ["a string formatVersion", (a) => ({ ...a, formatVersion: "1" }), /formatVersion: expected 1, got "1"/],
        ["inputSize 0", (a) => ({ ...a, inputSize: 0 }), /inputSize: expected a positive integer, got 0/],
        ["fractional inputSize", (a) => ({ ...a, inputSize: 2.5 }), /inputSize: expected a positive integer/],
        ["string inputSize", (a) => ({ ...a, inputSize: "3" }), /inputSize: expected a positive integer, got "3"/],
        ["non-array layers", (a) => ({ ...a, layers: {} }), /layers: expected an array of layer configs, got an object/],
        ["a null layer", (a) => ({ ...a, layers: [null] }), /layers\[0\]: expected a layer config object, got null/],
        ["a layer without type", (a) => { delete (a.layers[0] as Partial<LayerConfig>).type; return a; }, /layers\[0\]\.type: expected a non-empty string/],
        ["a numeric layer name", (a) => { a.layers[1].name = 5 as never; return a; }, /layers\[1\]\.name: expected a non-empty string, got 5/],
        ["duplicate layer names", (a) => { a.layers[1].name = "dense_1"; return a; }, /layers\[1\]\.name: duplicate layer name "dense_1"/],
        ["NaN in a layer config", (a) => { a.layers[0].units = NaN; return a; }, /layers\[0\]\.units: expected a finite number.*got NaN/],
        ["a function in a layer config", (a) => { a.layers[0].fn = (() => 1) as never; return a; }, /layers\[0\]\.fn: expected a JSON value.*got a function/],
        ["a Date in a layer config", (a) => { a.layers[0].when = new Date() as never; return a; }, /layers\[0\]\.when: expected a JSON value.*got a Date/],
        ["non-array weights", (a) => ({ ...a, weights: "none" }), /weights: expected an array of weight entries/],
        ["an empty weight name", (a) => { a.weights[0].name = ""; return a; }, /weights\[0\]\.name: expected a non-empty string, got ""/],
        ["duplicate weight names", (a) => { a.weights[2].name = "dense_1/kernel"; return a; }, /weights\[2\]\.name: duplicate weight name "dense_1\/kernel"/],
        ["a 1-D shape", (a) => { a.weights[0].shape = [12] as never; return a; }, /weights\[0\] \("dense_1\/kernel"\)\.shape: expected \[rows, cols\] of positive integers, got \[12\]/],
        ["a zero dimension", (a) => { a.weights[1].shape = [0, 4]; return a; }, /weights\[1\] \("dense_1\/bias"\)\.shape: .*got \[0, 4\]/],
        ["a fractional dimension", (a) => { a.weights[1].shape = [0.5, 8]; return a; }, /\.shape: expected \[rows, cols\] of positive integers/],
        ["a data length mismatch", (a) => { a.weights[0].shape = [4, 4]; return a; }, /dense_1\/kernel"\)\.data: expected 16 values for shape \[4, 4\], got 12/],
        ["string data", (a) => { a.weights[1].data = "1,2,3,4" as never; return a; }, /\.data: expected a number\[\], Float64Array or Float32Array, got "1,2,3,4"/],
        ["Int32Array data", (a) => { a.weights[1].data = new Int32Array(4) as never; return a; }, /got a Int32Array of length 4/],
        ["NaN in number[] data", (a) => { a.weights[1].data = [1, 2, NaN, 4]; return a; }, /dense_1\/bias"\)\.data\[2\]: expected a finite number, got NaN/],
        ["Infinity in Float64Array data", (a) => { (a.weights[0].data as Float64Array)[5] = Infinity; return a; }, /data\[5\]: expected a finite number, got Infinity/],
        ["a string in number[] data", (a) => { a.weights[3].data = [1, "2"] as never; return a; }, /data\[1\]: expected a finite number, got "2"/],
        ["a sparse data array", (a) => { a.weights[3].data = new Array<number>(2); return a; }, /data\[0\]: expected a finite number, got undefined/],
        ["training without loss", (a) => { delete (a.training as Partial<typeof a.training>)!.loss; return a; }, /training\.loss: expected a loss config object, got undefined/],
        ["an unnamed optimizer", (a) => { a.training!.optimizer = {} as never; return a; }, /training\.optimizer\.name: expected a non-empty string/],
        ["non-array metrics", (a) => { a.training!.metrics = "accuracy" as never; return a; }, /training\.metrics: expected an array of metric names/],
        ["a non-string metric", (a) => { a.training!.metrics = [1] as never; return a; }, /training\.metrics\[0\]: expected a metric name string/],
        ["array metadata", (a) => ({ ...a, metadata: [] }), /metadata: expected a plain object/],
        ["Infinity in metadata", (a) => { a.metadata!.x = Infinity; return a; }, /metadata\.x: expected a finite number/],
        ["an odd metadata key", (a) => { a.metadata!["odd key"] = NaN; return a; }, /metadata\["odd key"\]: expected a finite number/],
        ["circular metadata", (a) => { const m = a.metadata as Record<string, unknown>; m.self = m; return a; }, /metadata\.self: circular reference/],
    ];

    for (const [label, mutate, message] of cases) {
        it(`rejects ${label}`, () => {
            const bad = mutate(sampleArtifact() as Record<string, unknown> & ModelArtifact);
            throwsSerialization(() => validateArtifact(bad), message);
        });
    }
});

// ---------------------------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------------------------

describe("JSON format", () => {
    it("round-trips every float64 exactly, including -0, subnormals and extremes", () => {
        const artifact = sampleArtifact(TRICKY);
        artifact.weights[0].data = Float64Array.from(TRICKY);
        artifact.weights[0].shape = [3, 4];
        const decoded = decodeJson(encodeJson(artifact));
        assertArtifactsEqual(decoded, artifact);
        assert.ok(Object.is(decoded.weights[0].data[11], -0), "negative zero must keep its sign");
    });

    it("round-trips pretty output identically to compact output", () => {
        const artifact = sampleArtifact(TRICKY);
        const compact = encodeJson(artifact);
        const pretty = encodeJson(artifact, { pretty: true });
        assert.ok(!compact.includes("\n"), "compact output is a single line");
        assert.ok(pretty.split("\n").length > 20, "pretty output spans many lines");
        assertArtifactsEqual(decodeJson(pretty), artifact);
        assert.deepEqual(JSON.parse(pretty), JSON.parse(compact));
    });

    it("writes pretty kernels one matrix row per line and biases on one line", () => {
        const artifact = sampleArtifact();
        artifact.weights[0].data = Float64Array.from({ length: 12 }, (_, i) => i + 0.5);
        artifact.weights[1].data = [1, 2, 3, 4];
        const pretty = encodeJson(artifact, { pretty: true });
        assert.match(pretty, /\n {8}0\.5, 1\.5, 2\.5, 3\.5,\n {8}4\.5, 5\.5, 6\.5, 7\.5,\n {8}8\.5, 9\.5, 10\.5, 11\.5\n {6}\]/);
        assert.match(pretty, /"data": \[1, 2, 3, 4\]/);
    });

    it("uses a fixed key order with weights last", () => {
        const text = encodeJson(sampleArtifact());
        assert.ok(text.startsWith('{"format":"orion-engine","formatVersion":1,"inputSize":3,"layers":['));
        const keys = Object.keys(JSON.parse(text) as object);
        assert.deepEqual(keys, ["format", "formatVersion", "inputSize", "layers", "training", "metadata", "weights"]);
    });

    it("widens Float32Array data exactly", () => {
        const artifact = sampleArtifact();
        const f32 = new Float32Array([0.1, -1 / 3, 3.4e38, 1e-45]);
        artifact.weights[1].data = f32;
        const text = encodeJson(artifact);
        assert.match(text, /0\.10000000149011612/);
        const decoded = decodeJson(text);
        assert.deepEqual(Array.from(decoded.weights[1].data), Array.from(f32));
    });

    it("omits undefined training/metadata and drops unknown top-level keys", () => {
        const artifact: ModelArtifact = { format: "orion-engine", formatVersion: 1, inputSize: 2, layers: [], weights: [] };
        const text = encodeJson({ ...artifact, extra: 1 } as ModelArtifact);
        assert.equal(text, '{"format":"orion-engine","formatVersion":1,"inputSize":2,"layers":[],"weights":[]}');
        assert.equal(encodeJson(artifact, { pretty: true }).split("\n").at(-2), '  "weights": []');
        const decoded = decodeJson(text.replace("{", '{"future":true,'));
        assert.deepEqual(Object.keys(decoded), ["format", "formatVersion", "inputSize", "layers", "weights"]);
    });

    it("ignores a leading byte-order mark", () => {
        const artifact = sampleArtifact();
        assertArtifactsEqual(decodeJson(`﻿${encodeJson(artifact)}`), artifact);
    });

    it("rejects malformed JSON with SerializationError", () => {
        throwsSerialization(() => decodeJson(""), /Invalid JSON model: .*JSON/);
        throwsSerialization(() => decodeJson('{"format":'), /Invalid JSON model/);
        throwsSerialization(() => decodeJson("[]"), /artifact: expected an object, got \[\]/);
        throwsSerialization(() => decodeJson("null"), /artifact: expected an object, got null/);
        throwsSerialization(() => decodeJson('{"format":"orion-engine","formatVersion":1}'), /inputSize/);
        throwsSerialization(() => decodeJson(42 as never), /Invalid JSON model: expected a string, got 42/);
        const text = encodeJson(sampleArtifact()).replace('"shape":[1,2]', '"shape":[1,3]');
        throwsSerialization(() => decodeJson(text), /dense_2\/bias"\)\.data: expected 3 values for shape \[1, 3\], got 2/);
    });

    it("refuses to encode an invalid artifact", () => {
        const artifact = sampleArtifact();
        (artifact.weights[0].data as Float64Array)[0] = NaN;
        throwsSerialization(() => encodeJson(artifact), /dense_1\/kernel"\)\.data\[0\]: expected a finite number, got NaN/);
    });
});

// ---------------------------------------------------------------------------------------------
// Binary
// ---------------------------------------------------------------------------------------------

describe("binary format", () => {
    it("implements standard CRC-32", () => {
        assert.equal(crc32(utf8("123456789")), 0xcbf43926);
        assert.equal(crc32(new Uint8Array(0)), 0);
        assert.equal(crc32(utf8("The quick brown fox jumps over the lazy dog")), 0x414fa339);
    });

    it("writes the documented layout", () => {
        const bytes = encodeBinary(sampleArtifact(), { precision: "float64" });
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        assert.deepEqual(Array.from(bytes.subarray(0, 8)), [0x89, 0x4f, 0x52, 0x49, 0x4f, 0x4e, 0x0d, 0x0a]);
        assert.equal(view.getUint16(8, true), 1, "container version");
        assert.equal(view.getUint16(10, true), 0, "flags");
        const headerLength = view.getUint32(12, true);
        const header = JSON.parse(new TextDecoder().decode(bytes.subarray(16, 16 + headerLength))) as {
            weights: { name: string; dtype: string; byteOffset: number; length: number }[];
            dataByteLength: number;
        };
        const dataStart = Math.ceil((16 + headerLength) / 8) * 8;
        for (let i = 16 + headerLength; i < dataStart; i++) assert.equal(bytes[i], 0, "header padding is zero");
        assert.equal(bytes.length, dataStart + header.dataByteLength + 4);
        assert.deepEqual(header.weights.map((w) => [w.name, w.dtype, w.byteOffset, w.length]), [
            ["dense_1/kernel", "float64", 0, 12],
            ["dense_1/bias", "float64", 96, 4],
            ["dense_2/kernel", "float64", 128, 8],
            ["dense_2/bias", "float64", 192, 2],
        ]);
        assert.equal(view.getFloat64(dataStart, true), MODERATE[0]);
        assert.equal(view.getUint32(bytes.length - 4, true), crc32(bytes.subarray(0, bytes.length - 4)));
    });

    it("aligns every float32 tensor to 8 bytes", () => {
        const artifact = sampleArtifact();
        artifact.weights[1] = { name: "dense_1/bias", shape: [1, 3], data: [1, 2, 3] };
        artifact.weights[0] = { name: "dense_1/kernel", shape: [1, 1], data: [1] };
        const bytes = encodeBinary(artifact);
        const view = new DataView(bytes.buffer);
        const header = JSON.parse(new TextDecoder().decode(bytes.subarray(16, 16 + view.getUint32(12, true)))) as {
            weights: { byteOffset: number }[];
            dataByteLength: number;
        };
        assert.deepEqual(header.weights.map((w) => w.byteOffset), [0, 8, 24, 56]);
        assert.equal(header.dataByteLength, 64);
    });

    it("round-trips float64 exactly, including -0, subnormals and extremes", () => {
        const artifact = sampleArtifact(TRICKY);
        const bytes = encodeBinary(artifact, { precision: "float64" });
        assertArtifactsEqual(decodeBinary(bytes), artifact);
    });

    it("round-trips float32 (the default) within 1e-6 relative error at half the data size", () => {
        const artifact = sampleArtifact();
        artifact.weights[2].data = new Float32Array(artifact.weights[2].data);
        const f32 = encodeBinary(artifact);
        const f64 = encodeBinary(artifact, { precision: "float64" });
        assertArtifactsEqual(decodeBinary(f32), artifact, assertClose32);
        assert.ok(f32.length < f64.length);
        const headerText = new TextDecoder().decode(f32.subarray(16, 16 + new DataView(f32.buffer).getUint32(12, true)));
        assert.equal(headerText.match(/"dtype":"float32"/g)?.length, 4);
    });

    it("round-trips an artifact with no layers or weights", () => {
        const artifact: ModelArtifact = { format: "orion-engine", formatVersion: 1, inputSize: 7, layers: [], weights: [] };
        const decoded = decodeBinary(encodeBinary(artifact));
        assert.deepEqual(decoded, artifact);
    });

    it("decodes a Uint8Array view at a non-zero, unaligned byteOffset", () => {
        const artifact = sampleArtifact(TRICKY);
        const bytes = encodeBinary(artifact, { precision: "float64" });
        const backing = new Uint8Array(bytes.length + 64).fill(0xaa);
        backing.set(bytes, 3);
        const view = backing.subarray(3, 3 + bytes.length);
        assert.equal(view.byteOffset, 3);
        assertArtifactsEqual(decodeBinary(view), artifact);
    });

    it("decodes an ArrayBuffer and a pooled Node Buffer", () => {
        const artifact = sampleArtifact();
        const bytes = encodeBinary(artifact, { precision: "float64" });
        const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
        assertArtifactsEqual(decodeBinary(buffer), artifact);
        assertArtifactsEqual(decodeBinary(Buffer.from(bytes)), artifact);
    });

    it("detects a corrupted byte with the CRC", () => {
        const bytes = encodeBinary(sampleArtifact());
        const corrupted = bytes.slice();
        corrupted[corrupted.length - 20] ^= 0x01; // inside the weight data
        throwsSerialization(() => decodeBinary(corrupted), /CRC-32 mismatch \(stored 0x[0-9A-F]{8}, computed 0x[0-9A-F]{8}\)/);

        const header = bytes.slice();
        const at = new TextDecoder().decode(header).indexOf("dense_2");
        header[at + 6] = 0x33; // "dense_2" -> "dense_3", still valid JSON
        throwsSerialization(() => decodeBinary(header), /CRC-32 mismatch/);
    });

    it("rejects every single-bit flip anywhere in the file", () => {
        const bytes = encodeBinary(sampleArtifact());
        for (let i = 0; i < bytes.length; i++) {
            const corrupted = bytes.slice();
            corrupted[i] ^= 0x10;
            assert.throws(() => decodeBinary(corrupted), SerializationError, `flip at byte ${i} was not detected`);
        }
    });

    it("reports truncation", () => {
        const bytes = encodeBinary(sampleArtifact());
        throwsSerialization(() => decodeBinary(bytes.subarray(0, bytes.length - 10)), /truncated: expected \d+ bytes, got \d+/);
        throwsSerialization(() => decodeBinary(bytes.subarray(0, 40)), /truncated: header declares \d+ bytes/);
        throwsSerialization(() => decodeBinary(bytes.subarray(0, 12)), /truncated: 12 bytes/);
        throwsSerialization(() => decodeBinary(bytes.subarray(0, 4)), /truncated: 4 bytes/);
        for (let n = 0; n < bytes.length; n++) {
            assert.throws(() => decodeBinary(bytes.subarray(0, n)), SerializationError, `truncation to ${n} bytes`);
        }
    });

    it("reports trailing bytes", () => {
        const bytes = encodeBinary(sampleArtifact());
        const longer = new Uint8Array(bytes.length + 5);
        longer.set(bytes);
        throwsSerialization(() => decodeBinary(longer), /5 unexpected trailing bytes/);
    });

    it("rejects the wrong magic", () => {
        const bytes = encodeBinary(sampleArtifact());
        const wrong = bytes.slice();
        wrong[0] = 0x50;
        wrong[1] = 0x4b;
        throwsSerialization(() => decodeBinary(wrong), /not an Orion Engine binary model: expected magic bytes 89 4F 52 49 4F 4E 0D 0A, found 50 4B/);
        throwsSerialization(() => decodeBinary(utf8(encodeJson(sampleArtifact()))), /expected magic bytes/);
        throwsSerialization(() => decodeBinary(new Uint8Array(0)), /Invalid binary model: file is empty/);
    });

    it("explains text-mode line-ending corruption", () => {
        const bytes = encodeBinary(sampleArtifact());
        const crlfToLf = new Uint8Array(bytes.length - 1);
        crlfToLf.set(bytes.subarray(0, 6));
        crlfToLf.set(bytes.subarray(7), 6);
        throwsSerialization(() => decodeBinary(crlfToLf), /damaged signature .*text mode/);
        const highBitStripped = bytes.slice();
        highBitStripped[0] = 0x09;
        throwsSerialization(() => decodeBinary(highBitStripped), /damaged signature/);
    });

    it("rejects newer container versions and non-zero flags", () => {
        const bytes = encodeBinary(sampleArtifact());
        const newer = bytes.slice();
        newer[8] = 2;
        throwsSerialization(() => decodeBinary(newer), /container version 2 is newer.*upgrade/);
        const flagged = bytes.slice();
        flagged[10] = 1;
        throwsSerialization(() => decodeBinary(flagged), /reserved flags field must be 0, got 0x0001/);
    });

    it("validates checksummed but inconsistent headers", () => {
        const base = { format: "orion-engine", formatVersion: 1, inputSize: 1, layers: [dense("dense_1", 2, "relu")] };
        const data = float64Bytes([1, 2]);
        const weight = { name: "dense_1/kernel", shape: [1, 2], dtype: "float64", byteOffset: 0, length: 2 };
        const ok = buildContainer({ ...base, weights: [weight], dataByteLength: 16 }, data);
        assert.deepEqual(Array.from(decodeBinary(ok).weights[0].data), [1, 2]);

        const bad = (w: object, message: RegExp, extra: object = {}) =>
            throwsSerialization(() => decodeBinary(buildContainer({ ...base, weights: [w], dataByteLength: 16, ...extra }, data)), message);
        bad({ ...weight, dtype: "int8" }, /header\.weights\[0\]\.dtype: expected "float32" or "float64", got "int8"/);
        bad({ ...weight, length: 3 }, /header\.weights\[0\]\.length: expected 2 \(shape \[1, 2\]\), got 3/);
        bad({ ...weight, byteOffset: 8 }, /header\.weights\[0\]: bytes \[8, 24\) lie outside the 16-byte data section/);
        bad({ ...weight, byteOffset: -8 }, /byteOffset: expected a non-negative integer, got -8/);
        bad({ ...weight, shape: [2] }, /header\.weights\[0\]\.shape: expected \[rows, cols\]/);
        bad(weight, /formatVersion 9 is newer/, { formatVersion: 9 });
        bad(weight, /layers\[0\]\.name: expected a non-empty string/, { layers: [{ type: "dense" }] });
        bad({ ...weight, name: 7 }, /weights\[0\]\.name: expected a non-empty string, got 7/);
        throwsSerialization(
            () => decodeBinary(buildContainer({ ...base, weights: [weight] }, data)),
            /header\.dataByteLength: expected a non-negative integer, got undefined/,
        );
        throwsSerialization(() => decodeBinary(buildContainer([1, 2])), /header: expected a JSON object/);
        throwsSerialization(
            () => decodeBinary(buildContainer({ ...base, weights: [weight], dataByteLength: 16 }, float64Bytes([1, NaN]))),
            /data\[1\]: expected a finite number, got NaN/,
        );
    });

    it("rejects values that overflow float32 and invalid precision options", () => {
        const artifact = sampleArtifact();
        artifact.weights[3].data = [1, 1e39];
        throwsSerialization(() => encodeBinary(artifact), /weight "dense_2\/bias" as float32: value 1e\+39 at index 1.*float64/);
        assert.doesNotThrow(() => encodeBinary(artifact, { precision: "float64" }));
        assert.throws(() => encodeBinary(sampleArtifact(), { precision: "float16" as never }), ValidationError);
    });
});

// ---------------------------------------------------------------------------------------------
// Legacy .onn
// ---------------------------------------------------------------------------------------------

describe("legacy .onn import", () => {
    it("converts the documented example", () => {
        const artifact = decodeLegacyOnn(DOC_EXAMPLE);
        assert.equal(artifact.inputSize, 2);
        assert.deepEqual(artifact.layers, [
            {
                type: "dense",
                name: "dense_1",
                units: 2,
                activation: { name: "swish" },
                useBias: true,
                kernelInitializer: { name: "glorotUniform" },
                biasInitializer: { name: "zeros" },
            },
        ]);
        assert.deepEqual(artifact.weights.map((w) => [w.name, w.shape]), [["dense_1/kernel", [2, 2]], ["dense_1/bias", [1, 2]]]);
        // kernel [[0.71, -1.82], [-0.2, 0.95]] row-major; bias [[0.19, 0.97]].
        assert.deepEqual(Array.from(artifact.weights[0].data), [0.71, -1.82, -0.2, 0.95]);
        assert.deepEqual(Array.from(artifact.weights[1].data), [0.19, 0.97]);
        assert.ok(artifact.weights[0].data instanceof Float64Array);
        assert.deepEqual(artifact.metadata, { convertedFrom: "legacy-onn" });
        assert.equal(artifact.training, undefined);
    });

    it("converts a 3-layer model that computes what the legacy engine computed", () => {
        const text = "3:relu:2:tanh:2:leakyRelu:1:sigmoid\n" +
            "0.1:0.2:0.3:0.4|0.5:0.6:0.7:0.8|-1:-2:3|4:5:-6|1.5:-2.5:0.25";
        const artifact = decodeLegacyOnn(text);
        assert.equal(artifact.inputSize, 3);
        assert.deepEqual(artifact.layers.map((l) => [l.name, l.units, l.activation]), [
            ["dense_1", 2, { name: "tanh" }],
            ["dense_2", 2, { name: "leakyRelu", alpha: 0.01 }],
            ["dense_3", 1, { name: "sigmoid" }],
        ]);
        const data = Object.fromEntries(artifact.weights.map((w: WeightEntry) => [w.name, [w.shape, Array.from(w.data)]]));
        assert.deepEqual(data, {
            "dense_1/kernel": [[3, 2], [0.1, 0.5, 0.2, 0.6, 0.3, 0.7]],
            "dense_1/bias": [[1, 2], [0.4, 0.8]],
            "dense_2/kernel": [[2, 2], [-1, 4, -2, 5]],
            "dense_2/bias": [[1, 2], [3, -6]],
            "dense_3/kernel": [[2, 1], [1.5, -2.5]],
            "dense_3/bias": [[1, 1], [0.25]],
        });

        // Legacy semantics, straight from the neuron list: out_j = f(sum_i in_i * w_ji + b_j).
        const neurons = [
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            [[-1, -2, 3], [4, 5, -6]],
            [[1.5, -2.5, 0.25]],
        ];
        const fns = [Math.tanh, (x: number) => (x > 0 ? x : 0.01 * x), (x: number) => 1 / (1 + Math.exp(-x))];
        const input = [0.3, -1.2, 2];
        let legacy = input;
        neurons.forEach((layer, l) => {
            legacy = layer.map((n) => fns[l](legacy.reduce((s, x, i) => s + x * n[i], n[n.length - 1])));
        });
        // New semantics: Y = X · W + b with W [in, units].
        let current = input;
        for (let l = 0; l < 3; l++) {
            const kernel = artifact.weights[2 * l];
            const bias = artifact.weights[2 * l + 1].data;
            const [rows, cols] = kernel.shape;
            const z = Array.from({ length: cols }, (_, j) => {
                let s = bias[j];
                for (let i = 0; i < rows; i++) s += current[i] * kernel.data[i * cols + j];
                return s;
            });
            current = z.map(fns[l]);
        }
        assert.equal(current.length, 1);
        assert.ok(Math.abs(current[0] - legacy[0]) < 1e-12, `${current[0]} vs ${legacy[0]}`);
    });

    it("maps every legacy activation, keeping legacy alpha values", () => {
        const names = ["linear", "sigmoid", "tanh", "relu", "leakyRelu", "elu", "softmax", "swish"];
        const text = `1:linear:${names.map((n) => `1:${n}`).join(":")}\n` + names.map(() => "0.5:0.5").join("|");
        const activations = decodeLegacyOnn(text).layers.map((l) => l.activation);
        assert.deepEqual(activations, [
            { name: "linear" }, { name: "sigmoid" }, { name: "tanh" }, { name: "relu" },
            { name: "leakyRelu", alpha: 0.01 }, { name: "elu", alpha: 1 }, { name: "softmax" }, { name: "swish" },
        ]);
    });

    it("tolerates CRLF, a BOM, surrounding whitespace, trailing newlines and trailing pipes", () => {
        const expected = decodeLegacyOnn(DOC_EXAMPLE);
        for (const variant of [
            `${DOC_EXAMPLE}\n`,
            `${DOC_EXAMPLE}\r\n\r\n`,
            DOC_EXAMPLE.replace("\n", "\r\n"),
            `${DOC_EXAMPLE}|`,
            `${DOC_EXAMPLE}||\n`,
            `﻿${DOC_EXAMPLE}`,
            `\n  ${DOC_EXAMPLE.replace("\n", " \n ")}  \n`,
        ]) {
            assertArtifactsEqual(decodeLegacyOnn(variant), expected);
        }
    });

    it("parses every number form Number#toString produces", () => {
        const artifact = decodeLegacyOnn("2:linear:2:linear\n1e-7:-2.5E+3:.5|5.:+3:-0.000001");
        assert.deepEqual(Array.from(artifact.weights[0].data), [1e-7, 5, -2500, 3]);
        assert.deepEqual(Array.from(artifact.weights[1].data), [0.5, -0.000001]);
    });

    const malformed: [string, string, RegExp][] = [
        ["empty input", "  \n ", /input is empty/],
        ["a missing weights line", "2:relu:1:sigmoid", /missing line 2/],
        ["an extra line", `${DOC_EXAMPLE}\n1:2:3`, /expected exactly 2 lines \(structure, weights\), found 3/],
        ["a blank line between", DOC_EXAMPLE.replace("\n", "\n\n"), /expected exactly 2 lines/],
        ["an odd structure", "2:relu:1\n1:2:3", /odd number of ":"-separated fields \(3\)/],
        ["only an input layer", "2:relu\n1:2:3", /input layer and at least one more layer/],
        ["a zero input size", "0:relu:1:sigmoid\n1", /input layer: size must be a positive integer, got "0"/],
        ["a non-numeric size", "2:relu:x:sigmoid\n1:2:3", /layer 1 \(dense_1\): size must be a positive integer, got "x"/],
        ["a fractional size", "2:relu:1.5:sigmoid\n1:2:3", /size must be a positive integer, got "1.5"/],
        ["an unknown activation", "2:relu:1:gelu\n1:2:3", /layer 1 \(dense_1\): unknown activation "gelu" \(expected one of linear, sigmoid/],
        ["an empty activation", "2::1:relu\n1:2:3", /input layer: unknown activation ""/],
        ["too few neurons", "2:relu:2:swish\n0.71:-0.2:0.19", /line 2 has only 1 neuron, but the layer sizes 2:2 require 2 neurons \(2\)/],
        ["too many neurons", `${DOC_EXAMPLE}|1:2:3`, /line 2 has 3 neurons, but the layer sizes 2:2 require 2 neurons/],
        ["too few values", "2:relu:1:relu\n1:2", /dense_1 neuron 1 \(neuron 1 on the line\): expected 3 values \(2 weights \+ 1 bias\), got 2/],
        ["too many values", "2:relu:1:relu:1:relu\n1:2:3|4:5:6", /dense_2 neuron 1 \(neuron 2 on the line\): expected 2 values \(1 weights \+ 1 bias\), got 3/],
        ["a non-numeric weight", "2:relu:1:relu\nabc:2:3", /dense_1 neuron 1 .*weight 1: expected a finite number, got "abc"/],
        ["an empty weight", "2:relu:1:relu\n1::3", /weight 2: expected a finite number, got ""/],
        ["a non-numeric bias", "2:relu:1:relu\n1:2:b", /bias: expected a finite number, got "b"/],
        ["NaN", "2:relu:1:relu\nNaN:2:3", /got "NaN"/],
        ["Infinity", "2:relu:1:relu\n1:-Infinity:3", /got "-Infinity"/],
        ["hex", "2:relu:1:relu\n0x10:2:3", /got "0x10"/],
        ["an overflowing number", "2:relu:1:relu\n1e400:2:3", /1e400 overflows a float64/],
        ["an empty middle neuron", "2:relu:2:relu\n1:2:3| |4:5:6", /dense_1 neuron 2 \(neuron 2 on the line\) is empty \(stray "\|"\)/],
    ];

    for (const [label, text, message] of malformed) {
        it(`rejects ${label}`, () => throwsSerialization(() => decodeLegacyOnn(text), message));
    }
});

// ---------------------------------------------------------------------------------------------
// Detection & dispatch
// ---------------------------------------------------------------------------------------------

describe("detectFormat / decodeArtifact / encodeArtifact", () => {
    const artifact = sampleArtifact();
    const binary = encodeBinary(artifact, { precision: "float64" });
    const json = encodeJson(artifact);

    it("detects all three formats from strings, bytes and ArrayBuffers", () => {
        assert.equal(detectFormat(binary), "binary");
        assert.equal(detectFormat(binary.slice().buffer), "binary");
        assert.equal(detectFormat(json), "json");
        assert.equal(detectFormat(`﻿ \r\n\t${json}`), "json");
        assert.equal(detectFormat(utf8(json)), "json");
        assert.equal(detectFormat(utf8(`﻿\n  ${json}`)), "json");
        assert.equal(detectFormat(DOC_EXAMPLE), "legacy");
        assert.equal(detectFormat(utf8(DOC_EXAMPLE)), "legacy");
        assert.equal(detectFormat(`\n${DOC_EXAMPLE}`), "legacy");
    });

    it("routes damaged binary signatures to the binary decoder", () => {
        const damaged = new Uint8Array(binary.length - 1);
        damaged.set(binary.subarray(0, 6));
        damaged.set(binary.subarray(7), 6);
        assert.equal(detectFormat(damaged), "binary");
        throwsSerialization(() => decodeArtifact(damaged), /text mode/);
    });

    it("rejects unrecognized input", () => {
        throwsSerialization(() => detectFormat(""), /Unrecognized model format: input is empty/);
        throwsSerialization(() => detectFormat(" \n "), /input is empty/);
        throwsSerialization(() => detectFormat(new Uint8Array(0)), /input is empty/);
        throwsSerialization(() => detectFormat("hello"), /input starts with "hello"/);
        throwsSerialization(() => detectFormat("[1, 2]"), /expected a binary model .*JSON .*legacy/);
        throwsSerialization(() => detectFormat(Uint8Array.of(0, 1, 2)), /input starts with bytes 00 01 02/);
        const latin1 = String.fromCharCode(...binary.subarray(0, 64));
        throwsSerialization(() => detectFormat(latin1), /binary model that was converted to a string/);
        throwsSerialization(() => detectFormat(42 as never), /expected a Uint8Array or ArrayBuffer, got 42/);
    });

    it("decodes every format through decodeArtifact", () => {
        assertArtifactsEqual(decodeArtifact(binary), artifact);
        assertArtifactsEqual(decodeArtifact(binary.slice().buffer), artifact);
        assertArtifactsEqual(decodeArtifact(json), artifact);
        assertArtifactsEqual(decodeArtifact(utf8(json)), artifact);
        assertArtifactsEqual(decodeArtifact(utf8(encodeJson(artifact, { pretty: true }))), artifact);
        assertArtifactsEqual(decodeArtifact(DOC_EXAMPLE), decodeLegacyOnn(DOC_EXAMPLE));
        assertArtifactsEqual(decodeArtifact(utf8(DOC_EXAMPLE)), decodeLegacyOnn(DOC_EXAMPLE));
        throwsSerialization(() => decodeArtifact(Uint8Array.of(0x7b, 0xff, 0xfe)), /not valid UTF-8/);
    });

    it("encodes through encodeArtifact with the right return types", () => {
        const text: string = encodeArtifact(artifact, { format: "json", pretty: true });
        const bytes: Uint8Array = encodeArtifact(artifact, { format: "binary", precision: "float64" });
        assert.equal(text, encodeJson(artifact, { pretty: true }));
        assert.deepEqual(bytes, binary);
        assert.ok(encodeArtifact(artifact, { format: "binary" }).length < bytes.length);
        assert.throws(() => encodeArtifact(artifact, { format: "yaml" } as never), ValidationError);
        assert.throws(() => encodeArtifact(artifact, undefined as never), ValidationError);
    });

    it("converts legacy → binary → JSON → artifact without loss", () => {
        const legacy = decodeLegacyOnn("3:relu:2:tanh:2:elu:1:sigmoid\n0.1:0.2:0.3:0.4|0.5:0.6:0.7:0.8|-1:-2:3|4:5:-6|1.5:-2.5:0.25");
        const viaBinary = decodeArtifact(encodeArtifact(legacy, { format: "binary", precision: "float64" }));
        const viaJson = decodeArtifact(encodeArtifact(viaBinary, { format: "json" }));
        assertArtifactsEqual(viaJson, legacy);
    });
});
