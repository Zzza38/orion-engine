/**
 * Structural validation of {@link ModelArtifact} values.
 *
 * Every decoder funnels its result through {@link validateArtifact}, and every encoder validates
 * its input with it, so a malformed artifact is rejected at the boundary with a message that names
 * the offending field instead of surfacing later as a confusing shape error.
 */
import { SerializationError } from "../core/errors.js";
import type { ModelArtifact } from "../core/types.js";

const PREFIX = "Invalid model artifact";

/** The only `format` marker this library reads and writes. */
export const ARTIFACT_FORMAT = "orion-engine";

/** The newest artifact schema version this library understands. */
export const ARTIFACT_FORMAT_VERSION = 1;

/**
 * Checks that `value` is a structurally valid {@link ModelArtifact} and returns it, narrowed.
 *
 * Checks performed:
 *  - `format` is `"orion-engine"` and `formatVersion` is `1`;
 *  - `inputSize` is a positive integer;
 *  - `layers` is an array of plain objects with non-empty, unique string `name`s and non-empty
 *    string `type`s, whose values are all JSON-representable;
 *  - `weights` is an array of entries with non-empty, unique `name`s, a `[rows, cols]` shape of
 *    positive integers, and `data` (number[], Float64Array or Float32Array) of exactly
 *    `rows * cols` finite numbers;
 *  - optional `training` has `loss.name`, `optimizer.name` and a string array `metrics`;
 *  - optional `metadata` is a plain object of JSON values.
 *
 * Unknown top-level keys are tolerated (and ignored by the encoders). The value is not copied or
 * normalized; the returned reference is `value` itself.
 *
 * @param value Anything, typically freshly parsed JSON.
 * @returns `value`, typed as a {@link ModelArtifact}.
 * @throws {SerializationError} Naming the first offending field, what was expected, and what was found.
 */
export function validateArtifact(value: unknown): ModelArtifact {
    if (!isPlainObject(value)) fail("artifact", "an object", value);

    if (value.format !== ARTIFACT_FORMAT) fail("format", `"${ARTIFACT_FORMAT}"`, value.format);

    const version = value.formatVersion;
    if (version !== ARTIFACT_FORMAT_VERSION) {
        if (typeof version === "number" && Number.isInteger(version) && version > ARTIFACT_FORMAT_VERSION) {
            throw new SerializationError(
                `${PREFIX}: formatVersion ${version} is newer than this library supports ` +
                `(${ARTIFACT_FORMAT_VERSION}); upgrade @zzza38/orion-engine to read it`,
            );
        }
        fail("formatVersion", String(ARTIFACT_FORMAT_VERSION), version);
    }

    if (!isPositiveInteger(value.inputSize)) fail("inputSize", "a positive integer", value.inputSize);

    validateLayers(value.layers);
    validateWeights(value.weights);
    if (value.training !== undefined) validateTraining(value.training);
    if (value.metadata !== undefined) {
        if (!isPlainObject(value.metadata)) fail("metadata", "a plain object", value.metadata);
        checkJsonValue(value.metadata, "metadata", new Set());
    }

    return value as unknown as ModelArtifact;
}

function validateLayers(layers: unknown): void {
    if (!Array.isArray(layers)) fail("layers", "an array of layer configs", layers);
    const seen = new Set<string>();
    for (let i = 0; i < layers.length; i++) {
        const path = `layers[${i}]`;
        const layer: unknown = layers[i];
        if (!isPlainObject(layer)) fail(path, "a layer config object", layer);
        if (typeof layer.type !== "string" || layer.type.length === 0) {
            fail(`${path}.type`, "a non-empty string", layer.type);
        }
        if (typeof layer.name !== "string" || layer.name.length === 0) {
            fail(`${path}.name`, "a non-empty string", layer.name);
        }
        if (seen.has(layer.name)) {
            throw new SerializationError(`${PREFIX}: ${path}.name: duplicate layer name ${JSON.stringify(layer.name)}`);
        }
        seen.add(layer.name);
        checkJsonValue(layer, path, new Set());
    }
}

function validateWeights(weights: unknown): void {
    if (!Array.isArray(weights)) fail("weights", "an array of weight entries", weights);
    const seen = new Set<string>();
    for (let i = 0; i < weights.length; i++) {
        let path = `weights[${i}]`;
        const entry: unknown = weights[i];
        if (!isPlainObject(entry)) fail(path, "a weight entry object", entry);

        const name = entry.name;
        if (typeof name !== "string" || name.length === 0) fail(`${path}.name`, "a non-empty string", name);
        if (seen.has(name)) {
            throw new SerializationError(`${PREFIX}: ${path}.name: duplicate weight name ${JSON.stringify(name)}`);
        }
        seen.add(name);
        path = `${path} (${JSON.stringify(name)})`;

        const shape = entry.shape;
        if (!isShape(shape)) fail(`${path}.shape`, "[rows, cols] of positive integers", shape);

        const data = entry.data;
        if (!(Array.isArray(data) || data instanceof Float64Array || data instanceof Float32Array)) {
            fail(`${path}.data`, "a number[], Float64Array or Float32Array", data);
        }
        const expected = shape[0] * shape[1];
        if (data.length !== expected) {
            throw new SerializationError(
                `${PREFIX}: ${path}.data: expected ${expected} values for shape [${shape[0]}, ${shape[1]}], ` +
                `got ${data.length}`,
            );
        }
        for (let k = 0; k < data.length; k++) {
            const v: unknown = data[k];
            if (!Number.isFinite(v)) fail(`${path}.data[${k}]`, "a finite number", v);
        }
    }
}

function validateTraining(training: unknown): void {
    if (!isPlainObject(training)) fail("training", "an object", training);
    for (const key of ["loss", "optimizer"] as const) {
        const config = training[key];
        if (!isPlainObject(config)) fail(`training.${key}`, `a ${key} config object`, config);
        if (typeof config.name !== "string" || config.name.length === 0) {
            fail(`training.${key}.name`, "a non-empty string", config.name);
        }
    }
    const metrics = training.metrics;
    if (!Array.isArray(metrics)) fail("training.metrics", "an array of metric names", metrics);
    for (let i = 0; i < metrics.length; i++) {
        const metric: unknown = metrics[i];
        if (typeof metric !== "string") fail(`training.metrics[${i}]`, "a metric name string", metric);
    }
    checkJsonValue(training, "training", new Set());
}

/**
 * Recursively checks that `value` survives a JSON round-trip unchanged: plain objects, arrays,
 * strings, booleans, null, and finite numbers. `undefined` is allowed only as an object property
 * value (it is dropped on encode, like JSON.stringify does).
 */
function checkJsonValue(value: unknown, path: string, ancestors: Set<object>): void {
    switch (typeof value) {
        case "string":
        case "boolean":
            return;
        case "number":
            if (!Number.isFinite(value)) fail(path, "a finite number (JSON cannot represent NaN or Infinity)", value);
            return;
        case "object": {
            if (value === null) return;
            if (ancestors.has(value)) throw new SerializationError(`${PREFIX}: ${path}: circular reference`);
            ancestors.add(value);
            if (Array.isArray(value)) {
                for (let i = 0; i < value.length; i++) {
                    const item: unknown = value[i];
                    if (item === undefined) fail(`${path}[${i}]`, "a JSON value", item);
                    checkJsonValue(item, `${path}[${i}]`, ancestors);
                }
            } else if (isPlainObject(value)) {
                for (const key of Object.keys(value)) {
                    const item = value[key];
                    if (item !== undefined) checkJsonValue(item, propertyPath(path, key), ancestors);
                }
            } else {
                fail(path, "a JSON value (plain object, array, string, finite number, boolean or null)", value);
            }
            ancestors.delete(value);
            return;
        }
        default:
            fail(path, "a JSON value (plain object, array, string, finite number, boolean or null)", value);
    }
}

function fail(path: string, expected: string, got: unknown): never {
    throw new SerializationError(`${PREFIX}: ${path}: expected ${expected}, got ${describeValue(got)}`);
}

function propertyPath(path: string, key: string): string {
    return /^[A-Za-z_$][\w$]*$/.test(key) ? `${path}.${key}` : `${path}[${JSON.stringify(key)}]`;
}

/** @internal True for `{...}` objects (including null-prototype ones); false for arrays, typed arrays, Dates, etc. */
export function isPlainObject(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && Object.prototype.toString.call(value) === "[object Object]";
}

/** @internal */
export function isPositiveInteger(value: unknown): value is number {
    return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

/** @internal */
export function isNonNegativeInteger(value: unknown): value is number {
    return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

/** @internal A `[rows, cols]` tuple of positive integers. */
export function isShape(value: unknown): value is [number, number] {
    return Array.isArray(value) && value.length === 2 && isPositiveInteger(value[0]) && isPositiveInteger(value[1]);
}

/** @internal Short, human-readable description of a value for error messages. */
export function describeValue(value: unknown): string {
    switch (typeof value) {
        case "undefined":
            return "undefined";
        case "string":
            return value.length > 40 ? `${JSON.stringify(value.slice(0, 40))}…` : JSON.stringify(value);
        case "number":
            return Object.is(value, -0) ? "-0" : String(value);
        case "boolean":
            return String(value);
        case "bigint":
            return `${value}n (a bigint)`;
        case "function":
            return "a function";
        case "symbol":
            return "a symbol";
    }
    if (value === null) return "null";
    if (Array.isArray(value)) {
        const short = value.length <= 4 && value.every((v) => v === null || typeof v !== "object");
        return short ? `[${value.map(describeValue).join(", ")}]` : `an array of length ${value.length}`;
    }
    const tag = Object.prototype.toString.call(value).slice(8, -1);
    if (ArrayBuffer.isView(value) && "length" in value) return `a ${tag} of length ${String(value.length)}`;
    return tag === "Object" ? "an object" : `a ${tag}`;
}
