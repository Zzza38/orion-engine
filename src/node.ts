/**
 * Node.js entry point (`@zzza38/orion-engine/node`): everything from the main entry, plus
 * saving and loading models to and from files.
 *
 * @example
 * import { Sequential, dense, saveModel, loadModel } from "@zzza38/orion-engine/node";
 *
 * await saveModel(model, "models/xor.onn");        // binary
 * await saveModel(model, "models/xor.onn.json");   // JSON (from the extension)
 * const restored = await loadModel("models/xor.onn");
 *
 * @packageDocumentation
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { ValidationError } from "./core/errors.js";
import type { Sequential } from "./model.js";
import type { DeserializeOptions, SerializeOptions } from "./serialization.js";
import { deserializeModel, serializeModel } from "./serialization.js";
import { describeValue } from "./utils.js";

export * from "./index.js";

/**
 * Options for {@link saveModel}: those of `serializeModel`. When `format` is omitted it is
 * inferred from the file name: `.json` (e.g. `model.onn.json`) → JSON, anything else → binary.
 */
export type SaveModelOptions = SerializeOptions;

/** Options for {@link loadModel}. */
export type LoadModelOptions = DeserializeOptions;

/**
 * Writes a model to `path` (creating parent directories as needed).
 * @example
 * await saveModel(model, "out/model.onn", { precision: "float64" });
 */
export async function saveModel(model: Sequential, path: string, options: SaveModelOptions = {}): Promise<void> {
    const data = encodeForPath(model, path, options, "saveModel");
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, data);
}

/** Synchronous {@link saveModel}. */
export function saveModelSync(model: Sequential, path: string, options: SaveModelOptions = {}): void {
    const data = encodeForPath(model, path, options, "saveModelSync");
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, data);
}

/**
 * Reads a model saved with {@link saveModel} (binary or JSON, detected from the content) or a
 * legacy `.onn` text model.
 * @example
 * const model = await loadModel("out/model.onn");
 */
export async function loadModel(path: string, options: LoadModelOptions = {}): Promise<Sequential> {
    checkPath(path, "loadModel");
    return decodeBytes(await readFile(path), options);
}

/** Synchronous {@link loadModel}. */
export function loadModelSync(path: string, options: LoadModelOptions = {}): Sequential {
    checkPath(path, "loadModelSync");
    return decodeBytes(readFileSync(path), options);
}

function encodeForPath(model: Sequential, path: string, options: SaveModelOptions, where: string): Uint8Array | string {
    checkPath(path, where);
    if (options === null || typeof options !== "object") {
        throw new ValidationError(`${where}: options must be an object, got ${describeValue(options)}`);
    }
    const format = options.format ?? (path.toLowerCase().endsWith(".json") ? "json" : "binary");
    return serializeModel(model, { ...options, format });
}

function decodeBytes(buffer: Uint8Array, options: LoadModelOptions): Sequential {
    return deserializeModel(new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength), options);
}

function checkPath(path: unknown, where: string): void {
    if (typeof path !== "string" || path.length === 0) {
        throw new ValidationError(`${where}: path must be a non-empty string, got ${describeValue(path)}`);
    }
}
