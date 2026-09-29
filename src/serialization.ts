/**
 * Saving and loading models as bytes or text (browser-safe; for files see `@zzza38/orion-engine/node`).
 */
import { OrionError, SerializationError, ValidationError } from "./core/errors.js";
import type { JsonValue } from "./core/types.js";
import type { BinaryPrecision } from "./io/index.js";
import { decodeArtifact, encodeArtifact } from "./io/index.js";
import { Sequential } from "./model.js";
import { checkOptions, describeValue } from "./utils.js";

/** Options for {@link serializeModel}. */
export interface SerializeOptions {
    /** "binary" (compact, checksummed; the default) or "json" (human-readable, bit-exact). */
    format?: "binary" | "json";
    /**
     * Binary only. "float32" (default) halves the size and keeps ~7 significant digits;
     * "float64" is bit-exact.
     */
    precision?: BinaryPrecision;
    /** JSON only. Indent the document. Default false. */
    pretty?: boolean;
    /** Extra JSON metadata stored with the model (the model name is stored as `metadata.name`). */
    metadata?: { [key: string]: JsonValue | undefined };
}

/** Options for {@link deserializeModel}. */
export interface DeserializeOptions {
    /** Seed for the loaded model's shuffling and dropout. Default random. */
    seed?: number;
}

const SERIALIZE_KEYS = ["format", "precision", "pretty", "metadata"];

/**
 * Encodes a built model (architecture, weights, and training config when compiled) in the binary
 * container or as JSON. Optimizer state is not saved.
 *
 * @example
 * const bytes = serializeModel(model);                                   // Uint8Array, float32 weights
 * const exact = serializeModel(model, { precision: "float64" });         // Uint8Array, bit-exact
 * const text = serializeModel(model, { format: "json", pretty: true });  // string
 */
export function serializeModel(model: Sequential, options: SerializeOptions & { format: "json" }): string;
export function serializeModel(model: Sequential, options?: SerializeOptions & { format?: "binary" }): Uint8Array;
export function serializeModel(model: Sequential, options?: SerializeOptions): Uint8Array | string;
export function serializeModel(model: Sequential, options: SerializeOptions = {}): Uint8Array | string {
    const where = "serializeModel";
    checkOptions(where, options, SERIALIZE_KEYS);
    if (!(model instanceof Sequential)) {
        throw new ValidationError(`${where}: expected a Sequential model, got ${describeValue(model)}`);
    }
    const format = options.format ?? "binary";
    const artifact = model.toArtifact(options.metadata);
    if (format === "json") {
        if (options.precision !== undefined) {
            throw new ValidationError(`${where}: "precision" only applies to format "binary" (JSON is always exact)`);
        }
        return encodeArtifact(artifact, { format: "json", pretty: options.pretty });
    }
    if (format === "binary") {
        if (options.pretty !== undefined) throw new ValidationError(`${where}: "pretty" only applies to format "json"`);
        return encodeArtifact(artifact, { format: "binary", precision: options.precision });
    }
    throw new ValidationError(`${where}: "format" must be "binary" or "json", got ${describeValue(format)}`);
}

/**
 * Decodes a model saved with {@link serializeModel} (binary or JSON, detected automatically) or
 * a legacy `.onn` text model from orion-engine ≤ 0.0.2. Models saved compiled come back compiled
 * (with a fresh optimizer).
 *
 * @example
 * const model = deserializeModel(new Uint8Array(await (await fetch("model.onn")).arrayBuffer()));
 * model.predict([0, 1]);
 * @throws SerializationError if the data is not a valid model.
 */
export function deserializeModel(
    data: string | Uint8Array | ArrayBuffer,
    options: DeserializeOptions = {},
): Sequential {
    checkOptions("deserializeModel", options, ["seed"]);
    const artifact = decodeArtifact(data);
    try {
        return Sequential.fromArtifact(artifact, options);
    } catch (error) {
        // A well-formed file describing an impossible model (unknown layer type, weight shape
        // mismatch, ...) is still a file that cannot be loaded: report it as one error type.
        if (error instanceof SerializationError || !(error instanceof OrionError)) throw error;
        throw new SerializationError(`deserializeModel: ${error.message}`, { cause: error });
    }
}
