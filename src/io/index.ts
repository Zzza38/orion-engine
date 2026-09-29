/**
 * Model file formats.
 *
 * - **Binary** (`.onn`): compact, checksummed container; float32 (default) or float64 weights.
 * - **JSON** (`.onn.json`): human-readable, lossless.
 * - **Legacy** (`.onn` text from orion-engine ≤ 0.0.2): import only.
 *
 * All of them decode to the same in-memory {@link ModelArtifact}. See docs/format.md.
 */
import { SerializationError, ValidationError } from "../core/errors.js";
import type { ModelArtifact } from "../core/types.js";
import { decodeBinary, encodeBinary, hasBinarySignature, toBytes } from "./binary.js";
import type { BinaryEncodeOptions } from "./binary.js";
import { decodeJson, encodeJson } from "./json.js";
import type { JsonEncodeOptions } from "./json.js";
import { decodeLegacyOnn } from "./legacy.js";
import { describeValue } from "./validate.js";

export { crc32, decodeBinary, encodeBinary } from "./binary.js";
export type { BinaryEncodeOptions, BinaryPrecision } from "./binary.js";
export { decodeJson, encodeJson } from "./json.js";
export type { JsonEncodeOptions } from "./json.js";
export { decodeLegacyOnn } from "./legacy.js";
export { validateArtifact } from "./validate.js";

/** Serialized model formats {@link decodeArtifact} understands. */
export type ArtifactFormat = "binary" | "json" | "legacy";

/** Options for {@link encodeArtifact}: the target format plus that format's own options. */
export type EncodeArtifactOptions =
    | ({ format: "json" } & JsonEncodeOptions)
    | ({ format: "binary" } & BinaryEncodeOptions);

/**
 * Identifies the format of serialized model data without decoding it.
 *
 * - bytes starting with `0x89 "ORION"` → `"binary"`;
 * - text (or UTF-8 bytes) whose first non-whitespace character, after an optional byte-order
 *   mark, is `{` → `"json"`;
 * - ... or is a digit → `"legacy"`.
 *
 * @param data File contents as text, bytes, or an ArrayBuffer.
 * @throws {SerializationError} If the data matches none of the formats (including empty input,
 *   and binary data that was converted to a string).
 */
export function detectFormat(data: string | Uint8Array | ArrayBuffer): ArtifactFormat {
    if (typeof data === "string") {
        if (data.startsWith("\u0089ORION")) {
            throw new SerializationError(
                "Unrecognized model format: this looks like a binary model that was converted to a string; " +
                "pass the raw bytes (Uint8Array or ArrayBuffer) instead",
            );
        }
        let i = data.charCodeAt(0) === 0xfeff ? 1 : 0;
        while (i < data.length && isWhitespace(data.charCodeAt(i))) i++;
        return classifyText(i < data.length ? data.charCodeAt(i) : -1, JSON.stringify(data.slice(i, i + 16)));
    }
    const bytes = toBytes(data, "Unrecognized model format");
    if (hasBinarySignature(bytes)) return "binary";
    let i = bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf ? 3 : 0;
    while (i < bytes.length && isWhitespace(bytes[i])) i++;
    const preview = Array.from(bytes.subarray(i, i + 8), (b) => b.toString(16).toUpperCase().padStart(2, "0")).join(" ");
    return classifyText(i < bytes.length ? bytes[i] : -1, `bytes ${preview}`);
}

/**
 * Decodes serialized model data of any supported format, detected with {@link detectFormat}.
 *
 * @param data File contents: a string (JSON or legacy), or bytes / an ArrayBuffer (any format;
 *   text formats must be UTF-8).
 * @returns A validated artifact whose weight data are Float64Arrays.
 * @throws {SerializationError} If the format is unrecognized or the data is malformed.
 */
export function decodeArtifact(data: string | Uint8Array | ArrayBuffer): ModelArtifact {
    const format = detectFormat(data);
    if (format === "binary") return decodeBinary(data as Uint8Array | ArrayBuffer);
    const text = typeof data === "string" ? data : decodeUtf8(toBytes(data, "Invalid model"));
    return format === "json" ? decodeJson(text) : decodeLegacyOnn(text);
}

/**
 * Encodes an artifact as JSON text or as the binary container.
 *
 * @example
 * const bytes = encodeArtifact(artifact, { format: "binary", precision: "float64" });
 * const text = encodeArtifact(artifact, { format: "json", pretty: true });
 * @throws {SerializationError} If the artifact is invalid.
 * @throws {ValidationError} If the options are invalid.
 */
export function encodeArtifact(artifact: ModelArtifact, options: { format: "json" } & JsonEncodeOptions): string;
export function encodeArtifact(artifact: ModelArtifact, options: { format: "binary" } & BinaryEncodeOptions): Uint8Array;
export function encodeArtifact(artifact: ModelArtifact, options: EncodeArtifactOptions): string | Uint8Array;
export function encodeArtifact(artifact: ModelArtifact, options: EncodeArtifactOptions): string | Uint8Array {
    if (options?.format === "json") return encodeJson(artifact, options);
    if (options?.format === "binary") return encodeBinary(artifact, options);
    const format: unknown = (options as { format?: unknown } | undefined)?.format;
    throw new ValidationError(`encodeArtifact: options.format must be "json" or "binary", got ${describeValue(format)}`);
}

function classifyText(firstChar: number, preview: string): ArtifactFormat {
    if (firstChar === 0x7b /* { */) return "json";
    if (firstChar >= 0x30 && firstChar <= 0x39) return "legacy";
    if (firstChar === -1) throw new SerializationError("Unrecognized model format: input is empty");
    throw new SerializationError(
        "Unrecognized model format: expected a binary model (0x89 \"ORION\"), JSON (\"{\"), or a legacy " +
        `.onn file (starting with a digit); input starts with ${preview}`,
    );
}

function isWhitespace(code: number): boolean {
    return code === 0x20 || code === 0x09 || code === 0x0a || code === 0x0d;
}

function decodeUtf8(bytes: Uint8Array): string {
    try {
        return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch {
        throw new SerializationError("Invalid model: text model data is not valid UTF-8");
    }
}
