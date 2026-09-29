/**
 * JSON encoding of {@link ModelArtifact}: human-readable, diff-friendly, lossless.
 *
 * Weight data is written as plain number arrays using the shortest decimal that round-trips to the
 * same float64 (so decode(encode(x)) is bit-exact), with negative zero written as `-0`.
 */
import { SerializationError } from "../core/errors.js";
import type { ModelArtifact, WeightEntry } from "../core/types.js";
import { describeValue, validateArtifact } from "./validate.js";

/** Options for {@link encodeJson}. */
export interface JsonEncodeOptions {
    /**
     * Indent the document (2 spaces) and write each weight matrix one row per line.
     * Default `false`: a single compact line.
     */
    pretty?: boolean;
}

/**
 * Encodes an artifact as JSON text.
 *
 * Top-level keys are written in a fixed order (`format`, `formatVersion`, `inputSize`, `layers`,
 * `training`, `metadata`, `weights`); weights come last because they are by far the largest part.
 * Float32Array data is widened to float64 exactly. Unknown top-level keys are not written.
 *
 * @param artifact The artifact to encode. Validated first.
 * @param options See {@link JsonEncodeOptions}.
 * @returns JSON text (no trailing newline).
 * @throws {SerializationError} If the artifact is structurally invalid.
 */
export function encodeJson(artifact: ModelArtifact, options: JsonEncodeOptions = {}): string {
    validateArtifact(artifact);
    return options.pretty === true ? writePretty(artifact) : writeCompact(artifact);
}

/**
 * Decodes JSON text produced by {@link encodeJson} (or any equivalent JSON document).
 * A leading UTF-8 byte-order mark is ignored. Weight data is returned as `Float64Array`s, ready to
 * back a `Matrix` without copying. Unknown top-level keys are ignored.
 *
 * @param text The JSON document.
 * @returns A validated artifact.
 * @throws {SerializationError} If the text is not valid JSON or does not describe a valid artifact.
 */
export function decodeJson(text: string): ModelArtifact {
    if (typeof text !== "string") {
        throw new SerializationError(`Invalid JSON model: expected a string, got ${describeValue(text)}`);
    }
    const source = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
    let parsed: unknown;
    try {
        parsed = JSON.parse(source);
    } catch (error) {
        throw new SerializationError(`Invalid JSON model: ${error instanceof Error ? error.message : String(error)}`);
    }
    const artifact = validateArtifact(parsed);
    const result: ModelArtifact = {
        format: artifact.format,
        formatVersion: artifact.formatVersion,
        inputSize: artifact.inputSize,
        layers: artifact.layers,
        weights: artifact.weights.map((w) => ({
            name: w.name,
            shape: [w.shape[0], w.shape[1]],
            data: Float64Array.from(w.data),
        })),
    };
    if (artifact.training !== undefined) result.training = artifact.training;
    if (artifact.metadata !== undefined) result.metadata = artifact.metadata;
    return result;
}

/** Every top-level entry except `weights`, in canonical order. */
function headEntries(artifact: ModelArtifact): [string, unknown][] {
    const entries: [string, unknown][] = [
        ["format", artifact.format],
        ["formatVersion", artifact.formatVersion],
        ["inputSize", artifact.inputSize],
        ["layers", artifact.layers],
    ];
    if (artifact.training !== undefined) entries.push(["training", artifact.training]);
    if (artifact.metadata !== undefined) entries.push(["metadata", artifact.metadata]);
    return entries;
}

function writeCompact(artifact: ModelArtifact): string {
    const head = headEntries(artifact).map(([key, value]) => `${JSON.stringify(key)}:${JSON.stringify(value)}`);
    const weights = artifact.weights.map(
        (w) =>
            `{"name":${JSON.stringify(w.name)},"shape":[${w.shape[0]},${w.shape[1]}],` +
            `"data":[${formatRange(w.data, 0, w.data.length, ",")}]}`,
    );
    return `{${head.join(",")},"weights":[${weights.join(",")}]}`;
}

function writePretty(artifact: ModelArtifact): string {
    const lines: string[] = ["{"];
    for (const [key, value] of headEntries(artifact)) {
        // JSON strings never contain raw newlines, so re-indenting on "\n" is safe.
        lines.push(`  ${JSON.stringify(key)}: ${JSON.stringify(value, null, 2).replace(/\n/g, "\n  ")},`);
    }
    if (artifact.weights.length === 0) {
        lines.push(`  "weights": []`);
    } else {
        lines.push(`  "weights": [`);
        artifact.weights.forEach((w, i) => {
            lines.push("    {");
            lines.push(`      "name": ${JSON.stringify(w.name)},`);
            lines.push(`      "shape": [${w.shape[0]}, ${w.shape[1]}],`);
            lines.push(`      "data": ${formatPrettyData(w)}`);
            lines.push(i === artifact.weights.length - 1 ? "    }" : "    },");
        });
        lines.push("  ]");
    }
    lines.push("}");
    return lines.join("\n");
}

/** Single-row matrices stay on one line; larger ones get one matrix row per line. */
function formatPrettyData(weight: WeightEntry): string {
    const [rows, cols] = weight.shape;
    if (rows === 1) return `[${formatRange(weight.data, 0, cols, ", ")}]`;
    const out: string[] = [];
    for (let r = 0; r < rows; r++) {
        const row = formatRange(weight.data, r * cols, (r + 1) * cols, ", ");
        out.push(`        ${row}${r === rows - 1 ? "" : ","}`);
    }
    return `[\n${out.join("\n")}\n      ]`;
}

function formatRange(data: WeightEntry["data"], start: number, end: number, separator: string): string {
    const parts = new Array<string>(end - start);
    for (let i = start; i < end; i++) parts[i - start] = formatNumber(data[i]);
    return parts.join(separator);
}

/** Shortest round-trip decimal; unlike JSON.stringify, keeps the sign of negative zero. */
function formatNumber(value: number): string {
    return Object.is(value, -0) ? "-0" : String(value);
}
