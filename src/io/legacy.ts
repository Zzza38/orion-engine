/**
 * Importer for the legacy `.onn` text format written by orion-engine ≤ 0.0.2.
 *
 * ```text
 * 2:linear:4:relu:1:sigmoid
 * w:w:b|w:w:b|w:w:b|w:w:b|w:w:w:w:b
 * ```
 *
 * Line 1 lists `size:activation` pairs; the first pair is the input layer (its activation was
 * never applied). Line 2 lists the neurons of layers 1..n in order, separated by `|`; each neuron
 * is its incoming weights (one per previous-layer neuron) followed by its bias, separated by `:`.
 * Weight i of neuron j connects previous neuron i to neuron j, i.e. it is `kernel[i][j]`.
 */
import { SerializationError } from "../core/errors.js";
import type { JsonValue, LayerConfig, ModelArtifact, WeightEntry } from "../core/types.js";
import { describeValue, validateArtifact } from "./validate.js";

const ERROR_PREFIX = "Invalid legacy .onn model";

const LEGACY_ACTIVATIONS = ["linear", "sigmoid", "tanh", "relu", "leakyRelu", "elu", "softmax", "swish"] as const;
type LegacyActivation = (typeof LEGACY_ACTIVATIONS)[number];

/** Decimal number as written by JavaScript's Number#toString (no hex, no NaN/Infinity). */
const NUMBER_PATTERN = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

/**
 * Converts a legacy `.onn` text model into a {@link ModelArtifact} of dense layers named
 * `dense_1`, `dense_2`, …, with weights `"dense_k/kernel"` `[inputSize, units]` and
 * `"dense_k/bias"` `[1, units]` as Float64Arrays.
 *
 * Activations map 1:1. `leakyRelu` and `elu` carry their legacy `alpha` (0.01 and 1) explicitly so
 * the converted model computes exactly what the legacy engine did. The artifact's metadata records
 * `convertedFrom: "legacy-onn"`; it has no training config.
 *
 * Tolerates a leading byte-order mark, surrounding whitespace, CRLF line endings, a trailing
 * newline, and trailing `|` separators. Everything else is validated strictly.
 *
 * @param text Contents of the `.onn` file.
 * @returns A validated artifact.
 * @throws {SerializationError} Naming the line, layer, neuron and value that is wrong.
 */
export function decodeLegacyOnn(text: string): ModelArtifact {
    if (typeof text !== "string") throw legacyError(`expected a string, got ${describeValue(text)}`);
    const source = (text.charCodeAt(0) === 0xfeff ? text.slice(1) : text).trim();
    if (source.length === 0) throw legacyError("input is empty");

    const lines = source.split(/\r?\n/).map((line) => line.trim());
    if (lines.length < 2) {
        throw legacyError("missing line 2 (weights); a model needs at least one layer after the input layer");
    }
    if (lines.length > 2) {
        throw legacyError(`expected exactly 2 lines (structure, weights), found ${lines.length}`);
    }

    const { sizes, activations } = parseStructure(lines[0]);
    const segments = lines[1].split("|");
    while (segments.length > 0 && segments[segments.length - 1].trim() === "") segments.pop();
    const expectedNeurons = sizes.slice(1).reduce((sum, n) => sum + n, 0);
    const neuronCountHint = `layer sizes ${sizes.join(":")} require ${expectedNeurons} neurons (${sizes.slice(1).join(" + ")})`;

    const layers: LayerConfig[] = [];
    const weights: WeightEntry[] = [];
    let segment = 0;
    for (let layer = 1; layer < sizes.length; layer++) {
        const name = `dense_${layer}`;
        const fanIn = sizes[layer - 1];
        const units = sizes[layer];
        const kernel = new Float64Array(fanIn * units);
        const bias = new Float64Array(units);

        for (let j = 0; j < units; j++, segment++) {
            if (segment >= segments.length) {
                throw legacyError(`line 2 has only ${plural(segments.length, "neuron")}, but the ${neuronCountHint}`);
            }
            const where = `line 2, ${name} neuron ${j + 1} (neuron ${segment + 1} on the line)`;
            const raw = segments[segment];
            if (raw.trim() === "") throw legacyError(`${where} is empty (stray "|")`);
            const values = raw.split(":");
            if (values.length !== fanIn + 1) {
                throw legacyError(
                    `${where}: expected ${fanIn + 1} values (${fanIn} weights + 1 bias), got ${values.length}`,
                );
            }
            for (let i = 0; i < fanIn; i++) kernel[i * units + j] = parseNumber(values[i], `${where}, weight ${i + 1}`);
            bias[j] = parseNumber(values[fanIn], `${where}, bias`);
        }

        layers.push({
            type: "dense",
            name,
            units,
            activation: activationConfig(activations[layer]),
            useBias: true,
            kernelInitializer: { name: "glorotUniform" },
            biasInitializer: { name: "zeros" },
        });
        weights.push({ name: `${name}/kernel`, shape: [fanIn, units], data: kernel });
        weights.push({ name: `${name}/bias`, shape: [1, units], data: bias });
    }
    if (segments.length > expectedNeurons) {
        throw legacyError(`line 2 has ${plural(segments.length, "neuron")}, but the ${neuronCountHint}`);
    }

    return validateArtifact({
        format: "orion-engine",
        formatVersion: 1,
        inputSize: sizes[0],
        layers,
        weights,
        metadata: { convertedFrom: "legacy-onn" },
    } satisfies ModelArtifact);
}

function parseStructure(line: string): { sizes: number[]; activations: LegacyActivation[] } {
    const fields = line.split(":").map((field) => field.trim());
    if (fields.length % 2 !== 0) {
        throw legacyError(
            `line 1 must be "size:activation" pairs, but has an odd number of ":"-separated fields (${fields.length})`,
        );
    }
    if (fields.length < 4) {
        throw legacyError("line 1 must describe an input layer and at least one more layer (e.g. \"2:linear:1:sigmoid\")");
    }
    const sizes: number[] = [];
    const activations: LegacyActivation[] = [];
    for (let p = 0; p < fields.length; p += 2) {
        const layer = p / 2;
        const label = layer === 0 ? "input layer" : `layer ${layer} (dense_${layer})`;
        const sizeField = fields[p];
        const size = Number(sizeField);
        if (!/^\d+$/.test(sizeField) || !Number.isSafeInteger(size) || size < 1) {
            throw legacyError(`line 1, ${label}: size must be a positive integer, got ${JSON.stringify(sizeField)}`);
        }
        const activation = fields[p + 1];
        if (!isLegacyActivation(activation)) {
            throw legacyError(
                `line 1, ${label}: unknown activation ${JSON.stringify(activation)} ` +
                `(expected one of ${LEGACY_ACTIVATIONS.join(", ")})`,
            );
        }
        sizes.push(size);
        activations.push(activation);
    }
    return { sizes, activations };
}

function isLegacyActivation(name: string): name is LegacyActivation {
    return (LEGACY_ACTIVATIONS as readonly string[]).includes(name);
}

/** Typed as a JSON object (not ActivationConfig, whose `undefined`-able index signature is not a JsonValue). */
function activationConfig(name: LegacyActivation): { [key: string]: JsonValue } {
    switch (name) {
        case "leakyRelu":
            return { name, alpha: 0.01 };
        case "elu":
            return { name, alpha: 1 };
        default:
            return { name };
    }
}

function parseNumber(field: string, where: string): number {
    const trimmed = field.trim();
    if (!NUMBER_PATTERN.test(trimmed)) {
        throw legacyError(`${where}: expected a finite number, got ${JSON.stringify(trimmed)}`);
    }
    const value = Number(trimmed);
    if (!Number.isFinite(value)) throw legacyError(`${where}: ${trimmed} overflows a float64`);
    return value;
}

function plural(count: number, noun: string): string {
    return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function legacyError(message: string): SerializationError {
    return new SerializationError(`${ERROR_PREFIX}: ${message}`);
}
