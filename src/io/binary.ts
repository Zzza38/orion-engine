/**
 * Compact binary container for {@link ModelArtifact} (recommended extension: `.onn`).
 *
 * Layout (all integers little-endian; see docs/format.md for the full specification):
 *
 * | offset | size | field                                                     |
 * |--------|------|-----------------------------------------------------------|
 * | 0      | 8    | magic `89 4F 52 49 4F 4E 0D 0A` (`\x89ORION\r\n`)         |
 * | 8      | 2    | u16 container version (1)                                 |
 * | 10     | 2    | u16 flags (reserved, 0)                                   |
 * | 12     | 4    | u32 header byte length H                                  |
 * | 16     | H    | UTF-8 JSON header                                         |
 * | 16+H   | P    | zero padding to the next multiple of 8 (D = 16 + H + P)   |
 * | D      | L    | weight data (L = header.dataByteLength)                   |
 * | D+L    | 4    | u32 CRC-32 of bytes [0, D+L)                              |
 *
 * Only DataView / TextEncoder / TextDecoder / typed arrays are used, so this runs unchanged in
 * browsers, and every multi-byte value goes through DataView with an explicit little-endian flag,
 * so host endianness never matters.
 */
import { SerializationError, ValidationError } from "../core/errors.js";
import type { ModelArtifact, WeightEntry } from "../core/types.js";
import { describeValue, isNonNegativeInteger, isPlainObject, isShape, validateArtifact } from "./validate.js";

/** Numeric precision of weight data in the binary container. */
export type BinaryPrecision = "float32" | "float64";

/** Options for {@link encodeBinary}. */
export interface BinaryEncodeOptions {
    /**
     * Storage precision for weights. `"float32"` (default) halves the file size and keeps ~7
     * significant digits; `"float64"` is bit-exact.
     */
    precision?: BinaryPrecision;
}

/** Per-weight record stored in the JSON header in place of the weight's data. */
interface WeightDescriptor {
    name: string;
    shape: [number, number];
    dtype: BinaryPrecision;
    /** Offset from the start of the data section, in bytes. */
    byteOffset: number;
    /** Element count (rows * cols). */
    length: number;
}

/** `0x89 "ORION" CR LF`: the high byte rules out text/JSON, the CR LF detects line-ending mangling. */
const MAGIC = Uint8Array.of(0x89, 0x4f, 0x52, 0x49, 0x4f, 0x4e, 0x0d, 0x0a);
const CONTAINER_VERSION = 1;
/** Magic + version + flags + header length. */
const PREFIX_BYTES = 16;
const CRC_BYTES = 4;
const MIN_FILE_BYTES = PREFIX_BYTES + CRC_BYTES;
const DTYPE_BYTES: Record<BinaryPrecision, number> = { float32: 4, float64: 8 };
const ERROR_PREFIX = "Invalid binary model";

/**
 * Encodes an artifact into the binary container.
 *
 * Weights are stored in artifact order, each starting on an 8-byte boundary of the data section,
 * so a reader holding an aligned buffer on a little-endian host could view them in place.
 *
 * @param artifact The artifact to encode. Validated first.
 * @param options See {@link BinaryEncodeOptions}.
 * @returns A new, tightly sized Uint8Array (its byteOffset is 0).
 * @throws {SerializationError} If the artifact is invalid, or a value overflows float32.
 * @throws {ValidationError} If `options.precision` is not `"float32"` or `"float64"`.
 */
export function encodeBinary(artifact: ModelArtifact, options: BinaryEncodeOptions = {}): Uint8Array {
    const precision = options.precision ?? "float32";
    if (precision !== "float32" && precision !== "float64") {
        throw new ValidationError(`encodeBinary: precision must be "float32" or "float64", got ${describeValue(precision)}`);
    }
    validateArtifact(artifact);

    const elementBytes = DTYPE_BYTES[precision];
    const descriptors: WeightDescriptor[] = [];
    let cursor = 0;
    for (const weight of artifact.weights) {
        const byteOffset = align8(cursor);
        descriptors.push({
            name: weight.name,
            shape: [weight.shape[0], weight.shape[1]],
            dtype: precision,
            byteOffset,
            length: weight.data.length,
        });
        cursor = byteOffset + weight.data.length * elementBytes;
    }
    const dataByteLength = align8(cursor);

    const header: Record<string, unknown> = {
        format: artifact.format,
        formatVersion: artifact.formatVersion,
        inputSize: artifact.inputSize,
        layers: artifact.layers,
    };
    if (artifact.training !== undefined) header.training = artifact.training;
    if (artifact.metadata !== undefined) header.metadata = artifact.metadata;
    header.weights = descriptors;
    header.dataByteLength = dataByteLength;

    const headerBytes = new TextEncoder().encode(JSON.stringify(header));
    if (headerBytes.length > 0xffffffff) {
        throw new SerializationError(`Cannot encode binary model: header is ${headerBytes.length} bytes (max 4 GiB)`);
    }
    const dataStart = align8(PREFIX_BYTES + headerBytes.length);
    const totalBytes = dataStart + dataByteLength + CRC_BYTES;

    const bytes = new Uint8Array(totalBytes); // zero-filled, so all padding is already zero
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    bytes.set(MAGIC, 0);
    view.setUint16(8, CONTAINER_VERSION, true);
    view.setUint16(10, 0, true);
    view.setUint32(12, headerBytes.length, true);
    bytes.set(headerBytes, PREFIX_BYTES);

    for (let w = 0; w < artifact.weights.length; w++) {
        const { name, data } = artifact.weights[w];
        let offset = dataStart + descriptors[w].byteOffset;
        if (precision === "float64") {
            for (let i = 0; i < data.length; i++, offset += 8) view.setFloat64(offset, data[i], true);
        } else {
            for (let i = 0; i < data.length; i++, offset += 4) {
                const narrowed = Math.fround(data[i]);
                if (!Number.isFinite(narrowed)) {
                    throw new SerializationError(
                        `Cannot encode weight ${JSON.stringify(name)} as float32: value ${data[i]} at index ${i} ` +
                        `exceeds the float32 range (±3.4028234663852886e38); use precision "float64"`,
                    );
                }
                view.setFloat32(offset, narrowed, true);
            }
        }
    }

    const crcOffset = totalBytes - CRC_BYTES;
    view.setUint32(crcOffset, crc32(bytes.subarray(0, crcOffset)), true);
    return bytes;
}

/**
 * Decodes a binary container produced by {@link encodeBinary}.
 *
 * Checks, in order: magic (with a hint when line endings were mangled by a text-mode transfer),
 * container version, reserved flags, declared sizes against the actual length (truncation /
 * trailing bytes), the CRC-32, and finally the header and weight descriptors. Weight data is
 * returned as `Float64Array`s (float32 values are widened exactly).
 *
 * @param bytes The file contents. A Uint8Array may be a view at any byteOffset into a larger buffer.
 * @returns A validated artifact.
 * @throws {SerializationError} Describing the first problem found.
 */
export function decodeBinary(bytes: Uint8Array | ArrayBuffer): ModelArtifact {
    const input = toBytes(bytes, ERROR_PREFIX);
    checkMagic(input);
    if (input.length < MIN_FILE_BYTES) {
        throw binaryError(`file is truncated: ${input.length} bytes, but even an empty container needs ${MIN_FILE_BYTES}`);
    }
    const view = new DataView(input.buffer, input.byteOffset, input.byteLength);

    const version = view.getUint16(8, true);
    if (version !== CONTAINER_VERSION) {
        throw binaryError(
            version > CONTAINER_VERSION
                ? `container version ${version} is newer than this library supports (${CONTAINER_VERSION}); ` +
                  "upgrade @zzza38/orion-engine to read it"
                : `unsupported container version ${version}`,
        );
    }
    const flags = view.getUint16(10, true);
    if (flags !== 0) throw binaryError(`reserved flags field must be 0, got 0x${hex16(flags)}`);

    const headerLength = view.getUint32(12, true);
    const headerEnd = PREFIX_BYTES + headerLength;
    if (headerEnd + CRC_BYTES > input.length) {
        throw binaryError(
            `file is truncated: header declares ${headerLength} bytes, but the file is only ${input.length} bytes long`,
        );
    }
    const dataStart = align8(headerEnd);

    // Parse the header before the CRC check only to turn truncation into a precise message; the
    // header is not trusted (or even required to parse) until the checksum has been verified.
    let header: unknown;
    let headerError: string | undefined;
    try {
        header = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(input.subarray(PREFIX_BYTES, headerEnd)));
    } catch (error) {
        headerError = error instanceof Error ? error.message : String(error);
    }
    const declaredDataLength =
        isPlainObject(header) && isNonNegativeInteger(header.dataByteLength) ? header.dataByteLength : undefined;
    if (declaredDataLength !== undefined) {
        const expectedBytes = dataStart + declaredDataLength + CRC_BYTES;
        if (input.length < expectedBytes) {
            throw binaryError(`file is truncated: expected ${expectedBytes} bytes, got ${input.length}`);
        }
        if (input.length > expectedBytes) {
            throw binaryError(
                `${input.length - expectedBytes} unexpected trailing bytes (expected ${expectedBytes} bytes, got ${input.length})`,
            );
        }
    }

    const crcOffset = input.length - CRC_BYTES;
    const storedCrc = view.getUint32(crcOffset, true);
    const actualCrc = crc32(input.subarray(0, crcOffset));
    if (storedCrc !== actualCrc) {
        throw binaryError(
            `CRC-32 mismatch (stored 0x${hex32(storedCrc)}, computed 0x${hex32(actualCrc)}); the file is corrupted`,
        );
    }

    if (headerError !== undefined) throw binaryError(`header is not valid UTF-8 JSON: ${headerError}`);
    if (!isPlainObject(header)) throw binaryError(`header: expected a JSON object, got ${describeValue(header)}`);
    if (declaredDataLength === undefined) {
        throw binaryError(`header.dataByteLength: expected a non-negative integer, got ${describeValue(header.dataByteLength)}`);
    }

    const descriptors = header.weights;
    if (!Array.isArray(descriptors)) {
        throw binaryError(`header.weights: expected an array of weight descriptors, got ${describeValue(descriptors)}`);
    }
    const weights: WeightEntry[] = [];
    for (let i = 0; i < descriptors.length; i++) {
        const path = `header.weights[${i}]`;
        const d: unknown = descriptors[i];
        if (!isPlainObject(d)) throw binaryError(`${path}: expected a weight descriptor object, got ${describeValue(d)}`);
        if (!isShape(d.shape)) {
            throw binaryError(`${path}.shape: expected [rows, cols] of positive integers, got ${describeValue(d.shape)}`);
        }
        const dtype = d.dtype;
        if (dtype !== "float32" && dtype !== "float64") {
            throw binaryError(`${path}.dtype: expected "float32" or "float64", got ${describeValue(dtype)}`);
        }
        const [rows, cols] = d.shape;
        const length = rows * cols;
        if (d.length !== length) {
            throw binaryError(`${path}.length: expected ${length} (shape [${rows}, ${cols}]), got ${describeValue(d.length)}`);
        }
        const byteOffset = d.byteOffset;
        if (!isNonNegativeInteger(byteOffset)) {
            throw binaryError(`${path}.byteOffset: expected a non-negative integer, got ${describeValue(byteOffset)}`);
        }
        const byteEnd = byteOffset + length * DTYPE_BYTES[dtype];
        if (byteEnd > declaredDataLength) {
            throw binaryError(
                `${path}: bytes [${byteOffset}, ${byteEnd}) lie outside the ${declaredDataLength}-byte data section`,
            );
        }
        weights.push({ name: d.name as string, shape: [rows, cols], data: readData(view, dataStart + byteOffset, length, dtype) });
    }

    const artifact: Record<string, unknown> = {
        format: header.format,
        formatVersion: header.formatVersion,
        inputSize: header.inputSize,
        layers: header.layers,
        weights,
    };
    if (header.training !== undefined) artifact.training = header.training;
    if (header.metadata !== undefined) artifact.metadata = header.metadata;
    return validateArtifact(artifact);
}

let crcTable: Uint32Array | undefined;

/**
 * CRC-32 (IEEE 802.3 / zlib / PNG: reflected polynomial 0xEDB88320, init and final XOR
 * 0xFFFFFFFF). `crc32(utf8("123456789")) === 0xCBF43926`.
 *
 * @param bytes Data to checksum.
 * @returns The checksum as an unsigned 32-bit integer.
 */
export function crc32(bytes: Uint8Array): number {
    const table = (crcTable ??= buildCrcTable());
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) crc = table[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
}

function buildCrcTable(): Uint32Array {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        table[n] = c;
    }
    return table;
}

/**
 * @internal True when `bytes` starts with `0x89 "ORION"`: the binary signature, possibly with its
 * trailing CR LF mangled. Used for format sniffing so damaged files still reach
 * {@link decodeBinary}, which explains what went wrong.
 */
export function hasBinarySignature(bytes: Uint8Array): boolean {
    if (bytes.length < 6) return false;
    for (let i = 0; i < 6; i++) if (bytes[i] !== MAGIC[i]) return false;
    return true;
}

/** @internal Normalizes a Uint8Array / other ArrayBufferView / ArrayBuffer into a Uint8Array view (no copy). */
export function toBytes(data: unknown, errorPrefix: string): Uint8Array {
    if (data instanceof Uint8Array) return data;
    if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const tag = Object.prototype.toString.call(data);
    if (tag === "[object ArrayBuffer]" || tag === "[object SharedArrayBuffer]") {
        return new Uint8Array(data as ArrayBufferLike);
    }
    throw new SerializationError(`${errorPrefix}: expected a Uint8Array or ArrayBuffer, got ${describeValue(data)}`);
}

function checkMagic(bytes: Uint8Array): void {
    if (bytes.length === 0) throw binaryError("file is empty");
    const n = Math.min(bytes.length, MAGIC.length);
    let matches = true;
    for (let i = 0; i < n; i++) {
        if (bytes[i] !== MAGIC[i]) {
            matches = false;
            break;
        }
    }
    if (matches && n === MAGIC.length) return;
    if (matches) {
        throw binaryError(`file is truncated: ${bytes.length} bytes, but even an empty container needs ${MIN_FILE_BYTES}`);
    }
    const found = hexBytes(bytes.subarray(0, MAGIC.length));
    let signatureDamaged = bytes.length >= 6;
    for (let i = 1; i < 6 && signatureDamaged; i++) signatureDamaged = bytes[i] === MAGIC[i];
    if (signatureDamaged) {
        throw binaryError(
            `damaged signature (expected ${hexBytes(MAGIC)}, found ${found}); the file was probably ` +
            "transferred or saved in text mode (line-ending or 7-bit conversion). Transfer it as binary",
        );
    }
    throw binaryError(`not an Orion Engine binary model: expected magic bytes ${hexBytes(MAGIC)}, found ${found}`);
}

function readData(view: DataView, offset: number, length: number, dtype: BinaryPrecision): Float64Array {
    const out = new Float64Array(length);
    if (dtype === "float64") {
        for (let i = 0; i < length; i++, offset += 8) out[i] = view.getFloat64(offset, true);
    } else {
        for (let i = 0; i < length; i++, offset += 4) out[i] = view.getFloat32(offset, true);
    }
    return out;
}

function binaryError(message: string): SerializationError {
    return new SerializationError(`${ERROR_PREFIX}: ${message}`);
}

function align8(n: number): number {
    return Math.ceil(n / 8) * 8;
}

function hexBytes(bytes: Uint8Array): string {
    return Array.from(bytes, (b) => b.toString(16).toUpperCase().padStart(2, "0")).join(" ");
}

function hex16(n: number): string {
    return n.toString(16).toUpperCase().padStart(4, "0");
}

function hex32(n: number): string {
    return n.toString(16).toUpperCase().padStart(8, "0");
}
