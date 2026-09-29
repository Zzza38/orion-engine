# Model file formats

Orion Engine saves models in two formats and can import one legacy format. All three decode to
the same in-memory `ModelArtifact` (see `src/core/types.ts`); the API lives in `src/io`.

| Format | Recommended extension | Encoder | Decoder | Use it for |
|---|---|---|---|---|
| Binary container | `.onn` | `encodeBinary` | `decodeBinary` | Shipping and loading models: compact, fast, checksummed |
| JSON | `.onn.json` | `encodeJson` | `decodeJson` | Inspecting, diffing and hand-editing models; bit-exact |
| Legacy text | `.onn` (orion-engine ≤ 0.0.2) | none | `decodeLegacyOnn` | Importing old models |

`decodeArtifact(data)` detects the format automatically and `encodeArtifact(artifact, { format })`
dispatches to the right encoder:

```ts
import { decodeArtifact, encodeArtifact } from "@zzza38/orion-engine";

const bytes = encodeArtifact(artifact, { format: "binary" });              // Uint8Array
const text = encodeArtifact(artifact, { format: "json", pretty: true });   // string
const again = decodeArtifact(bytes);                                       // any format in, artifact out
```

### Format detection

`detectFormat(data)` looks at the first bytes only:

1. Bytes starting with `89 4F 52 49 4F 4E` (`0x89` `ORION`) are **binary**. Only six bytes are
   compared, so a file whose line-ending bytes were damaged still reaches the binary decoder,
   which then explains the damage.
2. Otherwise, after skipping an optional UTF-8 byte-order mark and whitespace (space, tab, CR, LF):
   `{` means **JSON**, and an ASCII digit means **legacy**.
3. Anything else, including empty input, throws `SerializationError`. A string that starts
   with `\u0089ORION` is a binary file that was decoded as text, and the error says so.

Text formats may be passed as strings or as UTF-8 bytes. The binary format must be passed as
bytes (`Uint8Array` or `ArrayBuffer`).

### Errors

Every decoder throws `SerializationError` for malformed input. The message names the exact field
or position, what was expected, and what was found, for example
`Invalid model artifact: weights[1] ("dense_1/bias").data: expected 4 values for shape [1, 4], got 3`.
Encoders validate their input the same way before writing anything.

---

## The artifact (shared by every format)

| Key | Type | Required | Meaning |
|---|---|---|---|
| `format` | `"orion-engine"` | yes | Format marker |
| `formatVersion` | `1` | yes | Artifact schema version (see [Versioning](#versioning)) |
| `inputSize` | positive integer | yes | Number of input features |
| `layers` | array of layer configs | yes | Each is a JSON object with a non-empty `type` and a unique, non-empty `name` |
| `training` | object | no | `{ loss: {name, …}, optimizer: {name, …}, metrics: string[] }` |
| `metadata` | JSON object | no | Free-form, user-defined |
| `weights` | array of weight entries | yes | In `model.parameters()` order |

A **weight entry** is `{ name, shape: [rows, cols], data }`:

- `name` is unique and non-empty, and follows the `"<layerName>/<param>"` convention, for example
  `dense_1/kernel` or `dense_1/bias`.
- `shape` holds two positive integers. Dense kernels are `[inputSize, units]` and biases are `[1, units]`.
- `data` holds exactly `rows × cols` finite numbers in **row-major** order, so element `[r][c]`
  is at `data[r * cols + c]`.

Layer configs, `training`, and `metadata` must be JSON-representable: plain objects, arrays,
strings, booleans, `null`, and **finite** numbers. Object properties whose value is `undefined`
are dropped.

A dense layer config looks like this:

```json
{
  "type": "dense", "name": "dense_1", "units": 4,
  "activation": { "name": "relu" }, "useBias": true,
  "kernelInitializer": { "name": "glorotUniform" }, "biasInitializer": { "name": "zeros" }
}
```

Decoders always return weight `data` as `Float64Array`s, which can back a `Matrix` without a copy.
Encoders accept `number[]`, `Float64Array`, or `Float32Array`.

---

## JSON format (`.onn.json`)

A single JSON object holding the artifact, with each weight's `data` written as a flat number array.

```json
{
  "format": "orion-engine",
  "formatVersion": 1,
  "inputSize": 2,
  "layers": [
    {
      "type": "dense",
      "name": "dense_1",
      "units": 2,
      "activation": {
        "name": "swish"
      },
      "useBias": true,
      "kernelInitializer": {
        "name": "glorotUniform"
      },
      "biasInitializer": {
        "name": "zeros"
      }
    }
  ],
  "weights": [
    {
      "name": "dense_1/kernel",
      "shape": [2, 2],
      "data": [
        0.71, -1.82,
        -0.2, 0.95
      ]
    },
    {
      "name": "dense_1/bias",
      "shape": [1, 2],
      "data": [0.19, 0.97]
    }
  ]
}
```

This is the exact output of `encodeJson(artifact, { pretty: true })` for the converted legacy
example (with its `metadata` removed).

- **Precision.** Each number is written as the shortest decimal that parses back to the same
  float64, which is what `Number#toString` produces. Decoding is therefore **bit-exact**.
  Negative zero is written as `-0`, which is valid JSON, so its sign survives. `Float32Array`
  data is widened to float64 exactly: `Math.fround(0.1)` is written as `0.10000000149011612`.
- **Key order.** Keys are written in this order: `format`, `formatVersion`, `inputSize`, `layers`,
  `training`, `metadata`, `weights`. Weights come last because they are the bulk of the file.
  Readers must not depend on key order.
- **Layout.** By default (`{ pretty: false }`) the whole document is one line with no insignificant
  whitespace. With `{ pretty: true }` it is indented by 2 spaces, a `[1, n]` weight is written
  on one line, and larger matrices are written one row per line. Both layouts decode identically.
- **Tolerance.** Decoders ignore a leading byte-order mark and unknown top-level keys.

---

## Binary container (`.onn`)

All multi-byte integers and floats are **little-endian**, whatever the host's byte order.

| Offset | Size (bytes) | Type | Field |
|---:|---:|---|---|
| 0 | 8 | bytes | Magic: `89 4F 52 49 4F 4E 0D 0A` (`\x89` `ORION` `\r\n`) |
| 8 | 2 | u16 | Container version, currently `1` |
| 10 | 2 | u16 | Flags: reserved, must be `0` |
| 12 | 4 | u32 | `H`, the byte length of the header |
| 16 | `H` | UTF-8 | Header: a JSON object (below) |
| 16 + `H` | `P` | zero bytes | Padding, 0–7 bytes, so that `D = 16 + H + P` is a multiple of 8 |
| `D` | `L` | bytes | Data section; `L` is `header.dataByteLength`, a multiple of 8 |
| `D + L` | 4 | u32 | CRC-32 of bytes `[0, D + L)` |

The file is exactly `D + L + 4` bytes long. Readers reject files that are shorter (truncated) or
longer (trailing bytes).

### Magic

The magic follows the PNG signature design:

- `0x89` has the high bit set, so a 7-bit channel that strips the high bit is detected. It is
  also never the first byte of JSON (`{` or whitespace) or of a legacy file (a digit), so
  format sniffing is unambiguous.
- `ORION` identifies the file to people who open it in a hex dump.
- `\r\n` detects text-mode transfers that convert CRLF to LF or LF to CRLF.

### Header

The header is the artifact without weight data. Weights are listed as descriptors, and one extra
key gives the size of the data section:

```json
{
  "format": "orion-engine", "formatVersion": 1, "inputSize": 2,
  "layers": [ … ],
  "training": { … },
  "metadata": { … },
  "weights": [
    { "name": "dense_1/kernel", "shape": [2, 2], "dtype": "float32", "byteOffset": 0, "length": 4 },
    { "name": "dense_1/bias",   "shape": [1, 2], "dtype": "float32", "byteOffset": 16, "length": 2 }
  ],
  "dataByteLength": 24
}
```

| Descriptor key | Meaning |
|---|---|
| `name`, `shape` | As in the artifact |
| `dtype` | `"float32"` (IEEE 754 binary32, 4 bytes) or `"float64"` (binary64, 8 bytes) |
| `byteOffset` | Start of this tensor, **relative to the data section start `D`** |
| `length` | Number of elements; must equal `shape[0] × shape[1]` |

Each tensor occupies bytes `[D + byteOffset, D + byteOffset + length × sizeof(dtype))`, stored
row-major. That range must lie inside the data section.

### Data section (as written by `encodeBinary`)

- Tensors appear in artifact order. Each starts at a multiple of 8 bytes from `D`, with zero
  bytes between tensors, and the section is zero-padded to a multiple of 8. `D` is itself a
  multiple of 8 from the start of the file. A reader holding an 8-byte-aligned buffer on a
  little-endian host could therefore view every tensor in place as a typed array. The reference
  decoder does not rely on this: it reads through `DataView`.
- All tensors in a file share one dtype, chosen with `precision`:
  - `"float32"` (the default) halves the file size and keeps about 7 significant digits.
    Encoding fails with `SerializationError` if a value is outside the float32 range
    (±3.4028234663852886e38).
  - `"float64"` is bit-exact.
- Readers accept any in-bounds offsets and a different dtype for each tensor. Only writers are
  held to the layout above.

### CRC-32

The checksum is the standard CRC-32 used by zlib, PNG and Ethernet:

| Parameter | Value |
|---|---|
| Polynomial | `0x04C11DB7`, reflected (`0xEDB88320`) |
| Initial value | `0xFFFFFFFF` |
| Input and output reflected | yes |
| Final XOR | `0xFFFFFFFF` |
| Check value for `"123456789"` | `0xCBF43926` |

It covers every byte before it: the magic, the fixed fields, the header, the padding and the data.
It is stored as a little-endian u32.

### Reader algorithm

1. Check the magic. If only bytes 1–5 (`ORION`) match, report text-mode or 7-bit damage.
2. Check that the container version is ≤ 1 (a higher version means the library needs upgrading)
   and that the flags are `0`.
3. Check that `16 + H + 4` does not exceed the file length.
4. Read `dataByteLength` from the header if the header parses, and compare `D + L + 4` with the
   file length. This step only produces precise truncation and trailing-byte errors; the header
   is not trusted yet.
5. Verify the CRC-32.
6. Parse and validate the header and each descriptor, read the tensors, and validate the
   resulting artifact.

### Implementation notes

The codec uses only `DataView`, `TextEncoder`, `TextDecoder` and typed arrays, so it runs
unchanged in browsers. `Uint8Array` inputs may be views at any `byteOffset`.

---

## Versioning

There are two independent version numbers:

| Number | Where it appears | Changes when |
|---|---|---|
| Container version (u16 at offset 8) | Binary files only | The byte layout of the container changes |
| `formatVersion` | JSON documents and the binary header | The artifact schema changes incompatibly, for example a field is renamed or re-interpreted |

The policy for both:

- Readers reject versions newer than they support and name the version in the error, with a hint
  to upgrade. They never guess.
- **Additive** changes do not bump either number: for example, a new optional top-level key or a
  new layer `type`. Readers ignore top-level keys they do not know. An unknown layer `type` is
  the model loader's concern, not the file format's.
- New binary features that change how bytes are interpreted must use a new container version or
  a flag bit. Today's readers reject any non-zero flag.
- Writers always write the newest version they support. Newer library releases keep reading every
  older version.

---

## Legacy `.onn` text format (import only)

orion-engine 0.0.2 and earlier wrote two lines of text:

```text
2:linear:4:relu:1:sigmoid
w:w:b|w:w:b|w:w:b|w:w:b|w:w:w:w:b
```

- **Line 1** is a list of `size:activation` pairs. The **first pair is the input layer**; its
  activation was never applied and is ignored, though it must still be a known name.
- **Line 2** lists the neurons of layers 1…n in order, separated by `|`. Each neuron is its
  incoming weights (one per neuron of the previous layer) followed by its bias, separated by `:`.

### Conversion

`decodeLegacyOnn(text)` converts layer *k* (for *k* ≥ 1) of size `units`, with a previous layer of
size `fanIn`, into:

- a layer config `{ type: "dense", name: "dense_k", units, activation, useBias: true,
  kernelInitializer: { name: "glorotUniform" }, biasInitializer: { name: "zeros" } }`;
- `dense_k/kernel` with shape `[fanIn, units]`. **Weight i of neuron j connects previous neuron i to
  neuron j, so it is stored at `kernel[i][j]`**: the old per-neuron rows are transposed into columns;
- `dense_k/bias` with shape `[1, units]`, where `bias[0][j]` is neuron j's bias.

The artifact's `inputSize` is the size of the first pair. Its `metadata` is
`{ "convertedFrom": "legacy-onn" }`, and it has no `training` config.

Activation names map one to one: `linear`, `sigmoid`, `tanh`, `relu`, `leakyRelu`, `elu`,
`softmax` and `swish`. `leakyRelu` and `elu` get the legacy engine's slopes explicitly
(`{ "name": "leakyRelu", "alpha": 0.01 }` and `{ "name": "elu", "alpha": 1 }`), so the converted
model computes exactly what the old engine computed.

For example, the documented legacy example

```text
2:relu:2:swish
0.71:-0.2:0.19|-1.82:0.95:0.97
```

converts to a 2-input model with one `swish` dense layer of 2 units, where
`kernel = [[0.71, -1.82], [-0.2, 0.95]]` and `bias = [[0.19, 0.97]]`.

### What the importer accepts

The importer **tolerates**:

- a leading byte-order mark;
- whitespace around lines and fields;
- CRLF line endings;
- trailing newlines;
- trailing `|` separators.

It **rejects** the following, each with a message naming the line, layer, neuron and value:

- anything other than exactly two lines;
- an odd number of fields on line 1, or fewer than two layers;
- a size that is not a positive integer, or an unknown activation;
- an empty neuron in the middle of line 2;
- a neuron with the wrong number of values;
- too few or too many neurons for the declared layer sizes;
- a value that is not a plain decimal number (hex, `NaN` and `Infinity` are rejected), or that
  overflows float64.

To upgrade an old file, convert it and write it back in a current format:

```ts
import { readFileSync, writeFileSync } from "node:fs";
import { decodeLegacyOnn, encodeBinary } from "@zzza38/orion-engine";

const artifact = decodeLegacyOnn(readFileSync("model.onn", "utf8"));
writeFileSync("model.onn", encodeBinary(artifact, { precision: "float64" }));
```
