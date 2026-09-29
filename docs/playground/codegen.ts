/**
 * Emits a TypeScript snippet that rebuilds the current playground model with the public API.
 */
import type { LayerSpec, PlaygroundConfig } from "./config.js";
import { datasetInfo, FEATURES, outputSpec, taskKind } from "./config.js";

const PACKAGE = "@zzza38/orion-engine";

function formatNumber(value: number): string {
    // Avoid exponent notation (1e-5) in generated code; it is valid but reads poorly.
    if (value !== 0 && Math.abs(value) < 0.001) return value.toFixed(10).replace(/0+$/, "");
    return String(value);
}

function denseCall(units: number, activation: string, l2: number): string {
    if (l2 > 0) {
        return `dense(${units}, { activation: "${activation}", kernelRegularizer: { l2: ${formatNumber(l2)} } })`;
    }
    return `dense(${units}, "${activation}")`;
}

function layerLines(layers: readonly LayerSpec[], config: PlaygroundConfig): string[] {
    const lines: string[] = [];
    for (const layer of layers) {
        lines.push(`${denseCall(layer.units, layer.activation, config.l2)},`);
        if (layer.dropout > 0) lines.push(`dropout(${formatNumber(layer.dropout)}),`);
    }
    const output = outputSpec(config);
    lines.push(`${denseCall(output.units, output.activation, config.l2)},`);
    return lines;
}

export interface CodegenOptions {
    /** Epoch count for the generated `fit` call; defaults to 200. */
    epochs?: number;
}

export function generateCode(config: PlaygroundConfig, options: CodegenOptions = {}): string {
    const output = outputSpec(config);
    const task = taskKind(config);
    const usesDropout = config.layers.some((l) => l.dropout > 0);
    const epochs = Math.max(1, Math.round(options.epochs ?? 200));
    const featureLabels = config.features.map((id) => FEATURES.find((f) => f.id === id)?.label ?? id);
    const dataset = datasetInfo(config.dataset);

    const imports = ["Sequential", "dense", ...(usesDropout ? ["dropout"] : [])];
    const targetShape =
        task === "binary"
            ? "[n, 1] with 0/1 labels"
            : task === "multiclass"
              ? `[n, 1] with class indices 0–${output.units - 1}`
              : "[n, 1] with real-valued targets";
    const metrics = task === "regression" ? `["mae"]` : `["accuracy"]`;

    const optimizer = `{ name: "${config.optimizer}", learningRate: ${formatNumber(config.learningRate)} }`;

    return [
        `import { ${imports.join(", ")} } from "${PACKAGE}";`,
        "",
        `// Orion Engine Playground: ${dataset.label} dataset, inputs [${featureLabels.join(", ")}].`,
        `// x: number[][] of shape [n, ${config.features.length}]; y: number[][] of shape ${targetShape}.`,
        "declare const x: number[][];",
        "declare const y: number[][];",
        "",
        "const model = new Sequential({",
        `    inputSize: ${config.features.length},`,
        `    seed: ${config.seed},`,
        "    layers: [",
        ...layerLines(config.layers, config).map((line) => `        ${line}`),
        "    ],",
        "});",
        "",
        "model.compile({",
        `    loss: "${output.loss}",`,
        `    optimizer: ${optimizer},`,
        `    metrics: ${metrics},`,
        "});",
        "",
        `const history = model.fit(x, y, { epochs: ${epochs}, batchSize: ${config.batchSize}, shuffle: true });`,
        `console.log("final loss", history.last("loss"));`,
        "",
    ].join("\n");
}
