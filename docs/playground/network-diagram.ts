/**
 * Network diagram: one column of nodes per layer, edges colored by weight sign and sized by
 * magnitude. Drawn on canvas so even 6 × 32-unit networks (thousands of edges) stay cheap.
 */
import { prepareCanvas } from "./canvas.js";
import type { LayerSpec, OutputSpec } from "./config.js";
import type { Palette, Rgb } from "./theme.js";
import { rgbCss } from "./theme.js";

export interface Kernel {
    /** [inputs, units] */
    shape: readonly [number, number] | readonly number[];
    data: ArrayLike<number>;
}

export interface Architecture {
    inputs: string[];
    hidden: LayerSpec[];
    output: OutputSpec;
    /** Output node labels, e.g. class names; defaults to none. */
    outputColors?: Rgb[];
}

/** Edge magnitude buckets per sign: edges are batched into one path per bucket. */
const BUCKETS = 8;
const PAD = { top: 30, bottom: 14, left: 50, right: 22 };

export class NetworkDiagram {
    private readonly canvas: HTMLCanvasElement;
    private arch: Architecture | null = null;
    private kernels: Kernel[] | null = null;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
    }

    setArchitecture(arch: Architecture): void {
        this.arch = arch;
        this.kernels = null;
        const hidden = arch.hidden.map((l) => `${l.units} ${l.activation}`).join(", ");
        this.canvas.setAttribute(
            "aria-label",
            `Network diagram: ${arch.inputs.length} inputs (${arch.inputs.join(", ")}), hidden layers ${hidden}, ` +
                `${arch.output.units} ${arch.output.activation} output${arch.output.units > 1 ? "s" : ""}.`,
        );
    }

    setKernels(kernels: Kernel[] | null): void {
        this.kernels = kernels;
    }

    render(palette: Palette): void {
        const frame = prepareCanvas(this.canvas);
        if (!frame || !this.arch) return;
        const { ctx, width, height } = frame;
        ctx.clearRect(0, 0, width, height);

        const arch = this.arch;
        const sizes = [arch.inputs.length, ...arch.hidden.map((l) => l.units), arch.output.units];
        const columns = sizes.length;
        const innerW = width - PAD.left - PAD.right;
        const innerH = height - PAD.top - PAD.bottom;
        const colX = (i: number) => PAD.left + (columns === 1 ? 0 : (i / (columns - 1)) * innerW);
        const maxNodes = Math.max(...sizes);
        const spacing = Math.min(30, innerH / maxNodes);
        const radius = Math.max(1.8, Math.min(8, spacing * 0.33));
        const nodeY = (count: number, j: number) => PAD.top + innerH / 2 + (j - (count - 1) / 2) * spacing;

        // Edges.
        for (let layer = 0; layer < columns - 1; layer++) {
            const inCount = sizes[layer];
            const outCount = sizes[layer + 1];
            const kernel = this.kernels?.[layer];
            const valid = kernel && kernel.shape[0] === inCount && kernel.shape[1] === outCount;
            const x0 = colX(layer) + radius;
            const x1 = colX(layer + 1) - radius;
            if (!valid) {
                ctx.beginPath();
                for (let i = 0; i < inCount; i++) {
                    for (let j = 0; j < outCount; j++) {
                        ctx.moveTo(x0, nodeY(inCount, i));
                        ctx.lineTo(x1, nodeY(outCount, j));
                    }
                }
                ctx.strokeStyle = palette.weightNone;
                ctx.globalAlpha = 0.7;
                ctx.lineWidth = 0.75;
                ctx.stroke();
                ctx.globalAlpha = 1;
                continue;
            }
            const data = kernel.data;
            let maxAbs = 1e-12;
            for (let k = 0; k < data.length; k++) maxAbs = Math.max(maxAbs, Math.abs(data[k]));
            // bucket index → path; negative weights use buckets [0, BUCKETS), positive [BUCKETS, 2·BUCKETS).
            const paths: Path2D[] = Array.from({ length: BUCKETS * 2 }, () => new Path2D());
            for (let i = 0; i < inCount; i++) {
                const y0 = nodeY(inCount, i);
                for (let j = 0; j < outCount; j++) {
                    const w = data[i * outCount + j];
                    const t = Math.abs(w) / maxAbs;
                    const bucket = Math.min(BUCKETS - 1, Math.floor(t * BUCKETS));
                    const path = paths[(w >= 0 ? BUCKETS : 0) + bucket];
                    path.moveTo(x0, y0);
                    path.lineTo(x1, nodeY(outCount, j));
                }
            }
            const density = Math.min(1, 12 / Math.sqrt(inCount * outCount));
            for (let b = 0; b < BUCKETS * 2; b++) {
                const positive = b >= BUCKETS;
                const t = ((b % BUCKETS) + 0.5) / BUCKETS;
                ctx.strokeStyle = rgbCss(positive ? palette.weightPos : palette.weightNeg);
                ctx.globalAlpha = (0.1 + 0.85 * t) * (0.55 + 0.45 * density);
                ctx.lineWidth = 0.35 + 3.1 * t * (0.5 + 0.5 * density);
                ctx.stroke(paths[b]);
            }
            ctx.globalAlpha = 1;
        }

        // Nodes.
        ctx.lineWidth = Math.min(1.5, radius / 3);
        for (let layer = 0; layer < columns; layer++) {
            const count = sizes[layer];
            const x = colX(layer);
            const isOutput = layer === columns - 1;
            for (let j = 0; j < count; j++) {
                ctx.beginPath();
                ctx.arc(x, nodeY(count, j), radius, 0, Math.PI * 2);
                const color = isOutput ? arch.outputColors?.[j] : undefined;
                ctx.fillStyle = color ? rgbCss(color) : palette.node;
                ctx.strokeStyle = color ? palette.surface : palette.nodeStroke;
                ctx.fill();
                ctx.stroke();
            }
        }

        // Input labels.
        ctx.font = "500 11px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
        ctx.fillStyle = palette.ink2;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        if (spacing >= 11) {
            for (let j = 0; j < arch.inputs.length; j++) {
                ctx.fillText(arch.inputs[j], colX(0) - radius - 7, nodeY(arch.inputs.length, j));
            }
        }

        // Column headers.
        const colGap = columns > 1 ? innerW / (columns - 1) : innerW;
        ctx.font = "600 10.5px system-ui, -apple-system, 'Segoe UI', sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "alphabetic";
        for (let layer = 0; layer < columns; layer++) {
            let text: string;
            if (layer === 0) text = "Input";
            else if (layer === columns - 1) text = "Out";
            else {
                const spec = arch.hidden[layer - 1];
                const full = `${spec.units} ${spec.activation}`;
                text = ctx.measureText(full).width < colGap - 8 ? full : `H${layer}`;
            }
            ctx.fillStyle = palette.muted;
            const x = Math.max(
                ctx.measureText(text).width / 2 + 2,
                Math.min(width - ctx.measureText(text).width / 2 - 2, colX(layer)),
            );
            ctx.fillText(text, x, 16);
        }
    }
}
