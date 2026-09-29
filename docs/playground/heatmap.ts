/**
 * Decision-boundary view: a coarse grid of model predictions rendered as a smoothly upscaled
 * heatmap, a crisp boundary contour, and the train/test points on top.
 */
import { prepareCanvas } from "./canvas.js";
import type { DatasetId, TaskKind } from "./config.js";
import type { Point } from "./datasets.js";
import { DOMAIN, generatePoints, REGRESSION_TARGETS } from "./datasets.js";
import type { Palette, Rgb } from "./theme.js";

export interface BoundaryOptions {
    task: TaskKind;
    classes: number;
    showTest: boolean;
    discretize: boolean;
}

function mix(a: Rgb, b: Rgb, t: number): Rgb {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/** Heatmap color of one prediction row. */
function cellColor(
    predictions: ArrayLike<number>,
    index: number,
    outputs: number,
    options: BoundaryOptions,
    palette: Palette,
): Rgb {
    const { neutral, classes, mix: maxMix } = palette;
    if (options.task === "multiclass") {
        const base = index * outputs;
        let best = 0;
        let r = 0;
        let g = 0;
        let b = 0;
        for (let c = 0; c < outputs; c++) {
            const p = predictions[base + c];
            if (p > predictions[base + best]) best = c;
            const color = classes[c % classes.length];
            r += p * color[0];
            g += p * color[1];
            b += p * color[2];
        }
        if (options.discretize) return mix(neutral, classes[best % classes.length], maxMix);
        const floor = 1 / outputs;
        const confidence = Math.max(0, (predictions[base + best] - floor) / (1 - floor));
        return mix(neutral, [r, g, b], confidence * maxMix);
    }
    const raw = predictions[index];
    // Map to a signed strength in [-1, 1]: probability for binary, value for regression.
    let t = options.task === "binary" ? raw * 2 - 1 : Math.max(-1, Math.min(1, raw));
    if (!Number.isFinite(t)) t = 0;
    if (options.discretize) t = options.task === "binary" ? Math.sign(t) : Math.round(t * 4) / 4;
    const target = t < 0 ? classes[0] : classes[1];
    return mix(neutral, target, Math.abs(t) ** 0.85 * maxMix);
}

/** Color of a data point: its class, or its target value on the diverging scale. */
function pointColor(point: Point, task: TaskKind, palette: Palette): Rgb {
    if (task !== "regression") return palette.classes[point.label % palette.classes.length];
    const t = Math.max(-1, Math.min(1, point.label));
    return mix(palette.neutral, t < 0 ? palette.classes[0] : palette.classes[1], 0.3 + 0.7 * Math.abs(t));
}

function css([r, g, b]: Rgb): string {
    return `rgb(${Math.round(r)} ${Math.round(g)} ${Math.round(b)})`;
}

/**
 * Marching squares: line segments where `field` crosses `level`. The field is a size×size grid of
 * samples at cell centres; output coordinates are in grid units ([0, size], sample i at i + 0.5).
 */
export function contourSegments(field: ArrayLike<number>, size: number, level: number): number[] {
    const out: number[] = [];
    const lerp = (a: number, b: number) => {
        const d = b - a;
        return d === 0 ? 0.5 : Math.max(0, Math.min(1, (level - a) / d));
    };
    for (let r = 0; r < size - 1; r++) {
        for (let c = 0; c < size - 1; c++) {
            const tl = field[r * size + c];
            const tr = field[r * size + c + 1];
            const br = field[(r + 1) * size + c + 1];
            const bl = field[(r + 1) * size + c];
            const code = (tl > level ? 8 : 0) | (tr > level ? 4 : 0) | (br > level ? 2 : 0) | (bl > level ? 1 : 0);
            if (code === 0 || code === 15) continue;
            const x = c + 0.5;
            const y = r + 0.5;
            // Edge crossing points: top, right, bottom, left.
            const top = [x + lerp(tl, tr), y];
            const right = [x + 1, y + lerp(tr, br)];
            const bottom = [x + lerp(bl, br), y + 1];
            const left = [x, y + lerp(tl, bl)];
            const seg = (a: number[], b: number[]) => out.push(a[0], a[1], b[0], b[1]);
            switch (code) {
                case 1:
                case 14:
                    seg(left, bottom);
                    break;
                case 2:
                case 13:
                    seg(bottom, right);
                    break;
                case 3:
                case 12:
                    seg(left, right);
                    break;
                case 4:
                case 11:
                    seg(top, right);
                    break;
                case 6:
                case 9:
                    seg(top, bottom);
                    break;
                case 7:
                case 8:
                    seg(left, top);
                    break;
                case 5:
                    seg(left, top);
                    seg(bottom, right);
                    break;
                case 10:
                    seg(top, right);
                    seg(left, bottom);
                    break;
            }
        }
    }
    return out;
}

export class BoundaryView {
    private readonly canvas: HTMLCanvasElement;
    private readonly grid: HTMLCanvasElement;
    private gridSize = 0;
    private outputs = 1;
    private predictions: Float64Array | null = null;
    private contours: number[] = [];
    private train: Point[] = [];
    private test: Point[] = [];
    private options: BoundaryOptions = { task: "binary", classes: 2, showTest: true, discretize: false };

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.grid = document.createElement("canvas");
    }

    setPoints(train: Point[], test: Point[]): void {
        this.train = train;
        this.test = test;
    }

    setOptions(options: BoundaryOptions): void {
        this.options = options;
    }

    /** Clears the heatmap (e.g. after the model was rebuilt and before its first prediction). */
    clearPredictions(): void {
        this.predictions = null;
        this.contours = [];
    }

    /** Stores a row-major [gridSize², outputs] prediction grid (row 0 = top of the plot). */
    setPredictions(predictions: Float64Array, gridSize: number, outputs: number): void {
        this.predictions = predictions;
        this.gridSize = gridSize;
        this.outputs = outputs;
    }

    render(palette: Palette): void {
        const frame = prepareCanvas(this.canvas);
        if (!frame) return;
        const { ctx, width, height } = frame;
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = css(palette.neutral);
        ctx.fillRect(0, 0, width, height);

        if (this.predictions) {
            this.paintGrid(palette);
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = "high";
            ctx.drawImage(this.grid, 0, 0, width, height);
            this.drawContours(ctx, width, height, palette);
        }
        this.drawAxes(ctx, width, height, palette);
        this.drawPoints(ctx, width, height, palette);
    }

    private paintGrid(palette: Palette): void {
        const size = this.gridSize;
        const predictions = this.predictions!;
        if (this.grid.width !== size) {
            this.grid.width = size;
            this.grid.height = size;
        }
        const gctx = this.grid.getContext("2d");
        if (!gctx) return;
        const image = gctx.createImageData(size, size);
        const pixels = image.data;
        for (let i = 0; i < size * size; i++) {
            const [r, g, b] = cellColor(predictions, i, this.outputs, this.options, palette);
            pixels[i * 4] = r;
            pixels[i * 4 + 1] = g;
            pixels[i * 4 + 2] = b;
            pixels[i * 4 + 3] = 255;
        }
        gctx.putImageData(image, 0, 0);
        this.contours = this.computeContours();
    }

    private computeContours(): number[] {
        const size = this.gridSize;
        const predictions = this.predictions!;
        const { task } = this.options;
        if (task === "binary") return contourSegments(predictions, size, 0.5);
        if (task === "regression") return contourSegments(predictions, size, 0);
        // Multi-class: where each class's probability overtakes the best competitor.
        const k = this.outputs;
        const field = new Float64Array(size * size);
        const segments: number[] = [];
        for (let c = 0; c < k; c++) {
            for (let i = 0; i < size * size; i++) {
                let other = Number.NEGATIVE_INFINITY;
                for (let j = 0; j < k; j++) if (j !== c) other = Math.max(other, predictions[i * k + j]);
                field[i] = predictions[i * k + c] - other;
            }
            for (const v of contourSegments(field, size, 0)) segments.push(v);
        }
        return segments;
    }

    private drawContours(ctx: CanvasRenderingContext2D, width: number, height: number, palette: Palette): void {
        if (this.contours.length === 0) return;
        const sx = width / this.gridSize;
        const sy = height / this.gridSize;
        ctx.save();
        ctx.beginPath();
        const s = this.contours;
        for (let i = 0; i < s.length; i += 4) {
            ctx.moveTo(s[i] * sx, s[i + 1] * sy);
            ctx.lineTo(s[i + 2] * sx, s[i + 3] * sy);
        }
        ctx.strokeStyle = palette.ink;
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = 1.25;
        ctx.lineCap = "round";
        ctx.stroke();
        ctx.restore();
    }

    private drawAxes(ctx: CanvasRenderingContext2D, width: number, height: number, palette: Palette): void {
        ctx.save();
        ctx.font = "500 10px system-ui, -apple-system, 'Segoe UI', sans-serif";
        ctx.fillStyle = palette.ink2;
        ctx.globalAlpha = 0.7;
        const ticks = [-4, -2, 0, 2, 4];
        const toX = (v: number) => ((v + DOMAIN) / (2 * DOMAIN)) * width;
        const toY = (v: number) => ((DOMAIN - v) / (2 * DOMAIN)) * height;
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        for (const t of ticks) ctx.fillText(formatTick(t), toX(t), height - 4);
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        for (const t of ticks) if (t !== 0) ctx.fillText(formatTick(t), 5, toY(t));
        ctx.restore();
    }

    private drawPoints(ctx: CanvasRenderingContext2D, width: number, height: number, palette: Palette): void {
        const radius = Math.max(2.6, Math.min(4.2, width / 105));
        const toX = (x: number) => ((x + DOMAIN) / (2 * DOMAIN)) * width;
        const toY = (y: number) => ((DOMAIN - y) / (2 * DOMAIN)) * height;
        const { task } = this.options;
        ctx.save();
        ctx.lineWidth = 1.25;
        ctx.strokeStyle = palette.ring;
        for (const p of this.train) {
            ctx.beginPath();
            ctx.arc(toX(p.x), toY(p.y), radius, 0, Math.PI * 2);
            ctx.fillStyle = css(pointColor(p, task, palette));
            ctx.fill();
            ctx.stroke();
        }
        if (this.options.showTest) {
            ctx.lineWidth = 1.6;
            ctx.strokeStyle = palette.outline;
            for (const p of this.test) {
                ctx.beginPath();
                ctx.arc(toX(p.x), toY(p.y), radius, 0, Math.PI * 2);
                ctx.fillStyle = css(pointColor(p, task, palette));
                ctx.fill();
                ctx.stroke();
            }
        }
        ctx.restore();
    }
}

function formatTick(value: number): string {
    return value < 0 ? `−${-value}` : String(value);
}

/**
 * Renders a small preview of a dataset (used for the dataset picker tiles).
 * Regression previews show the target surface; classification previews show a sample.
 */
export function drawDatasetThumbnail(
    canvas: HTMLCanvasElement,
    dataset: DatasetId,
    classes: number,
    palette: Palette,
): void {
    const frame = prepareCanvas(canvas);
    if (!frame) return;
    const { ctx, width, height } = frame;
    ctx.fillStyle = css(palette.neutral);
    ctx.fillRect(0, 0, width, height);
    if (dataset === "plane" || dataset === "wave") {
        const size = 24;
        const grid = document.createElement("canvas");
        grid.width = size;
        grid.height = size;
        const gctx = grid.getContext("2d");
        if (!gctx) return;
        const image = gctx.createImageData(size, size);
        const fn = REGRESSION_TARGETS[dataset];
        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                const x = -DOMAIN + ((c + 0.5) / size) * 2 * DOMAIN;
                const y = DOMAIN - ((r + 0.5) / size) * 2 * DOMAIN;
                const v = Math.max(-1, Math.min(1, fn(x, y)));
                const [cr, cg, cb] = mix(
                    palette.neutral,
                    v < 0 ? palette.classes[0] : palette.classes[1],
                    Math.abs(v) * 0.85,
                );
                const i = (r * size + c) * 4;
                image.data[i] = cr;
                image.data[i + 1] = cg;
                image.data[i + 2] = cb;
                image.data[i + 3] = 255;
            }
        }
        gctx.putImageData(image, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(grid, 0, 0, width, height);
        return;
    }
    const points = generatePoints(dataset, 140, 0, 3, classes);
    const radius = Math.max(1.4, width / 42);
    for (const p of points) {
        ctx.beginPath();
        ctx.arc(
            ((p.x + DOMAIN) / (2 * DOMAIN)) * width,
            ((DOMAIN - p.y) / (2 * DOMAIN)) * height,
            radius,
            0,
            Math.PI * 2,
        );
        ctx.fillStyle = css(palette.classes[p.label % palette.classes.length]);
        ctx.fill();
    }
}
