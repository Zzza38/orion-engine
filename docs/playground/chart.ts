/**
 * Live loss chart (train vs test) on a DPR-aware canvas, with a log-scale option and a
 * crosshair tooltip on hover.
 */
import { prepareCanvas } from "./canvas.js";
import type { Palette } from "./theme.js";

export interface LossPoint {
    epoch: number;
    train: number;
    test: number;
}

/** Keeps the chart cheap to draw: older history is thinned once it exceeds this many points. */
const MAX_POINTS = 1200;
const MARGIN = { top: 10, right: 12, bottom: 22, left: 44 };

/** "Nice" tick values (1, 2, 5 × 10^n steps) covering [min, max]. */
export function niceTicks(min: number, max: number, target = 4): number[] {
    if (!(max > min)) return [min];
    const raw = (max - min) / target;
    const magnitude = 10 ** Math.floor(Math.log10(raw));
    const residual = raw / magnitude;
    const step = (residual > 5 ? 10 : residual > 2 ? 5 : residual > 1 ? 2 : 1) * magnitude;
    const ticks: number[] = [];
    const first = Math.ceil(min / step - 1e-9);
    for (let i = first; i * step <= max + step * 1e-9; i++) {
        // Round away float noise (0.6000000000000001) so labels and comparisons stay clean.
        ticks.push(Number((i * step).toPrecision(12)));
    }
    return ticks;
}

export function formatLoss(value: number): string {
    if (!Number.isFinite(value)) return "–";
    if (value === 0) return "0";
    const abs = Math.abs(value);
    if (abs >= 100) return value.toFixed(0);
    if (abs >= 1) return value.toFixed(3);
    if (abs >= 0.001) return value.toFixed(4);
    return value.toExponential(2);
}

function formatTick(value: number): string {
    if (value === 0) return "0";
    const abs = Math.abs(value);
    if (abs >= 1000) return `${(value / 1000).toFixed(abs >= 10000 ? 0 : 1)}k`;
    if (abs >= 1) return String(Math.round(value * 100) / 100);
    if (abs >= 0.001) return String(Number(value.toPrecision(2)));
    return value.toExponential(0);
}

export class LossChart {
    private readonly canvas: HTMLCanvasElement;
    private readonly tooltip: HTMLElement;
    private points: LossPoint[] = [];
    private hoverX: number | null = null;
    private logScale = false;
    private palette: Palette | null = null;

    constructor(canvas: HTMLCanvasElement, tooltip: HTMLElement) {
        this.canvas = canvas;
        this.tooltip = tooltip;
        canvas.addEventListener("pointermove", (event) => {
            const rect = canvas.getBoundingClientRect();
            this.hoverX = event.clientX - rect.left;
            if (this.palette) this.render(this.palette);
        });
        canvas.addEventListener("pointerleave", () => {
            this.hoverX = null;
            if (this.palette) this.render(this.palette);
        });
    }

    get length(): number {
        return this.points.length;
    }

    setLogScale(enabled: boolean): void {
        this.logScale = enabled;
    }

    clear(): void {
        this.points = [];
    }

    push(point: LossPoint): void {
        const last = this.points[this.points.length - 1];
        if (last && last.epoch === point.epoch) this.points[this.points.length - 1] = point;
        else this.points.push(point);
        if (this.points.length > MAX_POINTS) {
            // Thin the history by half, always keeping the first and most recent points.
            const lastPoint = this.points[this.points.length - 1];
            this.points = this.points.filter((_, i) => i % 2 === 0);
            if (this.points[this.points.length - 1] !== lastPoint) this.points.push(lastPoint);
        }
    }

    render(palette: Palette): void {
        this.palette = palette;
        const frame = prepareCanvas(this.canvas);
        if (!frame) return;
        const { ctx, width, height } = frame;
        ctx.clearRect(0, 0, width, height);
        const plotW = width - MARGIN.left - MARGIN.right;
        const plotH = height - MARGIN.top - MARGIN.bottom;
        ctx.font = "500 10.5px system-ui, -apple-system, 'Segoe UI', sans-serif";

        const values: number[] = [];
        for (const p of this.points) {
            if (Number.isFinite(p.train)) values.push(p.train);
            if (Number.isFinite(p.test)) values.push(p.test);
        }
        const positive = values.filter((v) => v > 0);
        const useLog = this.logScale && positive.length > 0;

        let yMin: number;
        let yMax: number;
        let yTicks: number[];
        if (useLog) {
            yMin = 10 ** Math.floor(Math.log10(Math.min(...positive)));
            yMax = 10 ** Math.ceil(Math.log10(Math.max(...positive)));
            if (yMax <= yMin) yMax = yMin * 10;
            yTicks = [];
            const decades = Math.log10(yMax / yMin);
            for (let v = yMin; v <= yMax * 1.0001; v *= 10) {
                yTicks.push(v);
                if (decades <= 2) {
                    if (2 * v < yMax) yTicks.push(2 * v);
                    if (5 * v < yMax) yTicks.push(5 * v);
                }
            }
        } else {
            yMin = 0;
            const top = values.length > 0 ? Math.max(...values) : 1;
            yTicks = niceTicks(0, top > 0 ? top : 1, Math.max(2, Math.min(5, Math.floor(plotH / 28))));
            yMax = Math.max(top * 1.05, yTicks[yTicks.length - 1]);
        }
        const lastEpoch = this.points.length > 0 ? this.points[this.points.length - 1].epoch : 0;
        const xMax = Math.max(10, lastEpoch);
        const xTicks = niceTicks(0, xMax, Math.max(2, Math.floor(plotW / 70)));

        const toX = (epoch: number) => MARGIN.left + (epoch / xMax) * plotW;
        const toY = useLog
            ? (v: number) =>
                  MARGIN.top +
                  plotH -
                  ((Math.log10(Math.max(v, yMin)) - Math.log10(yMin)) / Math.log10(yMax / yMin)) * plotH
            : (v: number) => MARGIN.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

        // Gridlines and tick labels.
        ctx.lineWidth = 1;
        ctx.strokeStyle = palette.grid;
        ctx.fillStyle = palette.muted;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        for (const t of yTicks) {
            const y = Math.round(toY(t)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(MARGIN.left, y);
            ctx.lineTo(width - MARGIN.right, y);
            ctx.stroke();
            ctx.fillText(formatTick(t), MARGIN.left - 8, y);
        }
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        for (const t of xTicks) {
            const x = toX(t);
            if (x > width - MARGIN.right + 1) continue;
            ctx.fillText(formatTick(t), Math.min(x, width - MARGIN.right - 6), height - MARGIN.bottom + 6);
        }
        ctx.strokeStyle = palette.axis;
        ctx.beginPath();
        ctx.moveTo(MARGIN.left, Math.round(MARGIN.top + plotH) + 0.5);
        ctx.lineTo(width - MARGIN.right, Math.round(MARGIN.top + plotH) + 0.5);
        ctx.stroke();

        if (this.points.length === 0) {
            ctx.fillStyle = palette.muted;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.font = "500 12px system-ui, -apple-system, 'Segoe UI', sans-serif";
            ctx.fillText("Press play to start training", MARGIN.left + plotW / 2, MARGIN.top + plotH / 2);
            this.tooltip.hidden = true;
            return;
        }

        // Series.
        ctx.save();
        ctx.beginPath();
        ctx.rect(MARGIN.left, MARGIN.top - 2, plotW + 2, plotH + 4);
        ctx.clip();
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.lineWidth = 2;
        for (const [key, color] of [
            ["train", palette.seriesTrain],
            ["test", palette.seriesTest],
        ] as const) {
            ctx.beginPath();
            let started = false;
            for (const p of this.points) {
                const v = p[key];
                if (!Number.isFinite(v)) {
                    started = false;
                    continue;
                }
                const x = toX(p.epoch);
                const y = toY(v);
                if (started) ctx.lineTo(x, y);
                else ctx.moveTo(x, y);
                started = true;
            }
            ctx.strokeStyle = color;
            ctx.stroke();
        }
        ctx.restore();

        this.drawHover(ctx, toX, toY, palette, width);
    }

    private drawHover(
        ctx: CanvasRenderingContext2D,
        toX: (epoch: number) => number,
        toY: (value: number) => number,
        palette: Palette,
        width: number,
    ): void {
        if (this.hoverX === null || this.points.length === 0) {
            this.tooltip.hidden = true;
            return;
        }
        // Nearest point by x.
        let best = this.points[0];
        for (const p of this.points) {
            if (Math.abs(toX(p.epoch) - this.hoverX) < Math.abs(toX(best.epoch) - this.hoverX)) best = p;
        }
        const x = toX(best.epoch);
        ctx.save();
        ctx.strokeStyle = palette.axis;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(Math.round(x) + 0.5, MARGIN.top);
        ctx.lineTo(Math.round(x) + 0.5, this.canvas.clientHeight - MARGIN.bottom);
        ctx.stroke();
        for (const [value, color] of [
            [best.train, palette.seriesTrain],
            [best.test, palette.seriesTest],
        ] as const) {
            if (!Number.isFinite(value)) continue;
            ctx.beginPath();
            ctx.arc(x, toY(value), 4, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.strokeStyle = palette.surface;
            ctx.lineWidth = 2;
            ctx.fill();
            ctx.stroke();
        }
        ctx.restore();

        const tip = this.tooltip;
        tip.replaceChildren();
        const head = document.createElement("div");
        head.className = "tt-epoch";
        head.textContent = `Epoch ${best.epoch.toLocaleString("en-US")}`;
        tip.append(head);
        for (const [label, value, cls] of [
            ["Train", best.train, "train"],
            ["Test", best.test, "test"],
        ] as const) {
            const row = document.createElement("div");
            row.className = "tt-row";
            const swatch = document.createElement("span");
            swatch.className = `swatch swatch-line ${cls}`;
            const b = document.createElement("b");
            b.textContent = formatLoss(value);
            row.append(swatch, label, b);
            tip.append(row);
        }
        tip.hidden = false;
        const tipWidth = tip.offsetWidth;
        const left = x + 12 + tipWidth > width ? x - 12 - tipWidth : x + 12;
        tip.style.left = `${Math.max(0, left)}px`;
    }
}
