/**
 * Orion Engine Playground: wires the controls, renderers and training loop together.
 */

import { observeResize } from "./canvas.js";
import { formatLoss, LossChart } from "./chart.js";
import { generateCode } from "./codegen.js";
import type { PlaygroundConfig } from "./config.js";
import { classCount, FEATURES, outputSpec, parameterCount, taskKind } from "./config.js";
import type { ChangeKind } from "./controls.js";
import { Controls } from "./controls.js";
import type { Dataset } from "./datasets.js";
import { accuracyOf, featureData, generateDataset, gridFeatureData, labelData, rSquaredOf } from "./datasets.js";
import { BoundaryView } from "./heatmap.js";
import { highlightTypeScript } from "./highlight.js";
import { NetworkDiagram } from "./network-diagram.js";
import type { Palette } from "./theme.js";
import { initTheme, readPalette } from "./theme.js";
import { ENGINE_VERSION, Matrix, Trainer } from "./trainer.js";
import { decodeConfig, encodeConfig } from "./url-state.js";

/** Per-frame time budget for training + inference, leaving headroom for layout and paint. */
const FRAME_BUDGET_MS = 12;
const DIAGRAM_INTERVAL_MS = 120;

/** Minimum spacing between runs of a step that costs `cost` ms, so it uses at most a third of the time. */
function throttle(cost: number): number {
    return cost > 4 ? cost * 2 : 0;
}

function byId<T extends HTMLElement>(id: string): T {
    const element = document.getElementById(id);
    if (!element) throw new Error(`Missing #${id}`);
    return element as T;
}

class Playground {
    private config: PlaygroundConfig;
    private palette: Palette;
    private readonly controls: Controls;
    private readonly boundary: BoundaryView;
    private readonly chart: LossChart;
    private readonly diagram: NetworkDiagram;

    private data!: Dataset;
    private xTrain!: Matrix;
    private yTrain!: Matrix;
    private xTest!: Matrix;
    private yTest!: Matrix;
    private grid!: Matrix;
    private gridSize = 64;

    private trainer: Trainer | null = null;
    private epoch = 0;
    private playing = false;
    private frameRequested = false;

    // Dirty flags + throttling state for the render pipeline.
    private predictionsDirty = true;
    private boundaryDirty = true;
    private chartDirty = true;
    private diagramDirty = true;
    private lastDiagramAt = 0;
    private lastPredictAt = 0;
    private predictCost = 0;
    private metricsDirty = false;
    private lastMetricsAt = 0;
    private metricsCost = 0;
    private renderCost = 0;
    /** Smoothed milliseconds per training epoch; decides between `fit` and batch slicing. */
    private epochCost = 0;
    private partialEpochMs = 0;
    private urlTimer = 0;
    private toastTimer = 0;

    private readonly playButton = byId<HTMLButtonElement>("play");
    private readonly stats = {
        epoch: byId("stat-epoch"),
        train: byId("stat-train"),
        test: byId("stat-test"),
        metric: byId("stat-metric"),
        metricLabel: byId("stat-metric-label"),
    };

    constructor() {
        this.config = decodeConfig(location.hash);
        this.palette = readPalette();
        this.boundary = new BoundaryView(byId<HTMLCanvasElement>("boundary-canvas"));
        this.chart = new LossChart(byId<HTMLCanvasElement>("loss-canvas"), byId("chart-tooltip"));
        this.diagram = new NetworkDiagram(byId<HTMLCanvasElement>("network-canvas"));
        this.controls = new Controls(this.config, (config, kind) => this.applyConfig(config, kind));

        byId("version").textContent = `v${ENGINE_VERSION}`;
        initTheme(byId<HTMLButtonElement>("theme-toggle"), () => this.onThemeChange());
        this.bindTransport();
        this.bindViewOptions();
        this.bindExport();
        this.bindKeyboard();

        observeResize(
            [byId("boundary-canvas"), byId("loss-canvas"), byId("network-canvas"), byId("dataset-tiles")],
            () => this.redrawAll(),
        );
        window.addEventListener("hashchange", () => this.onHashChange());

        this.rebuildData();
        this.rebuildModel();
        this.controls.renderThumbnails(this.palette);
        this.writeUrl();
    }

    // -----------------------------------------------------------------------------------------
    // Configuration changes
    // -----------------------------------------------------------------------------------------

    private applyConfig(config: PlaygroundConfig, kind: ChangeKind): void {
        this.config = config;
        if (kind === "data") this.rebuildData();
        if (kind === "features") this.rebuildInputs();
        if (kind !== "runtime") this.rebuildModel();
        this.scheduleUrlWrite();
    }

    private onHashChange(): void {
        const next = decodeConfig(location.hash);
        if (encodeConfig(next) === encodeConfig(this.config)) return;
        this.config = next;
        this.controls.setConfig(next);
        this.rebuildData();
        this.rebuildModel();
    }

    private scheduleUrlWrite(): void {
        clearTimeout(this.urlTimer);
        this.urlTimer = window.setTimeout(() => this.writeUrl(), 250);
    }

    private writeUrl(): void {
        const hash = `#${encodeConfig(this.config)}`;
        if (location.hash !== hash) history.replaceState(null, "", hash);
    }

    private rebuildData(): void {
        this.data = generateDataset(this.config);
        this.yTrain = new Matrix(this.data.train.length, 1, labelData(this.data.train));
        this.yTest = new Matrix(this.data.test.length, 1, labelData(this.data.test));
        this.boundary.setPoints(this.data.train, this.data.test);
        this.rebuildInputs();
    }

    /** Recomputes the feature matrices (data + heatmap grid) for the enabled input features. */
    private rebuildInputs(): void {
        const { features } = this.config;
        const cols = features.length;
        this.xTrain = new Matrix(this.data.train.length, cols, featureData(this.data.train, features));
        this.xTest = new Matrix(this.data.test.length, cols, featureData(this.data.test, features));
        this.grid = new Matrix(this.gridSize * this.gridSize, cols, gridFeatureData(this.gridSize, features));
    }

    private rebuildModel(): void {
        const config = this.config;
        const task = taskKind(config);
        this.epoch = 0;
        this.chart.clear();
        this.boundary.clearPredictions();
        this.boundary.setOptions(this.boundaryOptions());
        this.predictCost = 0;
        this.metricsCost = 0;
        this.epochCost = 0;
        this.partialEpochMs = 0;

        // Heavier networks get a coarser heatmap grid so inference stays inside the frame budget.
        const params = parameterCount(config);
        const size = params > 4000 ? 36 : params > 1500 ? 48 : params > 300 ? 64 : 88;
        if (size !== this.gridSize) {
            this.gridSize = size;
            this.grid = new Matrix(size * size, config.features.length, gridFeatureData(size, config.features));
        }

        const output = outputSpec(config);
        const colors = task === "multiclass" ? this.palette.classes.slice(0, output.units) : undefined;
        this.diagram.setArchitecture({
            inputs: config.features.map((id) => this.featureLabel(id)),
            hidden: config.layers,
            output,
            outputColors: colors,
        });
        this.renderModelInfo();
        this.renderLegend();
        this.updateCodePreview();

        try {
            this.trainer = new Trainer(config);
        } catch (error) {
            this.trainer = null;
            this.setPlaying(false);
            this.toast(`Could not build the model: ${(error as Error).message}`);
            console.error(error);
        }
        this.recordMetrics();
        this.predictionsDirty = true;
        this.diagramDirty = true;
        this.boundaryDirty = true;
        this.requestFrame();
    }

    private featureLabel(id: string): string {
        return FEATURES.find((f) => f.id === id)?.label ?? id;
    }

    private boundaryOptions() {
        return {
            task: taskKind(this.config),
            classes: classCount(this.config),
            showTest: byId<HTMLInputElement>("show-test").checked,
            discretize: byId<HTMLInputElement>("discretize").checked,
        };
    }

    // -----------------------------------------------------------------------------------------
    // Training loop
    // -----------------------------------------------------------------------------------------

    private setPlaying(playing: boolean): void {
        this.playing = playing && this.trainer !== null;
        this.playButton.setAttribute("aria-pressed", String(this.playing));
        this.playButton.setAttribute("aria-label", this.playing ? "Pause training (Space)" : "Start training (Space)");
        if (this.playing) this.requestFrame();
    }

    private requestFrame(): void {
        if (this.frameRequested) return;
        this.frameRequested = true;
        requestAnimationFrame((now) => this.frame(now));
    }

    private frame(now: number): void {
        this.frameRequested = false;
        let pending = false;
        if (this.playing && this.trainer) {
            // Leave room for the (throttled) evaluation and heatmap work that follows.
            const reserve =
                Math.min(this.metricsCost, 4) + Math.min(this.predictCost, 4) + Math.min(this.renderCost, 5);
            if (this.trainFor(Math.max(4, FRAME_BUDGET_MS - reserve))) {
                this.metricsDirty = true;
                this.predictionsDirty = true;
                this.diagramDirty = true;
            }
        }

        // Expensive steps are spread out while training so each takes at most ~a third of the time,
        // and never both in one frame.
        let heavyFrame = false;
        if (this.metricsDirty && this.trainer) {
            if (!this.playing || now - this.lastMetricsAt >= throttle(this.metricsCost)) {
                this.recordMetrics(now);
                heavyFrame = this.metricsCost > 4;
            } else {
                pending = true;
            }
        }
        if (this.predictionsDirty && this.trainer) {
            if (!this.playing || (!heavyFrame && now - this.lastPredictAt >= throttle(this.predictCost))) {
                this.updatePredictions(now);
            } else {
                pending = true;
            }
        }
        const renderStart = performance.now();
        if (this.boundaryDirty) {
            this.boundary.render(this.palette);
            this.boundaryDirty = false;
        }
        if (this.chartDirty) {
            this.chart.render(this.palette);
            this.chartDirty = false;
        }
        if (this.diagramDirty) {
            if (!this.playing || now - this.lastDiagramAt >= DIAGRAM_INTERVAL_MS) {
                this.diagram.setKernels(this.trainer ? this.trainer.kernels() : null);
                this.diagram.render(this.palette);
                this.diagramDirty = false;
                this.lastDiagramAt = now;
            } else {
                pending = true;
            }
        }
        const renderCost = performance.now() - renderStart;
        this.renderCost = this.renderCost * 0.8 + renderCost * 0.2;
        if (this.playing || pending) this.requestFrame();
    }

    /**
     * Trains for about `budgetMs`: whole epochs with `fit` while they are cheap (capped by the speed
     * setting), or mini-batch slices with `trainOnBatch` when a single epoch would blow the frame
     * budget. Returns true if any weights changed.
     */
    private trainFor(budgetMs: number, wholeEpoch = false): boolean {
        const trainer = this.trainer!;
        const start = performance.now();
        try {
            if (!wholeEpoch && this.epochCost > budgetMs * 0.75) {
                const done = trainer.trainUntil(this.xTrain, this.yTrain, this.config.batchSize, start + budgetMs);
                this.partialEpochMs += performance.now() - start;
                if (done) {
                    this.epoch++;
                    this.epochCost = this.partialEpochMs;
                    this.partialEpochMs = 0;
                }
                return true;
            }
            const maxEpochs = this.config.speed === "max" ? Number.POSITIVE_INFINITY : Number(this.config.speed);
            let trained = 0;
            do {
                const t0 = performance.now();
                trainer.fitEpoch(this.xTrain, this.yTrain, this.config.batchSize);
                const cost = performance.now() - t0;
                this.epochCost = this.epochCost === 0 ? cost : this.epochCost * 0.7 + cost * 0.3;
                this.epoch++;
                trained++;
            } while (!wholeEpoch && trained < maxEpochs && performance.now() - start + this.epochCost < budgetMs);
        } catch (error) {
            this.setPlaying(false);
            this.toast(`Training stopped: ${(error as Error).message}`);
            console.error(error);
        }
        return true;
    }

    private step(): void {
        if (!this.trainer) return;
        this.setPlaying(false);
        this.trainFor(0, true);
        this.recordMetrics();
        this.predictionsDirty = true;
        this.diagramDirty = true;
        this.requestFrame();
    }

    private reset(): void {
        this.rebuildModel();
    }

    /** Evaluates train/test loss and the task metric, then updates the stats and chart. */
    private recordMetrics(now = performance.now()): void {
        const trainer = this.trainer;
        const start = performance.now();
        this.metricsDirty = false;
        this.lastMetricsAt = now;
        if (!trainer) {
            this.renderStats(Number.NaN, Number.NaN, Number.NaN);
            return;
        }
        let train = Number.NaN;
        let test = Number.NaN;
        let metric = Number.NaN;
        try {
            train = trainer.evaluate(this.xTrain, this.yTrain).loss;
            test = trainer.evaluate(this.xTest, this.yTest).loss;
            if (this.data.test.length > 0) {
                const predictions = trainer.predict(this.xTest).data;
                metric =
                    taskKind(this.config) === "regression"
                        ? rSquaredOf(predictions, this.data.test)
                        : accuracyOf(predictions, trainer.outputs, this.data.test);
            }
        } catch (error) {
            console.error(error);
        }
        this.chart.push({ epoch: this.epoch, train, test });
        this.chartDirty = true;
        this.renderStats(train, test, metric);
        const cost = performance.now() - start;
        this.metricsCost = this.metricsCost === 0 ? cost : this.metricsCost * 0.8 + cost * 0.2;
    }

    private updatePredictions(now: number): void {
        const trainer = this.trainer!;
        const start = performance.now();
        try {
            const out = trainer.predict(this.grid);
            this.boundary.setPredictions(out.data, this.gridSize, out.cols);
        } catch (error) {
            console.error(error);
        }
        const cost = performance.now() - start;
        this.predictCost = this.predictCost === 0 ? cost : this.predictCost * 0.8 + cost * 0.2;
        this.lastPredictAt = now;
        this.predictionsDirty = false;
        this.boundaryDirty = true;
    }

    // -----------------------------------------------------------------------------------------
    // Rendering helpers
    // -----------------------------------------------------------------------------------------

    private renderStats(train: number, test: number, metric: number): void {
        const epoch = String(this.epoch).padStart(6, "0");
        this.stats.epoch.textContent = `${epoch.slice(0, -3)},${epoch.slice(-3)}`;
        this.stats.train.textContent = formatLoss(train);
        this.stats.test.textContent = formatLoss(test);
        const regression = taskKind(this.config) === "regression";
        this.stats.metricLabel.textContent = regression ? "Test R²" : "Test accuracy";
        this.stats.metric.textContent = !Number.isFinite(metric)
            ? "–"
            : regression
              ? metric.toFixed(3)
              : `${(metric * 100).toFixed(1)}%`;
    }

    private renderModelInfo(): void {
        const config = this.config;
        const output = outputSpec(config);
        const task = taskKind(config);
        const info = byId("output-info");
        const describe =
            task === "binary"
                ? "binary classification"
                : task === "multiclass"
                  ? `${output.units}-class classification`
                  : "regression";
        info.replaceChildren();
        const title = document.createElement("strong");
        title.textContent = "Output";
        const units = document.createElement("span");
        units.className = "tag";
        units.textContent = `${output.units} ${output.activation}`;
        const lossLabel = document.createElement("span");
        lossLabel.textContent = "loss";
        const loss = document.createElement("span");
        loss.className = "tag";
        loss.textContent = output.loss;
        const note = document.createElement("span");
        note.className = "legend-note";
        note.textContent = describe;
        info.append(title, units, lossLabel, loss, note);

        byId("param-count").textContent = `${parameterCount(config).toLocaleString("en-US")} parameters`;
    }

    private renderLegend(): void {
        const legend = byId("class-legend");
        legend.replaceChildren();
        const item = (swatch: HTMLElement, text: string) => {
            const span = document.createElement("span");
            span.className = "legend-item";
            span.append(swatch, text);
            legend.append(span);
        };
        const dot = (color: string) => {
            const s = document.createElement("span");
            s.className = "swatch";
            s.style.background = color;
            return s;
        };
        const task = taskKind(this.config);
        if (task === "regression") {
            const bar = document.createElement("span");
            bar.className = "swatch swatch-gradient";
            item(bar, "−1 to 1");
        } else {
            const count = classCount(this.config);
            const prefix = count > 2 ? "" : "Class ";
            if (count > 2) {
                const label = document.createElement("span");
                label.className = "legend-note";
                label.textContent = "Class";
                legend.append(label);
            }
            for (let c = 0; c < count; c++) item(dot(`var(--class-${c})`), `${prefix}${c}`);
        }
        const ring = document.createElement("span");
        ring.className = "swatch swatch-ring";
        item(ring, "Test");
    }

    private updateCodePreview(): void {
        const code = generateCode(this.config, { epochs: Math.max(200, this.epoch) });
        highlightTypeScript(byId("code-preview"), code);
    }

    private redrawAll(): void {
        this.boundaryDirty = true;
        this.chartDirty = true;
        this.diagramDirty = true;
        this.controls.renderThumbnails(this.palette);
        this.requestFrame();
    }

    private onThemeChange(): void {
        this.palette = readPalette();
        this.diagram.setArchitecture({
            inputs: this.config.features.map((id) => this.featureLabel(id)),
            hidden: this.config.layers,
            output: outputSpec(this.config),
            outputColors:
                taskKind(this.config) === "multiclass"
                    ? this.palette.classes.slice(0, outputSpec(this.config).units)
                    : undefined,
        });
        this.predictionsDirty = true;
        this.redrawAll();
    }

    // -----------------------------------------------------------------------------------------
    // UI bindings
    // -----------------------------------------------------------------------------------------

    private bindTransport(): void {
        this.playButton.addEventListener("click", () => this.setPlaying(!this.playing));
        byId("step").addEventListener("click", () => this.step());
        byId("reset").addEventListener("click", () => this.reset());
    }

    private bindViewOptions(): void {
        const refresh = () => {
            this.boundary.setOptions(this.boundaryOptions());
            this.boundaryDirty = true;
            this.requestFrame();
        };
        byId("show-test").addEventListener("change", refresh);
        byId("discretize").addEventListener("change", refresh);
        const log = byId<HTMLInputElement>("log-scale");
        log.addEventListener("change", () => {
            this.chart.setLogScale(log.checked);
            this.chartDirty = true;
            this.requestFrame();
        });
    }

    private bindExport(): void {
        byId("download-model").addEventListener("click", () => {
            if (!this.trainer) return;
            try {
                const bytes = this.trainer.toBinary();
                const blob = new Blob([new Uint8Array(bytes)], { type: "application/octet-stream" });
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.download = `orion-${this.config.dataset}-epoch${this.epoch}.onn`;
                document.body.append(link);
                link.click();
                link.remove();
                setTimeout(() => URL.revokeObjectURL(url), 1000);
                this.toast(`Downloaded model (${bytes.length.toLocaleString("en-US")} bytes)`);
            } catch (error) {
                this.toast(`Export failed: ${(error as Error).message}`);
            }
        });
        byId("copy-json").addEventListener("click", () => {
            if (!this.trainer) return;
            try {
                this.copy(this.trainer.toJson(), "Model JSON copied to clipboard");
            } catch (error) {
                this.toast(`Export failed: ${(error as Error).message}`);
            }
        });
        byId("copy-code").addEventListener("click", () => {
            this.copy(generateCode(this.config, { epochs: Math.max(200, this.epoch) }), "TypeScript snippet copied");
        });
        byId("share-link").addEventListener("click", () => {
            this.writeUrl();
            this.copy(location.href, "Link to this setup copied");
        });
        byId<HTMLDetailsElement>("code-details").addEventListener("toggle", () => this.updateCodePreview());
    }

    private bindKeyboard(): void {
        document.addEventListener("keydown", (event) => {
            if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
            const target = event.target as HTMLElement | null;
            const tag = target?.tagName ?? "";
            if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || target?.isContentEditable) return;
            const key = event.key.toLowerCase();
            if (key === " " && tag !== "BUTTON" && tag !== "SUMMARY" && tag !== "A") {
                event.preventDefault();
                this.setPlaying(!this.playing);
            } else if (key === "s") {
                this.step();
            } else if (key === "r") {
                this.reset();
            }
        });
    }

    private async copy(text: string, message: string): Promise<void> {
        try {
            await navigator.clipboard.writeText(text);
            this.toast(message);
        } catch {
            // Clipboard API unavailable (insecure context or denied): fall back to a hidden textarea.
            const area = document.createElement("textarea");
            area.value = text;
            area.setAttribute("readonly", "");
            area.style.position = "fixed";
            area.style.opacity = "0";
            document.body.append(area);
            area.select();
            const ok = document.execCommand("copy");
            area.remove();
            this.toast(ok ? message : "Copy failed: your browser blocked clipboard access");
        }
    }

    private toast(message: string): void {
        const toast = byId("toast");
        toast.textContent = message;
        toast.classList.add("show");
        clearTimeout(this.toastTimer);
        this.toastTimer = window.setTimeout(() => toast.classList.remove("show"), 2200);
    }
}

function start(): void {
    try {
        new Playground();
    } catch (error) {
        console.error(error);
        const main = document.getElementById("main");
        if (main) {
            const note = document.createElement("p");
            note.className = "noscript";
            note.textContent = `The playground failed to start: ${(error as Error).message}`;
            main.before(note);
        }
    }
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
else start();
