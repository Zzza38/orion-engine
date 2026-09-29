/**
 * Binds the settings UI (dataset picker, sliders, feature chips, layer builder, hyperparameters)
 * to a {@link PlaygroundConfig}. The DOM is updated in place so keyboard focus survives changes.
 */
import type { ActivationId, DatasetId, FeatureId, OptimizerId, PlaygroundConfig, SpeedId } from "./config.js";
import {
    ACTIVATIONS,
    BATCH_SIZES,
    cloneConfig,
    DATASETS,
    DROPOUT_RATES,
    FEATURES,
    L2_RATES,
    LEARNING_RATES,
    LIMITS,
    OPTIMIZERS,
    SPEEDS,
    sanitizeConfig,
} from "./config.js";
import { drawDatasetThumbnail } from "./heatmap.js";
import type { Palette } from "./theme.js";

/**
 * What a change invalidates: `data` regenerates points, `features` rebuilds inputs, `model`
 * rebuilds the network, `runtime` only affects how training runs (no reset).
 */
export type ChangeKind = "data" | "features" | "model" | "runtime";

export type ChangeHandler = (config: PlaygroundConfig, kind: ChangeKind) => void;

function byId<T extends HTMLElement>(id: string): T {
    const element = document.getElementById(id);
    if (!element) throw new Error(`Missing #${id}`);
    return element as T;
}

function svgIcon(path: string): SVGSVGElement {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 20 20");
    svg.setAttribute("aria-hidden", "true");
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", path);
    svg.append(p);
    return svg;
}

function fillOptions(select: HTMLSelectElement, options: readonly { value: string; label: string }[]): void {
    select.replaceChildren(
        ...options.map(({ value, label }) => {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = label;
            return option;
        }),
    );
}

function formatRate(value: number): string {
    if (value === 0) return "None";
    return value < 0.001 ? value.toFixed(6).replace(/0+$/, "") : String(value);
}

/** Keeps the WebKit slider track fill in sync with the value. */
function paintRange(input: HTMLInputElement): void {
    const min = Number(input.min);
    const max = Number(input.max);
    const pct = max > min ? ((Number(input.value) - min) / (max - min)) * 100 : 0;
    input.style.setProperty("--fill", `${pct}%`);
}

interface LayerRow {
    item: HTMLLIElement;
    name: HTMLSpanElement;
    minus: HTMLButtonElement;
    units: HTMLInputElement;
    plus: HTMLButtonElement;
    activation: HTMLSelectElement;
    dropout: HTMLSelectElement;
    remove: HTMLButtonElement;
}

export class Controls {
    private config: PlaygroundConfig;
    private readonly onChange: ChangeHandler;
    private readonly tiles = new Map<DatasetId, { input: HTMLInputElement; canvas: HTMLCanvasElement }>();
    private readonly featureInputs = new Map<FeatureId, HTMLInputElement>();
    private readonly rows: LayerRow[] = [];

    private readonly noise = byId<HTMLInputElement>("noise");
    private readonly samples = byId<HTMLInputElement>("samples");
    private readonly trainRatio = byId<HTMLInputElement>("train-ratio");
    private readonly classes = byId<HTMLInputElement>("classes");
    private readonly classesField = byId<HTMLElement>("classes-field");
    private readonly optimizer = byId<HTMLSelectElement>("optimizer");
    private readonly learningRate = byId<HTMLSelectElement>("learning-rate");
    private readonly batchSize = byId<HTMLSelectElement>("batch-size");
    private readonly l2 = byId<HTMLSelectElement>("l2");
    private readonly seed = byId<HTMLInputElement>("seed");
    private readonly speed = byId<HTMLSelectElement>("speed");
    private readonly layerList = byId<HTMLOListElement>("layer-list");
    private readonly addLayer = byId<HTMLButtonElement>("add-layer");
    private readonly removeLayer = byId<HTMLButtonElement>("remove-layer");
    private readonly layerCount = byId<HTMLElement>("layer-count");

    constructor(initial: PlaygroundConfig, onChange: ChangeHandler) {
        this.config = cloneConfig(initial);
        this.onChange = onChange;
        this.buildDatasetTiles();
        this.buildFeatureChips();
        this.buildSelects();
        this.bindInputs();
        this.sync();
    }

    get current(): PlaygroundConfig {
        return this.config;
    }

    /** Replaces the whole configuration (e.g. from the URL) without firing change events. */
    setConfig(config: PlaygroundConfig): void {
        this.config = cloneConfig(config);
        this.sync();
    }

    renderThumbnails(palette: Palette): void {
        for (const [id, { canvas }] of this.tiles) {
            drawDatasetThumbnail(canvas, id, id === "blobs" ? 3 : 2, palette);
        }
    }

    private update(kind: ChangeKind, mutate: (config: PlaygroundConfig) => void): void {
        const next = cloneConfig(this.config);
        mutate(next);
        this.config = sanitizeConfig(next);
        this.sync();
        this.onChange(this.config, kind);
    }

    // -----------------------------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------------------------

    private buildDatasetTiles(): void {
        const container = byId<HTMLElement>("dataset-tiles");
        for (const info of DATASETS) {
            const label = document.createElement("label");
            label.className = "tile";
            label.title = info.description;
            const input = document.createElement("input");
            input.type = "radio";
            input.name = "dataset";
            input.value = info.id;
            input.setAttribute("aria-describedby", `tile-desc-${info.id}`);
            const canvas = document.createElement("canvas");
            canvas.setAttribute("aria-hidden", "true");
            const text = document.createElement("span");
            text.textContent = info.label;
            const desc = document.createElement("span");
            desc.id = `tile-desc-${info.id}`;
            desc.hidden = true;
            desc.textContent = info.description;
            label.append(input, canvas, text, desc);
            container.append(label);
            input.addEventListener("change", () => {
                if (input.checked) this.update("data", (c) => (c.dataset = info.id));
            });
            this.tiles.set(info.id, { input, canvas });
        }
    }

    private buildFeatureChips(): void {
        const container = byId<HTMLElement>("feature-toggles");
        for (const feature of FEATURES) {
            const label = document.createElement("label");
            label.className = "chip";
            label.title = feature.title;
            const input = document.createElement("input");
            input.type = "checkbox";
            input.value = feature.id;
            input.setAttribute("aria-label", `Use feature ${feature.title}`);
            const text = document.createElement("span");
            text.textContent = feature.label;
            label.append(input, text);
            container.append(label);
            input.addEventListener("change", () => {
                const others = this.config.features.filter((f) => f !== feature.id);
                if (!input.checked && others.length === 0) {
                    input.checked = true; // The network needs at least one input.
                    return;
                }
                this.update("features", (c) => {
                    c.features = input.checked ? [...c.features, feature.id] : others;
                });
            });
            this.featureInputs.set(feature.id, input);
        }
    }

    private buildSelects(): void {
        fillOptions(
            this.optimizer,
            OPTIMIZERS.map((o) => ({ value: o.id, label: o.label })),
        );
        fillOptions(
            this.learningRate,
            LEARNING_RATES.map((v) => ({ value: String(v), label: formatRate(v) })),
        );
        fillOptions(
            this.batchSize,
            BATCH_SIZES.map((v) => ({ value: String(v), label: String(v) })),
        );
        fillOptions(
            this.l2,
            L2_RATES.map((v) => ({ value: String(v), label: formatRate(v) })),
        );
        fillOptions(
            this.speed,
            SPEEDS.map((s) => ({ value: s.id, label: s.label })),
        );
    }

    private bindInputs(): void {
        const slider = (input: HTMLInputElement, apply: (c: PlaygroundConfig, v: number) => void) => {
            input.addEventListener("input", () => {
                paintRange(input);
                this.update("data", (c) => apply(c, Number(input.value)));
            });
        };
        slider(this.noise, (c, v) => (c.noise = v));
        slider(this.samples, (c, v) => (c.samples = v));
        slider(this.trainRatio, (c, v) => (c.trainRatio = v));
        slider(this.classes, (c, v) => (c.classes = v));

        byId<HTMLButtonElement>("regenerate").addEventListener("click", () =>
            this.update("data", (c) => (c.dataSeed = (c.dataSeed + 1) >>> 0)),
        );

        this.optimizer.addEventListener("change", () =>
            this.update("model", (c) => (c.optimizer = this.optimizer.value as OptimizerId)),
        );
        this.learningRate.addEventListener("change", () =>
            this.update("model", (c) => (c.learningRate = Number(this.learningRate.value))),
        );
        this.l2.addEventListener("change", () => this.update("model", (c) => (c.l2 = Number(this.l2.value))));
        this.batchSize.addEventListener("change", () =>
            this.update("runtime", (c) => (c.batchSize = Number(this.batchSize.value))),
        );
        this.speed.addEventListener("change", () =>
            this.update("runtime", (c) => (c.speed = this.speed.value as SpeedId)),
        );
        this.seed.addEventListener("change", () => {
            const value = Number(this.seed.value);
            if (!Number.isFinite(value)) {
                this.seed.value = String(this.config.seed);
                return;
            }
            this.update("model", (c) => (c.seed = value));
        });
        byId<HTMLButtonElement>("reseed").addEventListener("click", () =>
            this.update("model", (c) => (c.seed = Math.floor(Math.random() * 100000))),
        );

        this.addLayer.addEventListener("click", () =>
            this.update("model", (c) => {
                const last = c.layers[c.layers.length - 1];
                c.layers.push(last ? { ...last, dropout: 0 } : { units: 4, activation: "tanh", dropout: 0 });
            }),
        );
        this.removeLayer.addEventListener("click", () => this.update("model", (c) => c.layers.pop()));
    }

    private createLayerRow(): LayerRow {
        const item = document.createElement("li");
        item.className = "layer";
        const name = document.createElement("span");
        name.className = "layer-name";

        const stepper = document.createElement("div");
        stepper.className = "stepper";
        stepper.setAttribute("role", "group");
        const minus = document.createElement("button");
        minus.type = "button";
        minus.append(svgIcon("M5 10h10"));
        const units = document.createElement("input");
        units.type = "number";
        units.min = String(LIMITS.units.min);
        units.max = String(LIMITS.units.max);
        units.inputMode = "numeric";
        const plus = document.createElement("button");
        plus.type = "button";
        plus.append(svgIcon("M10 5v10M5 10h10"));
        stepper.append(minus, units, plus);

        const activation = document.createElement("select");
        activation.className = "select";
        fillOptions(
            activation,
            ACTIVATIONS.map((a) => ({ value: a, label: a })),
        );
        const dropout = document.createElement("select");
        dropout.className = "select";
        fillOptions(
            dropout,
            DROPOUT_RATES.map((r) => ({ value: String(r), label: r === 0 ? "0%" : `${Math.round(r * 100)}%` })),
        );
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "layer-remove";
        remove.append(svgIcon("M5.5 5.5l9 9M14.5 5.5l-9 9"));

        item.append(name, stepper, activation, dropout, remove);
        const row: LayerRow = { item, name, minus, units, plus, activation, dropout, remove };

        const index = () => this.rows.indexOf(row);
        const setUnits = (value: number) =>
            this.update("model", (c) => {
                c.layers[index()].units = value;
            });
        minus.addEventListener("click", () => setUnits(this.config.layers[index()].units - 1));
        plus.addEventListener("click", () => setUnits(this.config.layers[index()].units + 1));
        units.addEventListener("change", () => {
            const value = Number(units.value);
            if (Number.isFinite(value)) setUnits(value);
            else units.value = String(this.config.layers[index()].units);
        });
        activation.addEventListener("change", () =>
            this.update("model", (c) => {
                c.layers[index()].activation = activation.value as ActivationId;
            }),
        );
        dropout.addEventListener("change", () =>
            this.update("model", (c) => {
                c.layers[index()].dropout = Number(dropout.value);
            }),
        );
        remove.addEventListener("click", () => {
            const i = index();
            this.update("model", (c) => c.layers.splice(i, 1));
            // Keep focus in the list after the row disappears.
            const next = this.rows[Math.min(i, this.rows.length - 1)];
            next?.remove.disabled ? next.units.focus() : next?.remove.focus();
        });
        return row;
    }

    // -----------------------------------------------------------------------------------------
    // Sync DOM ← config
    // -----------------------------------------------------------------------------------------

    private sync(): void {
        const c = this.config;
        for (const [id, { input }] of this.tiles) input.checked = id === c.dataset;
        for (const [id, input] of this.featureInputs) input.checked = c.features.includes(id);

        this.noise.value = String(c.noise);
        this.samples.value = String(c.samples);
        this.trainRatio.value = String(c.trainRatio);
        this.classes.value = String(c.classes);
        for (const input of [this.noise, this.samples, this.trainRatio, this.classes]) paintRange(input);
        byId("noise-out").textContent = `${c.noise}%`;
        byId("samples-out").textContent = String(c.samples);
        byId("train-ratio-out").textContent = `${c.trainRatio} / ${100 - c.trainRatio}`;
        byId("classes-out").textContent = String(c.classes);
        this.classesField.hidden = c.dataset !== "blobs";

        this.optimizer.value = c.optimizer;
        this.learningRate.value = String(c.learningRate);
        this.batchSize.value = String(c.batchSize);
        this.l2.value = String(c.l2);
        this.speed.value = c.speed;
        if (document.activeElement !== this.seed) this.seed.value = String(c.seed);

        while (this.rows.length < c.layers.length) {
            const row = this.createLayerRow();
            this.rows.push(row);
            this.layerList.append(row.item);
        }
        while (this.rows.length > c.layers.length) this.rows.pop()!.item.remove();
        c.layers.forEach((layer, i) => {
            const row = this.rows[i];
            const n = i + 1;
            row.name.textContent = `H${n}`;
            row.units.value = String(layer.units);
            row.units.setAttribute("aria-label", `Neurons in hidden layer ${n}`);
            row.minus.setAttribute("aria-label", `Remove a neuron from hidden layer ${n}`);
            row.plus.setAttribute("aria-label", `Add a neuron to hidden layer ${n}`);
            row.minus.disabled = layer.units <= LIMITS.units.min;
            row.plus.disabled = layer.units >= LIMITS.units.max;
            row.activation.value = layer.activation;
            row.activation.setAttribute("aria-label", `Activation of hidden layer ${n}`);
            row.dropout.value = String(layer.dropout);
            row.dropout.setAttribute("aria-label", `Dropout after hidden layer ${n}`);
            row.remove.setAttribute("aria-label", `Remove hidden layer ${n}`);
            row.remove.disabled = c.layers.length <= LIMITS.layers.min;
        });
        this.addLayer.disabled = c.layers.length >= LIMITS.layers.max;
        this.removeLayer.disabled = c.layers.length <= LIMITS.layers.min;
        this.layerCount.textContent = `${c.layers.length} / ${LIMITS.layers.max}`;
    }
}
