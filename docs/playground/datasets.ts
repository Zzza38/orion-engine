/**
 * Seeded 2-D toy datasets and the feature transforms fed to the network.
 * Coordinates live in the square [-DOMAIN, DOMAIN]². Pure functions, no DOM.
 */
import type { DatasetId, FeatureId, PlaygroundConfig } from "./config.js";

/** Half-width of the square the points live in. */
export const DOMAIN = 6;

export interface Point {
    x: number;
    y: number;
    /** Class index for classification, a value in [-1, 1] for regression. */
    label: number;
}

export interface Dataset {
    train: Point[];
    test: Point[];
}

/** Small, fast, seedable PRNG (mulberry32). Deterministic across platforms. */
export class Rng {
    private state: number;

    constructor(seed: number) {
        this.state = seed >>> 0;
    }

    /** Uniform float in [0, 1). */
    next(): number {
        this.state = (this.state + 0x6d2b79f5) >>> 0;
        let t = this.state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }

    uniform(min: number, max: number): number {
        return min + (max - min) * this.next();
    }

    normal(mean = 0, stddev = 1): number {
        let u = 0;
        while (u === 0) u = this.next();
        const v = this.next();
        return mean + stddev * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    shuffle<T>(items: T[]): T[] {
        for (let i = items.length - 1; i > 0; i--) {
            const j = Math.floor(this.next() * (i + 1));
            const tmp = items[i];
            items[i] = items[j];
            items[j] = tmp;
        }
        return items;
    }
}

type Generator = (count: number, noise: number, rng: Rng, classes: number) => Point[];

function circle(count: number, noise: number, rng: Rng): Point[] {
    const radius = 5;
    const points: Point[] = [];
    const labelOf = (x: number, y: number) => (Math.hypot(x, y) < radius * 0.5 ? 1 : 0);
    for (let i = 0; i < count; i++) {
        const inner = i < count / 2;
        const r = inner ? rng.uniform(0, radius * 0.5) : rng.uniform(radius * 0.7, radius);
        const angle = rng.uniform(0, 2 * Math.PI);
        const x = r * Math.sin(angle);
        const y = r * Math.cos(angle);
        const nx = rng.uniform(-radius, radius) * noise;
        const ny = rng.uniform(-radius, radius) * noise;
        points.push({ x, y, label: labelOf(x + nx, y + ny) });
    }
    return points;
}

function xor(count: number, noise: number, rng: Rng): Point[] {
    const padding = 0.3;
    const points: Point[] = [];
    for (let i = 0; i < count; i++) {
        let x = rng.uniform(-5, 5);
        x += x > 0 ? padding : -padding;
        let y = rng.uniform(-5, 5);
        y += y > 0 ? padding : -padding;
        const nx = rng.uniform(-5, 5) * noise;
        const ny = rng.uniform(-5, 5) * noise;
        points.push({ x, y, label: (x + nx) * (y + ny) >= 0 ? 1 : 0 });
    }
    return points;
}

function spiral(count: number, noise: number, rng: Rng): Point[] {
    const points: Point[] = [];
    const half = Math.floor(count / 2);
    for (let arm = 0; arm < 2; arm++) {
        const n = arm === 0 ? half : count - half;
        const delta = arm * Math.PI;
        for (let i = 0; i < n; i++) {
            const r = (i / n) * 5;
            const t = ((1.75 * i) / n) * 2 * Math.PI + delta;
            const x = r * Math.sin(t) + rng.uniform(-1, 1) * noise;
            const y = r * Math.cos(t) + rng.uniform(-1, 1) * noise;
            points.push({ x, y, label: arm });
        }
    }
    return points;
}

function moons(count: number, noise: number, rng: Rng): Point[] {
    const points: Point[] = [];
    const scale = 3;
    for (let i = 0; i < count; i++) {
        const upper = i < count / 2;
        const t = rng.uniform(0, Math.PI);
        let x = upper ? Math.cos(t) : 1 - Math.cos(t);
        let y = upper ? Math.sin(t) : 0.5 - Math.sin(t);
        x += rng.normal(0, noise * 0.6);
        y += rng.normal(0, noise * 0.6);
        points.push({ x: (x - 0.5) * scale, y: (y - 0.25) * scale, label: upper ? 0 : 1 });
    }
    return points;
}

function blobs(count: number, noise: number, rng: Rng, classes: number): Point[] {
    const points: Point[] = [];
    const k = Math.max(2, classes);
    // Two classes sit on the diagonal like TensorFlow Playground; more are spread on a circle.
    const centers =
        k === 2
            ? [
                  { x: -2.2, y: -2.2 },
                  { x: 2.2, y: 2.2 },
              ]
            : Array.from({ length: k }, (_, c) => {
                  const angle = Math.PI / 2 + (2 * Math.PI * c) / k;
                  return { x: 3.2 * Math.cos(angle), y: 3.2 * Math.sin(angle) };
              });
    const stddev = 0.55 + noise * 3.2;
    for (let i = 0; i < count; i++) {
        const label = i % k;
        const center = centers[label];
        points.push({ x: rng.normal(center.x, stddev), y: rng.normal(center.y, stddev), label });
    }
    return points;
}

function regression(fn: (x: number, y: number) => number): Generator {
    return (count, noise, rng) => {
        const points: Point[] = [];
        for (let i = 0; i < count; i++) {
            const x = rng.uniform(-DOMAIN, DOMAIN);
            const y = rng.uniform(-DOMAIN, DOMAIN);
            const value = fn(x, y) + rng.normal(0, noise * 0.8);
            points.push({ x, y, label: Math.max(-1, Math.min(1, value)) });
        }
        return points;
    };
}

/** Ground-truth functions of the regression datasets (useful for previews). */
export const REGRESSION_TARGETS: Readonly<Record<"plane" | "wave", (x: number, y: number) => number>> = {
    plane: (x, y) => (x + y) / 10,
    wave: (x, y) => Math.sin(x * 0.55) * Math.cos(y * 0.55),
};

const GENERATORS: Readonly<Record<DatasetId, Generator>> = {
    circle,
    xor,
    spiral,
    moons,
    blobs,
    plane: regression(REGRESSION_TARGETS.plane),
    wave: regression(REGRESSION_TARGETS.wave),
};

/** Generates `count` points (unsplit). `noise` is a fraction in [0, 0.5]. */
export function generatePoints(dataset: DatasetId, count: number, noise: number, seed: number, classes = 2): Point[] {
    const rng = new Rng((Math.imul(seed >>> 0, 0x9e3779b1) + 0x9e37) >>> 0);
    return GENERATORS[dataset](count, noise, rng, classes);
}

/** Generates the dataset described by `config` and splits it into shuffled train/test sets. */
export function generateDataset(
    config: Pick<PlaygroundConfig, "dataset" | "samples" | "noise" | "dataSeed" | "trainRatio" | "classes">,
): Dataset {
    const points = generatePoints(config.dataset, config.samples, config.noise / 100, config.dataSeed, config.classes);
    const rng = new Rng((config.dataSeed ^ 0x5bd1e995) >>> 0);
    rng.shuffle(points);
    const trainCount = Math.max(1, Math.min(points.length - 1, Math.round((points.length * config.trainRatio) / 100)));
    return { train: points.slice(0, trainCount), test: points.slice(trainCount) };
}

// ---------------------------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------------------------

const INV = 1 / DOMAIN;

/**
 * Feature transforms. Coordinates are scaled to [-1, 1] first so every input has a similar
 * range; the sine features use the raw coordinate so they complete about two periods.
 */
export const FEATURE_FNS: Readonly<Record<FeatureId, (x: number, y: number) => number>> = {
    x: (x) => x * INV,
    y: (_, y) => y * INV,
    x2: (x) => x * INV * (x * INV),
    y2: (_, y) => y * INV * (y * INV),
    xy: (x, y) => x * INV * (y * INV),
    sinx: (x) => Math.sin(x),
    siny: (_, y) => Math.sin(y),
};

/** Row-major [points.length, features.length] feature values. */
export function featureData(points: readonly { x: number; y: number }[], features: readonly FeatureId[]): Float64Array {
    const fns = features.map((f) => FEATURE_FNS[f]);
    const cols = fns.length;
    const out = new Float64Array(points.length * cols);
    for (let i = 0; i < points.length; i++) {
        const p = points[i];
        for (let c = 0; c < cols; c++) out[i * cols + c] = fns[c](p.x, p.y);
    }
    return out;
}

/** Row-major [size², features.length] feature values for a size×size grid covering the domain (row 0 = top). */
export function gridFeatureData(size: number, features: readonly FeatureId[]): Float64Array {
    const points: { x: number; y: number }[] = new Array(size * size);
    for (let row = 0; row < size; row++) {
        const y = DOMAIN - ((row + 0.5) / size) * 2 * DOMAIN;
        for (let col = 0; col < size; col++) {
            const x = -DOMAIN + ((col + 0.5) / size) * 2 * DOMAIN;
            points[row * size + col] = { x, y };
        }
    }
    return featureData(points, features);
}

/** Label column ([n, 1]): class indices for classification, targets for regression. */
export function labelData(points: readonly Point[]): Float64Array {
    return Float64Array.from(points, (p) => p.label);
}

// ---------------------------------------------------------------------------------------------
// Scoring helpers (computed from predictions so they do not depend on engine metric naming)
// ---------------------------------------------------------------------------------------------

/**
 * Accuracy of `predictions` ([n, outputs] row-major) against class labels.
 * One output column is read as a sigmoid probability of class 1.
 */
export function accuracyOf(predictions: ArrayLike<number>, outputs: number, points: readonly Point[]): number {
    if (points.length === 0) return Number.NaN;
    let correct = 0;
    for (let i = 0; i < points.length; i++) {
        let predicted: number;
        if (outputs === 1) {
            predicted = predictions[i] >= 0.5 ? 1 : 0;
        } else {
            predicted = 0;
            for (let c = 1; c < outputs; c++) {
                if (predictions[i * outputs + c] > predictions[i * outputs + predicted]) predicted = c;
            }
        }
        if (predicted === points[i].label) correct++;
    }
    return correct / points.length;
}

/** Coefficient of determination R² of single-column predictions against regression targets. */
export function rSquaredOf(predictions: ArrayLike<number>, points: readonly Point[]): number {
    if (points.length === 0) return Number.NaN;
    let mean = 0;
    for (const p of points) mean += p.label;
    mean /= points.length;
    let residual = 0;
    let total = 0;
    for (let i = 0; i < points.length; i++) {
        residual += (points[i].label - predictions[i]) ** 2;
        total += (points[i].label - mean) ** 2;
    }
    return total === 0 ? Number.NaN : 1 - residual / total;
}
