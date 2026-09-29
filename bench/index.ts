/**
 * Training throughput and inference latency, new engine vs the legacy (≤ 0.0.2) engine.
 *
 *   pnpm bench
 *
 * Both engines train with plain SGD. The new engine uses mini-batches of 32; the legacy engine
 * only supports per-sample updates, so it gets a small, time-capped workload.
 */

import type { ActivationName } from "../src/index.js";
import { dense, Matrix, Random, Sequential } from "../src/index.js";
import type { NeuralNetworkActivationFunction } from "./legacy-engine.js";
import { NeuralNetwork } from "./legacy-engine.js";

interface Case {
    name: string;
    sizes: number[];
    hidden: ActivationName & NeuralNetworkActivationFunction;
    output: ActivationName & NeuralNetworkActivationFunction;
    samples: number;
    epochs: number;
    /** Samples the legacy engine trains on (one epoch). */
    legacySamples: number;
}

const CASES: Case[] = [
    {
        name: "XOR 2-4-1",
        sizes: [2, 4, 1],
        hidden: "tanh",
        output: "sigmoid",
        samples: 4,
        epochs: 20000,
        legacySamples: 20000,
    },
    {
        name: "MLP 64-128-10",
        sizes: [64, 128, 10],
        hidden: "relu",
        output: "softmax",
        samples: 4096,
        epochs: 5,
        legacySamples: 2000,
    },
    {
        name: "MNIST-size 784-128-10",
        sizes: [784, 128, 10],
        hidden: "relu",
        output: "softmax",
        samples: 2048,
        epochs: 3,
        legacySamples: 200,
    },
];

interface Row {
    network: string;
    engine: string;
    trainSamplesPerSec: number;
    latencyMicros: number;
}

function syntheticData(c: Case, rng: Random): { x: number[][]; y: number[][] } {
    if (c.samples === 4)
        return {
            x: [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            y: [[0], [1], [1], [0]],
        };
    const inputs = c.sizes[0];
    const classes = c.sizes[c.sizes.length - 1];
    const x: number[][] = [];
    const y: number[][] = [];
    for (let i = 0; i < c.samples; i++) {
        const label = rng.int(classes);
        const row = new Array<number>(inputs);
        for (let f = 0; f < inputs; f++) row[f] = (f % classes === label ? 0.5 : 0) + rng.uniform(0, 0.5);
        x.push(row);
        const target = new Array<number>(classes).fill(0);
        target[label] = 1;
        y.push(target);
    }
    return { x, y };
}

function median(values: number[]): number {
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}

function latency(run: () => void, budgetMs = 300): number {
    for (let i = 0; i < 50; i++) run(); // warm up
    const samples: number[] = [];
    const end = performance.now() + budgetMs;
    while (performance.now() < end) {
        const t = performance.now();
        for (let i = 0; i < 20; i++) run();
        samples.push(((performance.now() - t) / 20) * 1000);
    }
    return median(samples);
}

function benchNew(c: Case, x: number[][], y: number[][]): Row {
    const model = new Sequential({
        inputSize: c.sizes[0],
        seed: 1,
        layers: c.sizes.slice(1).map((units, i) => dense(units, i === c.sizes.length - 2 ? c.output : c.hidden)),
    });
    model.compile({ loss: c.output === "softmax" ? "cce" : "bce", optimizer: { name: "sgd", learningRate: 0.1 } });
    const xm = Matrix.from(x);
    const ym = Matrix.from(y);
    const batchSize = Math.min(32, c.samples);
    model.fit(xm, ym, { epochs: Math.max(1, Math.floor(c.epochs / 10)), batchSize }); // warm up
    const t = performance.now();
    model.fit(xm, ym, { epochs: c.epochs, batchSize });
    const seconds = (performance.now() - t) / 1000;
    const sample = x[0];
    return {
        network: c.name,
        engine: "orion 0.1 (batch 32)",
        trainSamplesPerSec: (c.samples * c.epochs) / seconds,
        latencyMicros: latency(() => model.predict(sample)),
    };
}

function benchLegacy(c: Case, x: number[][], y: number[][]): Row {
    const net = new NeuralNetwork();
    for (let i = 0; i < c.sizes.length; i++) {
        net.addLayer(c.sizes[i], i === 0 ? "linear" : i === c.sizes.length - 1 ? c.output : c.hidden);
    }
    const count = Math.min(c.legacySamples, 50);
    net.train(x.slice(0, Math.min(count, x.length)), y.slice(0, Math.min(count, y.length)), 1, 0.1, "crossEntropy"); // warm up
    const inputs: number[][] = [];
    const targets: number[][] = [];
    for (let i = 0; i < c.legacySamples; i++) {
        inputs.push(x[i % x.length]);
        targets.push(y[i % y.length]);
    }
    const t = performance.now();
    net.train(inputs, targets, 1, 0.1, "crossEntropy");
    const seconds = (performance.now() - t) / 1000;
    const sample = x[0];
    return {
        network: c.name,
        engine: "legacy 0.0.2 (per sample)",
        trainSamplesPerSec: c.legacySamples / seconds,
        latencyMicros: latency(() => net.runNetwork(sample)),
    };
}

function format(value: number, digits = 0): string {
    return value.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

const started = performance.now();
const rows: string[] = [
    "| Network | Engine | Train (samples/s) | Predict 1 sample (µs) | Train speedup | Predict speedup |",
    "|---|---|--:|--:|--:|--:|",
];
for (const c of CASES) {
    const { x, y } = syntheticData(c, new Random(7));
    const fresh = benchNew(c, x, y);
    const legacy = benchLegacy(c, x, y);
    rows.push(
        `| ${c.name} | ${fresh.engine} | ${format(fresh.trainSamplesPerSec)} | ${format(fresh.latencyMicros, 2)} | ` +
            `${format(fresh.trainSamplesPerSec / legacy.trainSamplesPerSec, 1)}× | ${format(legacy.latencyMicros / fresh.latencyMicros, 1)}× |`,
    );
    rows.push(
        `| ${c.name} | ${legacy.engine} | ${format(legacy.trainSamplesPerSec)} | ${format(legacy.latencyMicros, 2)} | 1.0× | 1.0× |`,
    );
}
console.log(rows.join("\n"));
console.log(`\nNode ${process.version}, ${process.arch}; total ${format((performance.now() - started) / 1000, 1)} s`);
