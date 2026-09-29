/**
 * Training callbacks and the {@link History} object returned by `fit`.
 *
 * A callback is any object with some of the hooks of {@link Callback}. Hooks may return a
 * Promise; `fitAsync` awaits it, while the synchronous `fit` rejects it with an error that points
 * to `fitAsync`. The factories below return plain objects, so they compose with your own:
 *
 * ```ts
 * model.fit(x, y, {
 *     epochs: 200,
 *     validationSplit: 0.2,
 *     callbacks: [earlyStopping({ patience: 10, restoreBestWeights: true }), { onEpochEnd: (e, logs) => plot(logs) }],
 * });
 * ```
 */
import { ValidationError } from "./core/errors.js";
import type { Optimizer } from "./core/types.js";
import type { Sequential } from "./model.js";
import {
    booleanOption,
    checkOptions,
    describeValue,
    nonNegativeInteger,
    numberOption,
    positiveInteger,
} from "./utils.js";

/**
 * Flat per-epoch (or per-batch) numbers: `loss`, one entry per metric (e.g. `accuracy`),
 * `valLoss` / `val<Metric>` when validating, `learningRate` and `durationMs`.
 */
export type Logs = Record<string, number>;

/** Direction in which a monitored value improves. "auto" means "max" for accuracy-like keys, else "min". */
export type MonitorMode = "min" | "max" | "auto";

/** What a callback can see and do during `fit`. */
export interface CallbackContext {
    /** The model being trained. */
    readonly model: Sequential;
    /** The model's optimizer; its `learningRate` may be changed between steps. */
    readonly optimizer: Optimizer;
    /** Planned number of epochs. */
    readonly epochs: number;
    readonly batchSize: number;
    /** Number of training samples (after any validation split). */
    readonly samples: number;
    /** Batches per epoch. */
    readonly stepsPerEpoch: number;
    /** True when `fit` computes `valLoss` and friends. */
    readonly hasValidation: boolean;
    /** Logs recorded so far. */
    readonly history: History;
    /** True once {@link CallbackContext.stopTraining} has been called. */
    readonly stopRequested: boolean;
    /** Ends training after the current epoch (the remaining callbacks of this epoch still run). */
    stopTraining(): void;
}

/** Hooks called by `fit` / `fitAsync`. All are optional; each may return a Promise (fitAsync only). */
export interface Callback {
    onTrainBegin?(ctx: CallbackContext): void | Promise<void>;
    onEpochBegin?(epoch: number, ctx: CallbackContext): void | Promise<void>;
    /** `logs` holds the batch's `loss` and metrics (computed before the weight update) and `size`. */
    onBatchEnd?(batch: number, logs: Logs, ctx: CallbackContext): void | Promise<void>;
    onEpochEnd?(epoch: number, logs: Logs, ctx: CallbackContext): void | Promise<void>;
    /** `logs` are the last epoch's logs (empty when no epoch ran). */
    onTrainEnd?(logs: Logs, ctx: CallbackContext): void | Promise<void>;
}

// ---------------------------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------------------------

/**
 * Per-epoch logs of one `fit` call. `history[key][i]` is the value of `key` at `epochs[i]`.
 *
 * @example
 * const history = model.fit(x, y, { epochs: 50, validationSplit: 0.2 });
 * history.last("loss");          // final training loss
 * history.best("valAccuracy");   // { epoch, value } of the highest validation accuracy
 */
export class History {
    /** 0-based indices of the completed epochs. */
    readonly epochs: number[] = [];
    /** One array per log key, aligned with {@link History.epochs} (NaN where a key was missing). */
    readonly history: Record<string, number[]> = {};

    /** Number of recorded epochs. */
    get length(): number {
        return this.epochs.length;
    }

    /** Records one epoch's logs. */
    append(epoch: number, logs: Logs): void {
        const count = this.epochs.length;
        this.epochs.push(epoch);
        for (const key of Object.keys(logs)) {
            let series = this.history[key];
            if (series === undefined) {
                series = new Array<number>(count).fill(Number.NaN);
                this.history[key] = series;
            }
            series.push(logs[key]);
        }
        for (const key of Object.keys(this.history)) {
            const series = this.history[key];
            if (series.length === count) series.push(Number.NaN);
        }
    }

    /** The most recent value of `key`, or undefined if it was never logged. */
    last(key: string): number | undefined {
        const series = this.history[key];
        return series === undefined || series.length === 0 ? undefined : series[series.length - 1];
    }

    /**
     * The best value of `key` and the epoch it occurred in (first occurrence on ties), or
     * undefined if it was never logged. NaN entries are ignored.
     * @param mode "min", "max", or "auto" (max for accuracy-like keys, min otherwise). Default "auto".
     */
    best(key: string, mode: MonitorMode = "auto"): { epoch: number; value: number } | undefined {
        const series = this.history[key];
        if (series === undefined) return undefined;
        const maximize = resolveMode(key, mode) === "max";
        let bestIndex = -1;
        for (let i = 0; i < series.length; i++) {
            const v = series[i];
            if (Number.isNaN(v)) continue;
            if (bestIndex < 0 || (maximize ? v > series[bestIndex] : v < series[bestIndex])) bestIndex = i;
        }
        return bestIndex < 0 ? undefined : { epoch: this.epochs[bestIndex], value: series[bestIndex] };
    }

    /** A plain `{ epochs, history }` snapshot. */
    toJSON(): { epochs: number[]; history: Record<string, number[]> } {
        const history: Record<string, number[]> = {};
        for (const key of Object.keys(this.history)) history[key] = this.history[key].slice();
        return { epochs: this.epochs.slice(), history };
    }
}

function resolveMode(key: string, mode: MonitorMode): "min" | "max" {
    if (mode === "min" || mode === "max") return mode;
    return /acc|auc/i.test(key) ? "max" : "min";
}

function checkMode(where: string, mode: unknown): MonitorMode {
    if (mode === undefined) return "auto";
    if (mode !== "min" && mode !== "max" && mode !== "auto") {
        throw new ValidationError(`${where}: "mode" must be "min", "max" or "auto", got ${describeValue(mode)}`);
    }
    return mode;
}

/**
 * Tracks one monitored log value. Falls back from `val<Key>` to `<key>` (with a single warning)
 * when training has no validation data, and throws for keys that are never logged.
 */
class Monitor {
    private key: string | null = null;
    private maximize = false;
    best = Number.NaN;

    constructor(
        private readonly where: string,
        readonly monitor: string,
        private readonly mode: MonitorMode,
        private readonly minDelta: number,
    ) {}

    reset(): void {
        this.key = null;
        this.best = Number.NaN;
    }

    /** The monitored value in `logs`, resolving the key on first use. */
    read(logs: Logs, ctx: CallbackContext): number {
        if (this.key === null) {
            this.key = this.resolveKey(logs, ctx);
            this.maximize = resolveMode(this.key, this.mode) === "max";
            this.best = this.maximize ? -Infinity : Infinity;
        }
        const value = logs[this.key];
        return typeof value === "number" ? value : Number.NaN;
    }

    /** True when `value` beats the best value so far by more than `minDelta`. */
    improves(value: number): boolean {
        if (Number.isNaN(value)) return false;
        return this.maximize ? value > this.best + this.minDelta : value < this.best - this.minDelta;
    }

    private resolveKey(logs: Logs, ctx: CallbackContext): string {
        const monitor = this.monitor;
        if (monitor in logs) return monitor;
        if (monitor.startsWith("val") && monitor.length > 3 && !ctx.hasValidation) {
            const base = monitor.charAt(3).toLowerCase() + monitor.slice(4);
            if (base in logs) {
                console.warn(
                    `${this.where}: "${monitor}" is not available because fit() has no validation data; ` +
                        `monitoring "${base}" instead. Pass validationSplit or validationData to monitor "${monitor}".`,
                );
                return base;
            }
        }
        throw new ValidationError(
            `${this.where}: monitored value "${monitor}" is not in the epoch logs. ` +
                `Available: ${Object.keys(logs).join(", ")}`,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// earlyStopping
// ---------------------------------------------------------------------------------------------

/** Options for {@link earlyStopping}. */
export interface EarlyStoppingOptions {
    /** Log key to watch. Default "valLoss" (falls back to "loss", with a warning, without validation data). */
    monitor?: string;
    /** Stop after this many consecutive epochs without improvement (0 behaves like 1). Default 0. */
    patience?: number;
    /** Minimum change that counts as an improvement. Default 0. */
    minDelta?: number;
    /** Default "auto": maximize accuracy-like keys, minimize everything else. */
    mode?: MonitorMode;
    /** When training ends, restore the weights from the best epoch. Default false. */
    restoreBestWeights?: boolean;
}

/** The object returned by {@link earlyStopping}; its fields describe the last run. */
export interface EarlyStoppingCallback extends Callback {
    /** Epoch at which training was stopped, or null if it ran to completion. */
    readonly stoppedEpoch: number | null;
    /** Epoch with the best monitored value, or null before the first epoch. */
    readonly bestEpoch: number | null;
    /** Best monitored value, or null before the first epoch. */
    readonly bestValue: number | null;
}

/**
 * Stops training once the monitored value has not improved for `patience` epochs.
 * With `restoreBestWeights`, the model ends training with the weights of its best epoch
 * (also when training ran to completion). State resets at the start of every `fit`, so the
 * same callback can be reused.
 *
 * @example
 * model.fit(x, y, { epochs: 500, validationSplit: 0.2, callbacks: [earlyStopping({ patience: 20, restoreBestWeights: true })] });
 */
export function earlyStopping(options: EarlyStoppingOptions = {}): EarlyStoppingCallback {
    const where = "earlyStopping";
    checkOptions(where, options, ["monitor", "patience", "minDelta", "mode", "restoreBestWeights"]);
    const monitor = new Monitor(
        where,
        monitorKey(where, options.monitor),
        checkMode(where, options.mode),
        numberOption(where, "minDelta", options.minDelta, 0, "a finite number >= 0", (v) => v >= 0),
    );
    const patience = nonNegativeInteger(where, "patience", options.patience, 0);
    const restoreBestWeights = booleanOption(where, "restoreBestWeights", options.restoreBestWeights, false);

    let wait = 0;
    let stoppedEpoch: number | null = null;
    let bestEpoch: number | null = null;
    let bestWeights: ReturnType<Sequential["getWeights"]> | null = null;

    return {
        get stoppedEpoch() {
            return stoppedEpoch;
        },
        get bestEpoch() {
            return bestEpoch;
        },
        get bestValue() {
            return bestEpoch === null ? null : monitor.best;
        },
        onTrainBegin() {
            monitor.reset();
            wait = 0;
            stoppedEpoch = null;
            bestEpoch = null;
            bestWeights = null;
        },
        onEpochEnd(epoch, logs, ctx) {
            const value = monitor.read(logs, ctx);
            if (monitor.improves(value)) {
                monitor.best = value;
                bestEpoch = epoch;
                wait = 0;
                if (restoreBestWeights) bestWeights = ctx.model.getWeights();
                return;
            }
            wait++;
            if (wait >= patience) {
                stoppedEpoch = epoch;
                ctx.stopTraining();
            }
        },
        onTrainEnd(_logs, ctx) {
            if (restoreBestWeights && bestWeights !== null) ctx.model.setWeights(bestWeights);
        },
    };
}

function monitorKey(where: string, value: unknown): string {
    if (value === undefined) return "valLoss";
    if (typeof value !== "string" || value.length === 0) {
        throw new ValidationError(
            `${where}: "monitor" must be a log key such as "valLoss", got ${describeValue(value)}`,
        );
    }
    return value;
}

// ---------------------------------------------------------------------------------------------
// learningRateScheduler
// ---------------------------------------------------------------------------------------------

/**
 * Sets the optimizer's learning rate at the start of every epoch from `schedule(epoch, currentLr)`.
 * Works with the factories in `schedules` (`cosineDecay`, `stepDecay`, …) or any function.
 *
 * @example
 * callbacks: [learningRateScheduler(cosineDecay({ initial: 0.01, epochs: 100 }))]
 * callbacks: [learningRateScheduler((epoch, lr) => (epoch < 10 ? lr : lr * 0.95))]
 */
export function learningRateScheduler(schedule: (epoch: number, learningRate: number) => number): Callback {
    if (typeof schedule !== "function") {
        throw new ValidationError(
            `learningRateScheduler: expected a function (epoch, learningRate) => number, got ${describeValue(schedule)}`,
        );
    }
    return {
        onEpochBegin(epoch, ctx) {
            const lr = schedule(epoch, ctx.optimizer.learningRate);
            if (typeof lr !== "number" || !Number.isFinite(lr) || lr < 0) {
                throw new ValidationError(
                    `learningRateScheduler: schedule returned ${describeValue(lr)} for epoch ${epoch}; ` +
                        "it must return a finite number >= 0",
                );
            }
            ctx.optimizer.learningRate = lr;
        },
    };
}

// ---------------------------------------------------------------------------------------------
// reduceLROnPlateau
// ---------------------------------------------------------------------------------------------

/** Options for {@link reduceLROnPlateau}. */
export interface ReduceLROnPlateauOptions {
    /** Log key to watch. Default "valLoss" (falls back to "loss", with a warning, without validation data). */
    monitor?: string;
    /** Multiplier in (0, 1) applied to the learning rate on a plateau. Default 0.1. */
    factor?: number;
    /** Epochs without improvement before reducing. Default 10. */
    patience?: number;
    /** Minimum change that counts as an improvement. Default 1e-4. */
    minDelta?: number;
    /** Default "auto". */
    mode?: MonitorMode;
    /** Epochs to wait after a reduction before counting again. Default 0. */
    cooldown?: number;
    /** Lower bound for the learning rate. Default 0. */
    minLearningRate?: number;
}

/**
 * Multiplies the learning rate by `factor` when the monitored value has not improved for
 * `patience` epochs, never going below `minLearningRate`.
 *
 * @example
 * callbacks: [reduceLROnPlateau({ factor: 0.5, patience: 5, minLearningRate: 1e-5 })]
 */
export function reduceLROnPlateau(options: ReduceLROnPlateauOptions = {}): Callback {
    const where = "reduceLROnPlateau";
    checkOptions(where, options, ["monitor", "factor", "patience", "minDelta", "mode", "cooldown", "minLearningRate"]);
    const monitor = new Monitor(
        where,
        monitorKey(where, options.monitor),
        checkMode(where, options.mode),
        numberOption(where, "minDelta", options.minDelta, 1e-4, "a finite number >= 0", (v) => v >= 0),
    );
    const factor = numberOption(where, "factor", options.factor, 0.1, "a number in (0, 1)", (v) => v > 0 && v < 1);
    const patience = nonNegativeInteger(where, "patience", options.patience, 10);
    const cooldown = nonNegativeInteger(where, "cooldown", options.cooldown, 0);
    const minLearningRate = numberOption(
        where,
        "minLearningRate",
        options.minLearningRate,
        0,
        "a finite number >= 0",
        (v) => v >= 0,
    );

    let wait = 0;
    let cooldownLeft = 0;
    return {
        onTrainBegin() {
            monitor.reset();
            wait = 0;
            cooldownLeft = 0;
        },
        onEpochEnd(_epoch, logs, ctx) {
            const value = monitor.read(logs, ctx);
            const inCooldown = cooldownLeft > 0;
            if (inCooldown) {
                cooldownLeft--;
                wait = 0;
            }
            if (monitor.improves(value)) {
                monitor.best = value;
                wait = 0;
                return;
            }
            if (inCooldown) return;
            wait++;
            if (wait >= patience) {
                const lr = ctx.optimizer.learningRate;
                if (lr > minLearningRate) {
                    ctx.optimizer.learningRate = Math.max(lr * factor, minLearningRate);
                    cooldownLeft = cooldown;
                    wait = 0;
                }
            }
        },
    };
}

// ---------------------------------------------------------------------------------------------
// progressLogger
// ---------------------------------------------------------------------------------------------

/** Options for {@link progressLogger}. */
export interface ProgressLoggerOptions {
    /** Log every N epochs (the final epoch is always logged). Default 1. */
    every?: number;
    /** Where lines go. Default `console.log`. */
    log?: (line: string) => void;
}

/**
 * Logs one line per `every` epochs, e.g. `Epoch 10/100 - loss: 0.2143 - accuracy: 0.9375 - 4ms`.
 * `fit(x, y, { verbose: true })` (or `verbose: 10`) installs one for you.
 */
export function progressLogger(options: ProgressLoggerOptions = {}): Callback {
    const where = "progressLogger";
    checkOptions(where, options, ["every", "log"]);
    const every = positiveInteger(where, "every", options.every, 1);
    const log = options.log ?? ((line: string) => console.log(line));
    if (typeof log !== "function")
        throw new ValidationError(`${where}: "log" must be a function, got ${describeValue(log)}`);
    return {
        onEpochEnd(epoch, logs, ctx) {
            const n = epoch + 1;
            if (n % every === 0 || n === ctx.epochs || ctx.stopRequested) log(formatEpoch(n, ctx.epochs, logs));
        },
    };
}

/** "Epoch 3/10 - loss: 0.6931 - accuracy: 0.5000 - 2ms" */
export function formatEpoch(epoch: number, epochs: number, logs: Logs): string {
    const width = String(epochs).length;
    let line = `Epoch ${String(epoch).padStart(width)}/${epochs}`;
    for (const key of Object.keys(logs)) {
        if (key === "durationMs" || key === "learningRate") continue;
        line += ` - ${key}: ${formatNumber(logs[key])}`;
    }
    const ms = logs.durationMs;
    if (typeof ms === "number") line += ` - ${ms < 10 ? ms.toFixed(1) : Math.round(ms)}ms`;
    return line;
}

function formatNumber(value: number): string {
    if (!Number.isFinite(value)) return String(value);
    const abs = Math.abs(value);
    return abs !== 0 && (abs < 1e-3 || abs >= 1e6) ? value.toExponential(3) : value.toFixed(4);
}
