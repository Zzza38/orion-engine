/**
 * Learning-rate schedules.
 *
 * A schedule is a pure function from a 0-based epoch index to a learning rate. Fractional
 * epochs are accepted (e.g. `epoch + batch / batchesPerEpoch` for per-batch schedules).
 * Every factory validates its options up front and throws `ValidationError` on bad input;
 * the returned schedule throws `ValidationError` for a negative or non-finite epoch.
 */
import { ValidationError } from "./core/errors.js";

/** Maps a 0-based epoch index (≥ 0, may be fractional) to a learning rate. */
export type LearningRateSchedule = (epoch: number) => number;

/** Options for {@link stepDecay}. */
export interface StepDecayOptions {
    /** Learning rate for epochs [0, every). */
    initial: number;
    /** Multiplier in (0, 1] applied every `every` epochs. */
    factor: number;
    /** Positive integer number of epochs between decays. */
    every: number;
}

/** Options for {@link exponentialDecay}. */
export interface ExponentialDecayOptions {
    /** Learning rate at epoch 0. */
    initial: number;
    /** Decay rate in (0, 1]: the learning rate is multiplied by `rate` every `every` epochs. */
    rate: number;
    /** Positive number of epochs over which the full `rate` factor applies. Default 1. */
    every?: number;
}

/** Options for {@link cosineDecay}. */
export interface CosineDecayOptions {
    /** Learning rate at epoch 0. */
    initial: number;
    /** Positive integer number of epochs to decay over; the rate stays at `minimum` afterwards. */
    epochs: number;
    /** Final learning rate, in [0, initial]. Default 0. */
    minimum?: number;
}

/** Options for {@link linearWarmup}. */
export interface LinearWarmupOptions {
    /** Non-negative integer number of warm-up epochs (0 disables the warm-up). */
    epochs: number;
    /** Learning rate at epoch 0, ≥ 0. Default 0. */
    from?: number;
}

/** Options for {@link piecewiseConstant}. */
export interface PiecewiseConstantOptions {
    /** Strictly increasing, non-negative epochs at which the next value takes effect. */
    boundaries: readonly number[];
    /** Learning rates, one more than `boundaries`. */
    values: readonly number[];
}

// ---------------------------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------------------------

function describeValue(value: unknown): string {
    return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function fail(where: string, key: string, requirement: string, value: unknown): never {
    throw new ValidationError(`${where}: "${key}" must be ${requirement}, got ${describeValue(value)}`);
}

function requireRate(where: string, key: string, value: unknown): number {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0)
        fail(where, key, "a finite number >= 0", value);
    return value;
}

function requirePositiveInteger(where: string, key: string, value: unknown): number {
    if (typeof value !== "number" || !Number.isInteger(value) || value <= 0)
        fail(where, key, "a positive integer", value);
    return value;
}

function requireFactor(where: string, key: string, value: unknown): number {
    if (typeof value !== "number" || !(value > 0 && value <= 1)) fail(where, key, "a number in (0, 1]", value);
    return value;
}

function requireOptions(where: string, options: unknown): void {
    if (options === null || typeof options !== "object" || Array.isArray(options)) {
        throw new ValidationError(`${where}: options must be an object, got ${describeValue(options)}`);
    }
}

function checkEpoch(where: string, epoch: number): void {
    if (typeof epoch !== "number" || !Number.isFinite(epoch) || epoch < 0) {
        throw new ValidationError(`${where}: epoch must be a finite number >= 0, got ${describeValue(epoch)}`);
    }
}

// ---------------------------------------------------------------------------------------------
// Schedules
// ---------------------------------------------------------------------------------------------

/** `lr(epoch) = learningRate` for every epoch. */
export function constantSchedule(learningRate: number): LearningRateSchedule {
    const where = "constantSchedule";
    const lr = requireRate(where, "learningRate", learningRate);
    return (epoch) => {
        checkEpoch(where, epoch);
        return lr;
    };
}

/**
 * Staircase decay: `lr(epoch) = initial · factor^⌊epoch / every⌋`.
 * E.g. `stepDecay({ initial: 0.1, factor: 0.5, every: 10 })` halves the rate at epochs 10, 20, ….
 */
export function stepDecay(options: StepDecayOptions): LearningRateSchedule {
    const where = "stepDecay";
    requireOptions(where, options);
    const initial = requireRate(where, "initial", options.initial);
    const factor = requireFactor(where, "factor", options.factor);
    const every = requirePositiveInteger(where, "every", options.every);
    return (epoch) => {
        checkEpoch(where, epoch);
        return initial * Math.pow(factor, Math.floor(epoch / every));
    };
}

/**
 * Smooth exponential decay: `lr(epoch) = initial · rate^(epoch / every)`.
 * With the default `every = 1` the rate is multiplied by `rate` each epoch.
 */
export function exponentialDecay(options: ExponentialDecayOptions): LearningRateSchedule {
    const where = "exponentialDecay";
    requireOptions(where, options);
    const initial = requireRate(where, "initial", options.initial);
    const rate = requireFactor(where, "rate", options.rate);
    const every = options.every ?? 1;
    if (typeof every !== "number" || !Number.isFinite(every) || every <= 0)
        fail(where, "every", "a finite number > 0", every);
    return (epoch) => {
        checkEpoch(where, epoch);
        return initial * Math.pow(rate, epoch / every);
    };
}

/**
 * Cosine annealing from `initial` to `minimum` over `epochs` epochs, then constant:
 * `lr(epoch) = minimum + (initial − minimum) · ½(1 + cos(π · min(epoch, epochs) / epochs))`.
 * `lr(0) = initial`, `lr(epochs) = minimum`.
 */
export function cosineDecay(options: CosineDecayOptions): LearningRateSchedule {
    const where = "cosineDecay";
    requireOptions(where, options);
    const initial = requireRate(where, "initial", options.initial);
    const epochs = requirePositiveInteger(where, "epochs", options.epochs);
    const minimum = requireRate(where, "minimum", options.minimum ?? 0);
    if (minimum > initial) fail(where, "minimum", `<= initial (${initial})`, minimum);
    const range = initial - minimum;
    return (epoch) => {
        checkEpoch(where, epoch);
        if (epoch >= epochs) return minimum;
        return minimum + range * 0.5 * (1 + Math.cos((Math.PI * epoch) / epochs));
    };
}

/**
 * Prepends a linear warm-up to another schedule. For `epoch < epochs` the rate ramps linearly
 * from `from` (at epoch 0) toward `schedule(0)`; afterwards the wrapped schedule runs shifted
 * by the warm-up length:
 *
 * ```text
 * lr(epoch) = from + (schedule(0) − from) · epoch / epochs     epoch < epochs
 * lr(epoch) = schedule(epoch − epochs)                         otherwise
 * ```
 * E.g. `linearWarmup(cosineDecay({ initial: 1e-3, epochs: 100 }), { epochs: 5 })` warms up for
 * 5 epochs, then decays over the next 100.
 */
export function linearWarmup(schedule: LearningRateSchedule, options: LinearWarmupOptions): LearningRateSchedule {
    const where = "linearWarmup";
    if (typeof schedule !== "function") {
        throw new ValidationError(`${where}: schedule must be a function, got ${describeValue(schedule)}`);
    }
    requireOptions(where, options);
    const warmup = options.epochs;
    if (typeof warmup !== "number" || !Number.isInteger(warmup) || warmup < 0) {
        fail(where, "epochs", "a non-negative integer", warmup);
    }
    const from = requireRate(where, "from", options.from ?? 0);
    return (epoch) => {
        checkEpoch(where, epoch);
        if (epoch >= warmup) return schedule(epoch - warmup);
        return from + (schedule(0) - from) * (epoch / warmup);
    };
}

/**
 * Piecewise-constant rate. Each boundary is the first epoch at which the next value applies:
 *
 * ```text
 * lr(epoch) = values[0]   epoch < boundaries[0]
 *           = values[k]   boundaries[k−1] ≤ epoch < boundaries[k]
 *           = values[n]   epoch ≥ boundaries[n−1]
 * ```
 * E.g. `piecewiseConstant({ boundaries: [10, 20], values: [0.1, 0.01, 0.001] })`.
 */
export function piecewiseConstant(options: PiecewiseConstantOptions): LearningRateSchedule {
    const where = "piecewiseConstant";
    requireOptions(where, options);
    const { boundaries, values } = options;
    if (!Array.isArray(boundaries)) fail(where, "boundaries", "an array of numbers", boundaries);
    if (!Array.isArray(values)) fail(where, "values", "an array of numbers", values);
    if (values.length !== boundaries.length + 1) {
        throw new ValidationError(
            `${where}: expected ${boundaries.length + 1} values for ${boundaries.length} boundaries, got ${values.length}`,
        );
    }
    const bounds = Float64Array.from(boundaries, (b, i) => {
        if (typeof b !== "number" || !Number.isFinite(b) || b < 0)
            fail(where, `boundaries[${i}]`, "a finite number >= 0", b);
        if (i > 0 && !(b > boundaries[i - 1]))
            fail(where, `boundaries[${i}]`, `> boundaries[${i - 1}] (${boundaries[i - 1]})`, b);
        return b;
    });
    const rates = Float64Array.from(values, (v, i) => requireRate(where, `values[${i}]`, v));
    return (epoch) => {
        checkEpoch(where, epoch);
        let k = 0;
        while (k < bounds.length && epoch >= bounds[k]) k++;
        return rates[k];
    };
}
