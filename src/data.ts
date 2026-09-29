/**
 * Data preparation helpers: one-hot encoding, argmax, train/test splitting, shuffling and
 * feature scaling. They accept plain arrays (and Matrices) and return the same kind.
 */
import { ShapeError, ValidationError } from "./core/errors.js";
import type { MatrixLike } from "./core/matrix.js";
import { gatherRows, Matrix } from "./core/matrix.js";
import { Random } from "./core/random.js";
import { booleanOption, checkOptions, describeValue } from "./utils.js";

// ---------------------------------------------------------------------------------------------
// oneHot / argmax
// ---------------------------------------------------------------------------------------------

/**
 * One-hot encodes integer class labels.
 * @param labels Class indices, each an integer in [0, numClasses).
 * @param numClasses Number of classes. Default: the largest label + 1.
 * @example
 * oneHot([0, 2, 1]); // [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
 */
export function oneHot(labels: ArrayLike<number>, numClasses?: number): number[][] {
    if (labels === null || typeof labels !== "object" || typeof labels.length !== "number") {
        throw new ValidationError(`oneHot: labels must be an array of class indices, got ${describeValue(labels)}`);
    }
    let classes = numClasses;
    if (classes === undefined) {
        classes = 0;
        for (let i = 0; i < labels.length; i++) if (labels[i] + 1 > classes) classes = labels[i] + 1;
    } else if (!Number.isInteger(classes) || classes < 1) {
        throw new ValidationError(`oneHot: numClasses must be a positive integer, got ${describeValue(classes)}`);
    }
    const out: number[][] = new Array(labels.length);
    for (let i = 0; i < labels.length; i++) {
        const label = labels[i];
        if (!Number.isInteger(label) || label < 0 || label >= classes) {
            throw new ValidationError(
                `oneHot: labels[${i}] = ${describeValue(label)} is not an integer class index in [0, ${classes})`,
            );
        }
        const row = new Array<number>(classes).fill(0);
        row[label] = 1;
        out[i] = row;
    }
    return out;
}

/**
 * Index of the largest value (first on ties): of one vector, or of every row.
 * @example
 * argmax([0.1, 0.7, 0.2]);                  // 1
 * argmax(model.predict([[1, 2], [3, 4]]));  // [2, 0]
 */
export function argmax(values: readonly number[]): number;
export function argmax(rows: readonly (readonly number[])[] | Matrix): number[];
export function argmax(input: readonly number[] | readonly (readonly number[])[] | Matrix): number | number[] {
    if (input instanceof Matrix) {
        const out = new Array<number>(input.rows);
        for (let r = 0; r < input.rows; r++) out[r] = argmaxOf(input.row(r));
        return out;
    }
    if (!Array.isArray(input))
        throw new ValidationError(`argmax: expected an array or a Matrix, got ${describeValue(input)}`);
    if (input.length > 0 && Array.isArray(input[0]))
        return (input as readonly (readonly number[])[]).map((row) => argmaxOf(row));
    if (input.length === 0) throw new ValidationError("argmax: input is empty");
    return argmaxOf(input as readonly number[]);
}

function argmaxOf(values: ArrayLike<number>): number {
    let best = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
    return best;
}

// ---------------------------------------------------------------------------------------------
// Splitting & shuffling
// ---------------------------------------------------------------------------------------------

/** A dataset: a Matrix, or an array with one entry (row or label) per sample. */
export type Samples = Matrix | readonly unknown[];

/** The type {@link trainTestSplit} and {@link shuffleTogether} return for a given input type. */
export type SamplesOf<T extends Samples> = T extends Matrix ? Matrix : T extends readonly (infer E)[] ? E[] : never;

/** Options for {@link trainTestSplit}. */
export interface TrainTestSplitOptions {
    /** Fraction in (0, 1) of samples for the test set, or an absolute count (integer ≥ 1). Default 0.2. */
    testSize?: number;
    /** Shuffle before splitting. Default true. Without shuffling the test set is the last samples. */
    shuffle?: boolean;
    /** Seed for the shuffle. Default random. */
    seed?: number;
}

/** Result of {@link trainTestSplit}. */
export interface TrainTestSplit<X extends Samples, Y extends Samples> {
    xTrain: SamplesOf<X>;
    xTest: SamplesOf<X>;
    yTrain: SamplesOf<Y>;
    yTest: SamplesOf<Y>;
}

/**
 * Splits inputs and targets into train and test sets, keeping samples paired.
 * The test set size is `ceil(testSize · n)` for fractions.
 * @example
 * const { xTrain, xTest, yTrain, yTest } = trainTestSplit(x, labels, { testSize: 0.25, seed: 7 });
 */
export function trainTestSplit<X extends Samples, Y extends Samples>(
    x: X,
    y: Y,
    options: TrainTestSplitOptions = {},
): TrainTestSplit<X, Y> {
    const where = "trainTestSplit";
    checkOptions(where, options, ["testSize", "shuffle", "seed"]);
    const n = sampleCount(x, where, "x");
    const ny = sampleCount(y, where, "y");
    if (n !== ny) throw new ShapeError(`${where}: x and y must have the same number of samples, got ${n} and ${ny}`);
    const testSize = options.testSize ?? 0.2;
    let testCount: number;
    if (typeof testSize === "number" && testSize > 0 && testSize < 1) testCount = Math.ceil(testSize * n);
    else if (Number.isInteger(testSize) && testSize >= 1) testCount = testSize;
    else {
        throw new ValidationError(
            `${where}: "testSize" must be a fraction in (0, 1) or a positive integer count, got ${describeValue(testSize)}`,
        );
    }
    if (testCount >= n) {
        throw new ValidationError(
            `${where}: a test set of ${testCount} leaves no training samples out of ${n}; lower testSize`,
        );
    }
    const order = identity(n);
    if (booleanOption(where, "shuffle", options.shuffle, true)) makeRandom(where, options.seed).shuffle(order);
    const trainIdx = order.subarray(0, n - testCount);
    const testIdx = order.subarray(n - testCount);
    return {
        xTrain: takeRows(x, trainIdx),
        xTest: takeRows(x, testIdx),
        yTrain: takeRows(y, trainIdx),
        yTest: takeRows(y, testIdx),
    };
}

/**
 * Shuffles inputs and targets with the same permutation, returning new arrays (or Matrices).
 * @param rng A `Random`, a seed, or nothing for a random shuffle.
 * @example
 * const [xs, ys] = shuffleTogether(x, y, 42);
 */
export function shuffleTogether<X extends Samples, Y extends Samples>(
    x: X,
    y: Y,
    rng?: Random | number,
): [SamplesOf<X>, SamplesOf<Y>] {
    const where = "shuffleTogether";
    const n = sampleCount(x, where, "x");
    const ny = sampleCount(y, where, "y");
    if (n !== ny) throw new ShapeError(`${where}: x and y must have the same number of samples, got ${n} and ${ny}`);
    const random = rng instanceof Random ? rng : makeRandom(where, rng);
    const order = random.shuffle(identity(n));
    return [takeRows(x, order), takeRows(y, order)];
}

function sampleCount(data: Samples, where: string, label: string): number {
    if (data instanceof Matrix) return data.rows;
    if (Array.isArray(data)) return data.length;
    throw new ValidationError(`${where}: ${label} must be an array or a Matrix, got ${describeValue(data)}`);
}

function identity(n: number): Uint32Array {
    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    return order;
}

function makeRandom(where: string, seed: unknown): Random {
    if (seed !== undefined && (typeof seed !== "number" || !Number.isInteger(seed))) {
        throw new ValidationError(`${where}: seed must be an integer, got ${describeValue(seed)}`);
    }
    return new Random(seed as number | undefined);
}

function takeRows<T extends Samples>(data: T, indices: Uint32Array): SamplesOf<T> {
    if (data instanceof Matrix) return gatherRows(data, indices) as SamplesOf<T>;
    const source = data as readonly unknown[];
    const out = new Array<unknown>(indices.length);
    for (let i = 0; i < indices.length; i++) {
        const row = source[indices[i]];
        out[i] = Array.isArray(row) ? row.slice() : row;
    }
    return out as SamplesOf<T>;
}

// ---------------------------------------------------------------------------------------------
// Scalers
// ---------------------------------------------------------------------------------------------

/** Serialized {@link StandardScaler}. */
export interface StandardScalerJSON {
    type: "standardScaler";
    mean: number[];
    std: number[];
}

/** Serialized {@link MinMaxScaler}. */
export interface MinMaxScalerJSON {
    type: "minMaxScaler";
    featureRange: [number, number];
    dataMin: number[];
    dataMax: number[];
}

abstract class Scaler {
    protected abstract readonly label: string;
    protected features = -1;

    /** True after `fit` (or `fromJSON`). */
    get fitted(): boolean {
        return this.features >= 0;
    }

    /** Learns per-feature statistics from `x`. */
    abstract fit(x: MatrixLike): this;

    /** Scales `x` (same kind out as in: Matrix, rows, or one sample). */
    transform(x: Matrix): Matrix;
    transform(x: readonly (readonly number[])[]): number[][];
    transform(x: readonly number[]): number[];
    transform(x: MatrixLike): Matrix | number[][] | number[] {
        return this.apply(x, "transform", false);
    }

    /** `fit(x)` then `transform(x)`. */
    fitTransform(x: Matrix): Matrix;
    fitTransform(x: readonly (readonly number[])[]): number[][];
    fitTransform(x: readonly number[]): number[];
    fitTransform(x: MatrixLike): Matrix | number[][] | number[] {
        this.fit(x);
        return this.apply(x, "fitTransform", false);
    }

    /** Undoes `transform`. */
    inverseTransform(x: Matrix): Matrix;
    inverseTransform(x: readonly (readonly number[])[]): number[][];
    inverseTransform(x: readonly number[]): number[];
    inverseTransform(x: MatrixLike): Matrix | number[][] | number[] {
        return this.apply(x, "inverseTransform", true);
    }

    protected abstract forwardValue(value: number, feature: number): number;
    protected abstract inverseValue(value: number, feature: number): number;

    protected toMatrix(x: MatrixLike, where: string): Matrix {
        if (!(x instanceof Matrix) && !Array.isArray(x)) {
            throw new ValidationError(
                `${this.label}.${where}: expected number[][], number[] or a Matrix, got ${describeValue(x)}`,
            );
        }
        const m = Matrix.from(x);
        for (let i = 0; i < m.data.length; i++) {
            if (!Number.isFinite(m.data[i])) {
                throw new ValidationError(
                    `${this.label}.${where}: value ${m.data[i]} at index ${i} is not a finite number`,
                );
            }
        }
        return m;
    }

    private apply(x: MatrixLike, where: string, inverse: boolean): Matrix | number[][] | number[] {
        if (!this.fitted) throw new ValidationError(`${this.label}.${where}: call fit(x) first`);
        const m = this.toMatrix(x, where);
        if (m.rows > 0 && m.cols !== this.features) {
            throw new ShapeError(`${this.label} was fitted on ${this.features} features, got ${m.cols}`);
        }
        const out = new Matrix(m.rows, m.cols);
        const cols = m.cols;
        for (let i = 0; i < m.data.length; i++) {
            const feature = i % cols;
            out.data[i] = inverse ? this.inverseValue(m.data[i], feature) : this.forwardValue(m.data[i], feature);
        }
        if (x instanceof Matrix) return out;
        if (x.length > 0 && !Array.isArray(x[0])) return Array.from(out.data);
        return out.toArray();
    }
}

/**
 * Standardizes each feature to zero mean and unit variance: `(x − mean) / std`, with the
 * population standard deviation (features with zero variance are only centered).
 *
 * @example
 * const scaler = new StandardScaler();
 * const xTrainScaled = scaler.fitTransform(xTrain);
 * const xTestScaled = scaler.transform(xTest);
 * localStorage.scaler = JSON.stringify(scaler); // StandardScaler.fromJSON(JSON.parse(...)) restores it
 */
export class StandardScaler extends Scaler {
    protected readonly label = "StandardScaler";
    private meanValues = new Float64Array(0);
    private stdValues = new Float64Array(0);

    /** Per-feature means (a copy). */
    get mean(): number[] {
        return Array.from(this.meanValues);
    }

    /** Per-feature scales: the standard deviation, or 1 for constant features (a copy). */
    get std(): number[] {
        return Array.from(this.stdValues);
    }

    fit(x: MatrixLike): this {
        const m = this.toMatrix(x, "fit");
        if (m.rows === 0 || m.cols === 0) throw new ValidationError(`${this.label}.fit: x has no samples`);
        const n = m.cols;
        const mean = new Float64Array(n);
        const variance = new Float64Array(n);
        for (let r = 0; r < m.rows; r++) for (let c = 0; c < n; c++) mean[c] += m.data[r * n + c];
        for (let c = 0; c < n; c++) mean[c] /= m.rows;
        for (let r = 0; r < m.rows; r++) {
            for (let c = 0; c < n; c++) {
                const d = m.data[r * n + c] - mean[c];
                variance[c] += d * d;
            }
        }
        const std = new Float64Array(n);
        for (let c = 0; c < n; c++) {
            const s = Math.sqrt(variance[c] / m.rows);
            std[c] = s > 0 ? s : 1;
        }
        this.meanValues = mean;
        this.stdValues = std;
        this.features = n;
        return this;
    }

    protected forwardValue(value: number, feature: number): number {
        return (value - this.meanValues[feature]) / this.stdValues[feature];
    }

    protected inverseValue(value: number, feature: number): number {
        return value * this.stdValues[feature] + this.meanValues[feature];
    }

    toJSON(): StandardScalerJSON {
        if (!this.fitted) throw new ValidationError(`${this.label}.toJSON: call fit(x) first`);
        return { type: "standardScaler", mean: this.mean, std: this.std };
    }

    /** Restores a scaler saved with `toJSON()`. */
    static fromJSON(json: StandardScalerJSON): StandardScaler {
        const where = "StandardScaler.fromJSON";
        if (json === null || typeof json !== "object" || json.type !== "standardScaler") {
            throw new ValidationError(
                `${where}: expected { type: "standardScaler", mean, std }, got ${describeValue(json)}`,
            );
        }
        const mean = finiteArray(where, "mean", json.mean);
        const std = finiteArray(where, "std", json.std);
        if (mean.length !== std.length)
            throw new ValidationError(`${where}: mean and std lengths differ (${mean.length} vs ${std.length})`);
        if (std.some((s) => s <= 0)) throw new ValidationError(`${where}: every std must be > 0`);
        const scaler = new StandardScaler();
        scaler.meanValues = Float64Array.from(mean);
        scaler.stdValues = Float64Array.from(std);
        scaler.features = mean.length;
        return scaler;
    }
}

/** Options for {@link MinMaxScaler}. */
export interface MinMaxScalerOptions {
    /** Target range `[min, max]` with min < max. Default [0, 1]. */
    featureRange?: readonly [number, number];
}

/**
 * Rescales each feature linearly so the training data spans `featureRange` (default [0, 1]).
 * Constant features map to the range minimum.
 *
 * @example
 * const scaler = new MinMaxScaler({ featureRange: [-1, 1] });
 * const scaled = scaler.fitTransform([[0, 10], [5, 20], [10, 30]]); // [[-1, -1], [0, 0], [1, 1]]
 */
export class MinMaxScaler extends Scaler {
    protected readonly label = "MinMaxScaler";
    readonly featureRange: readonly [number, number];
    private minValues = new Float64Array(0);
    private maxValues = new Float64Array(0);

    constructor(options: MinMaxScalerOptions = {}) {
        super();
        checkOptions("MinMaxScaler", options, ["featureRange"]);
        const range = options.featureRange ?? [0, 1];
        if (
            !Array.isArray(range) ||
            range.length !== 2 ||
            !Number.isFinite(range[0]) ||
            !Number.isFinite(range[1]) ||
            range[0] >= range[1]
        ) {
            throw new ValidationError(
                `MinMaxScaler: "featureRange" must be [min, max] with min < max, got ${describeValue(range)}`,
            );
        }
        this.featureRange = [range[0], range[1]];
    }

    /** Per-feature minimum seen by `fit` (a copy). */
    get dataMin(): number[] {
        return Array.from(this.minValues);
    }

    /** Per-feature maximum seen by `fit` (a copy). */
    get dataMax(): number[] {
        return Array.from(this.maxValues);
    }

    fit(x: MatrixLike): this {
        const m = this.toMatrix(x, "fit");
        if (m.rows === 0 || m.cols === 0) throw new ValidationError(`${this.label}.fit: x has no samples`);
        const n = m.cols;
        const min = new Float64Array(n).fill(Infinity);
        const max = new Float64Array(n).fill(-Infinity);
        for (let r = 0; r < m.rows; r++) {
            for (let c = 0; c < n; c++) {
                const v = m.data[r * n + c];
                if (v < min[c]) min[c] = v;
                if (v > max[c]) max[c] = v;
            }
        }
        this.minValues = min;
        this.maxValues = max;
        this.features = n;
        return this;
    }

    protected forwardValue(value: number, feature: number): number {
        const [lo, hi] = this.featureRange;
        const span = this.maxValues[feature] - this.minValues[feature];
        return lo + ((value - this.minValues[feature]) * (hi - lo)) / (span > 0 ? span : 1);
    }

    protected inverseValue(value: number, feature: number): number {
        const [lo, hi] = this.featureRange;
        const span = this.maxValues[feature] - this.minValues[feature];
        return this.minValues[feature] + ((value - lo) * (span > 0 ? span : 1)) / (hi - lo);
    }

    toJSON(): MinMaxScalerJSON {
        if (!this.fitted) throw new ValidationError(`${this.label}.toJSON: call fit(x) first`);
        return {
            type: "minMaxScaler",
            featureRange: [this.featureRange[0], this.featureRange[1]],
            dataMin: this.dataMin,
            dataMax: this.dataMax,
        };
    }

    /** Restores a scaler saved with `toJSON()`. */
    static fromJSON(json: MinMaxScalerJSON): MinMaxScaler {
        const where = "MinMaxScaler.fromJSON";
        if (json === null || typeof json !== "object" || json.type !== "minMaxScaler") {
            throw new ValidationError(
                `${where}: expected { type: "minMaxScaler", featureRange, dataMin, dataMax }, got ${describeValue(json)}`,
            );
        }
        const dataMin = finiteArray(where, "dataMin", json.dataMin);
        const dataMax = finiteArray(where, "dataMax", json.dataMax);
        if (dataMin.length !== dataMax.length) {
            throw new ValidationError(
                `${where}: dataMin and dataMax lengths differ (${dataMin.length} vs ${dataMax.length})`,
            );
        }
        const scaler = new MinMaxScaler({ featureRange: json.featureRange });
        scaler.minValues = Float64Array.from(dataMin);
        scaler.maxValues = Float64Array.from(dataMax);
        scaler.features = dataMin.length;
        return scaler;
    }
}

function finiteArray(where: string, key: string, value: unknown): number[] {
    if (!Array.isArray(value) || value.some((v) => typeof v !== "number" || !Number.isFinite(v))) {
        throw new ValidationError(`${where}: "${key}" must be an array of finite numbers, got ${describeValue(value)}`);
    }
    return value as number[];
}
