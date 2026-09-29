import { ShapeError, ValidationError } from "./errors.js";

/** Anything that can be turned into a 2-D matrix: a Matrix, rows of numbers, or a single row. */
export type MatrixLike = Matrix | readonly (readonly number[])[] | readonly number[];

/**
 * Dense, row-major 2-D matrix backed by a Float64Array.
 * Rows are samples (the batch dimension); columns are features/units.
 */
export class Matrix {
    readonly rows: number;
    readonly cols: number;
    readonly data: Float64Array;

    constructor(rows: number, cols: number, data?: Float64Array | ArrayLike<number>) {
        if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 0 || cols < 0) {
            throw new ShapeError(`Invalid matrix shape [${rows}, ${cols}]`);
        }
        this.rows = rows;
        this.cols = cols;
        if (data === undefined) {
            this.data = new Float64Array(rows * cols);
        } else {
            if (data.length !== rows * cols) {
                throw new ShapeError(`Data length ${data.length} does not match shape [${rows}, ${cols}]`);
            }
            this.data = data instanceof Float64Array ? data : Float64Array.from(data);
        }
    }

    static zeros(rows: number, cols: number): Matrix {
        return new Matrix(rows, cols);
    }

    static filled(rows: number, cols: number, value: number): Matrix {
        const m = new Matrix(rows, cols);
        m.data.fill(value);
        return m;
    }

    /** Builds a matrix from rows of numbers. Throws on ragged or non-finite-number input. */
    static fromArray(rows: readonly (readonly number[])[]): Matrix {
        const rowCount = rows.length;
        const colCount = rowCount === 0 ? 0 : rows[0].length;
        const m = new Matrix(rowCount, colCount);
        for (let r = 0; r < rowCount; r++) {
            const row = rows[r];
            if (!row || row.length !== colCount) {
                throw new ShapeError(`Ragged input: row ${r} has length ${row?.length ?? 0}, expected ${colCount}`);
            }
            for (let c = 0; c < colCount; c++) {
                const v = row[c];
                if (typeof v !== "number" || Number.isNaN(v)) {
                    throw new ValidationError(`Non-numeric value at [${r}, ${c}]: ${String(v)}`);
                }
                m.data[r * colCount + c] = v;
            }
        }
        return m;
    }

    /** Builds a 1 x n row matrix from a vector. */
    static fromVector(values: readonly number[] | Float64Array): Matrix {
        return new Matrix(1, values.length, Float64Array.from(values));
    }

    /**
     * Normalizes any MatrixLike into a Matrix. A flat number[] becomes a single row.
     * Matrices are returned as-is (not copied).
     */
    static from(value: MatrixLike): Matrix {
        if (value instanceof Matrix) return value;
        if (value.length === 0) return new Matrix(0, 0);
        if (Array.isArray(value[0])) return Matrix.fromArray(value as readonly (readonly number[])[]);
        return Matrix.fromVector(value as readonly number[]);
    }

    get size(): number {
        return this.data.length;
    }

    get shape(): [number, number] {
        return [this.rows, this.cols];
    }

    get(row: number, col: number): number {
        return this.data[row * this.cols + col];
    }

    set(row: number, col: number, value: number): void {
        this.data[row * this.cols + col] = value;
    }

    /** A live view (no copy) of one row. */
    row(index: number): Float64Array {
        return this.data.subarray(index * this.cols, (index + 1) * this.cols);
    }

    toArray(): number[][] {
        const out: number[][] = new Array(this.rows);
        for (let r = 0; r < this.rows; r++) out[r] = Array.from(this.row(r));
        return out;
    }

    clone(): Matrix {
        return new Matrix(this.rows, this.cols, this.data.slice());
    }

    fill(value: number): this {
        this.data.fill(value);
        return this;
    }

    copyFrom(other: Matrix): this {
        assertSameShape(this, other, "copyFrom");
        this.data.set(other.data);
        return this;
    }

    map(fn: (value: number, row: number, col: number) => number, out?: Matrix): Matrix {
        const target = prepareOut(out, this.rows, this.cols, "map");
        const { cols, data } = this;
        for (let i = 0; i < data.length; i++) target.data[i] = fn(data[i], (i / cols) | 0, i % cols);
        return target;
    }

    hasShape(rows: number, cols: number): boolean {
        return this.rows === rows && this.cols === cols;
    }

    toString(): string {
        return `Matrix[${this.rows}x${this.cols}]`;
    }
}

function prepareOut(out: Matrix | undefined, rows: number, cols: number, op: string): Matrix {
    if (!out) return new Matrix(rows, cols);
    if (out.rows !== rows || out.cols !== cols) {
        throw new ShapeError(`${op}: output buffer is [${out.rows}, ${out.cols}], expected [${rows}, ${cols}]`);
    }
    return out;
}

export function assertSameShape(a: Matrix, b: Matrix, op: string): void {
    if (a.rows !== b.rows || a.cols !== b.cols) {
        throw new ShapeError(`${op}: shape mismatch [${a.rows}, ${a.cols}] vs [${b.rows}, ${b.cols}]`);
    }
}

/** out = a · b, where a is [m, k] and b is [k, n]. */
export function matmul(a: Matrix, b: Matrix, out?: Matrix): Matrix {
    if (a.cols !== b.rows) throw new ShapeError(`matmul: [${a.rows}, ${a.cols}] · [${b.rows}, ${b.cols}]`);
    const m = a.rows, k = a.cols, n = b.cols;
    const target = prepareOut(out, m, n, "matmul");
    const A = a.data, B = b.data, C = target.data;
    C.fill(0);
    for (let i = 0; i < m; i++) {
        const cRow = i * n;
        for (let p = 0; p < k; p++) {
            const av = A[i * k + p];
            if (av === 0) continue;
            const bRow = p * n;
            for (let j = 0; j < n; j++) C[cRow + j] += av * B[bRow + j];
        }
    }
    return target;
}

/** out = aᵀ · b, where a is [k, m] and b is [k, n]. Used for weight gradients (Xᵀ · dY). */
export function matmulTransposeA(a: Matrix, b: Matrix, out?: Matrix): Matrix {
    if (a.rows !== b.rows) throw new ShapeError(`matmulTransposeA: [${a.rows}, ${a.cols}]ᵀ · [${b.rows}, ${b.cols}]`);
    const k = a.rows, m = a.cols, n = b.cols;
    const target = prepareOut(out, m, n, "matmulTransposeA");
    const A = a.data, B = b.data, C = target.data;
    C.fill(0);
    for (let p = 0; p < k; p++) {
        const aRow = p * m, bRow = p * n;
        for (let i = 0; i < m; i++) {
            const av = A[aRow + i];
            if (av === 0) continue;
            const cRow = i * n;
            for (let j = 0; j < n; j++) C[cRow + j] += av * B[bRow + j];
        }
    }
    return target;
}

/** out = a · bᵀ, where a is [m, k] and b is [n, k]. Used for input gradients (dY · Wᵀ). */
export function matmulTransposeB(a: Matrix, b: Matrix, out?: Matrix): Matrix {
    if (a.cols !== b.cols) throw new ShapeError(`matmulTransposeB: [${a.rows}, ${a.cols}] · [${b.rows}, ${b.cols}]ᵀ`);
    const m = a.rows, k = a.cols, n = b.rows;
    const target = prepareOut(out, m, n, "matmulTransposeB");
    const A = a.data, B = b.data, C = target.data;
    for (let i = 0; i < m; i++) {
        const aRow = i * k;
        for (let j = 0; j < n; j++) {
            const bRow = j * k;
            let sum = 0;
            for (let p = 0; p < k; p++) sum += A[aRow + p] * B[bRow + p];
            C[i * n + j] = sum;
        }
    }
    return target;
}

/** out = aᵀ. */
export function transpose(a: Matrix, out?: Matrix): Matrix {
    const target = prepareOut(out, a.cols, a.rows, "transpose");
    for (let r = 0; r < a.rows; r++) {
        for (let c = 0; c < a.cols; c++) target.data[c * a.rows + r] = a.data[r * a.cols + c];
    }
    return target;
}

/** out = a + b (element-wise). */
export function add(a: Matrix, b: Matrix, out?: Matrix): Matrix {
    assertSameShape(a, b, "add");
    const target = prepareOut(out, a.rows, a.cols, "add");
    for (let i = 0; i < a.data.length; i++) target.data[i] = a.data[i] + b.data[i];
    return target;
}

/** out = a - b (element-wise). */
export function subtract(a: Matrix, b: Matrix, out?: Matrix): Matrix {
    assertSameShape(a, b, "subtract");
    const target = prepareOut(out, a.rows, a.cols, "subtract");
    for (let i = 0; i < a.data.length; i++) target.data[i] = a.data[i] - b.data[i];
    return target;
}

/** out = a ⊙ b (element-wise / Hadamard product). */
export function multiply(a: Matrix, b: Matrix, out?: Matrix): Matrix {
    assertSameShape(a, b, "multiply");
    const target = prepareOut(out, a.rows, a.cols, "multiply");
    for (let i = 0; i < a.data.length; i++) target.data[i] = a.data[i] * b.data[i];
    return target;
}

/** out = a * scalar. */
export function scale(a: Matrix, scalar: number, out?: Matrix): Matrix {
    const target = prepareOut(out, a.rows, a.cols, "scale");
    for (let i = 0; i < a.data.length; i++) target.data[i] = a.data[i] * scalar;
    return target;
}

/** out[r, c] = a[r, c] + v[c]. Broadcasts a row vector (Matrix [1, cols] or Float64Array) over every row. */
export function addRowVector(a: Matrix, v: Matrix | Float64Array, out?: Matrix): Matrix {
    const vec = v instanceof Matrix ? v.data : v;
    if (vec.length !== a.cols) throw new ShapeError(`addRowVector: vector length ${vec.length} vs ${a.cols} columns`);
    const target = prepareOut(out, a.rows, a.cols, "addRowVector");
    const cols = a.cols;
    for (let r = 0; r < a.rows; r++) {
        const base = r * cols;
        for (let c = 0; c < cols; c++) target.data[base + c] = a.data[base + c] + vec[c];
    }
    return target;
}

/** Sums over rows (the batch axis), producing a [1, cols] matrix. Used for bias gradients. */
export function sumRows(a: Matrix, out?: Matrix): Matrix {
    const target = prepareOut(out, 1, a.cols, "sumRows");
    target.data.fill(0);
    const cols = a.cols;
    for (let r = 0; r < a.rows; r++) {
        const base = r * cols;
        for (let c = 0; c < cols; c++) target.data[c] += a.data[base + c];
    }
    return target;
}

/** Gathers the given rows (in order) into a new [indices.length, cols] matrix. */
export function gatherRows(a: Matrix, indices: ArrayLike<number>, out?: Matrix): Matrix {
    const target = prepareOut(out, indices.length, a.cols, "gatherRows");
    const cols = a.cols;
    for (let i = 0; i < indices.length; i++) {
        const src = indices[i];
        if (src < 0 || src >= a.rows) throw new ShapeError(`gatherRows: row ${src} out of range [0, ${a.rows})`);
        target.data.set(a.data.subarray(src * cols, (src + 1) * cols), i * cols);
    }
    return target;
}

/** Copies rows [start, end) into a new matrix. */
export function sliceRows(a: Matrix, start: number, end: number): Matrix {
    const s = Math.max(0, start), e = Math.min(a.rows, end);
    return new Matrix(Math.max(0, e - s), a.cols, a.data.slice(s * a.cols, e * a.cols));
}

/** Index of the largest value in each row. */
export function argmaxRows(a: Matrix): Int32Array {
    const out = new Int32Array(a.rows);
    for (let r = 0; r < a.rows; r++) {
        const base = r * a.cols;
        let best = 0;
        let bestVal = -Infinity;
        for (let c = 0; c < a.cols; c++) {
            const v = a.data[base + c];
            if (v > bestVal) {
                bestVal = v;
                best = c;
            }
        }
        out[r] = best;
    }
    return out;
}
