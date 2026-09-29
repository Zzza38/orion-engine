/**
 * Shared machinery for the built-in layers: naming, build bookkeeping, input validation,
 * parameters whose names follow their layer, and per-batch-size buffer reuse.
 */
import { ShapeError, ValidationError } from "../core/errors.js";
import { Matrix } from "../core/matrix.js";
import type { Random } from "../core/random.js";
import type { Layer, LayerConfig, Parameter } from "../core/types.js";
import { describeValue } from "../utils.js";

/** Options every built-in layer accepts. */
export interface LayerOptions {
    /**
     * Layer name, unique within a model. Leave it out to let the model assign one
     * (`dense_1`, `dense_2`, `dropout_1`, …) when the layer is added. Must not contain "/".
     */
    name?: string;
}

/**
 * A layer parameter whose name is derived from its owning layer (`"<layerName>/<key>"`), so it
 * stays correct when a model assigns the layer's name after construction.
 */
export class LayerParameter implements Parameter {
    readonly value: Matrix;
    readonly grad: Matrix;
    trainable: boolean;
    readonly regularize: boolean;

    constructor(
        private readonly owner: { readonly name: string },
        /** Parameter key within the layer, e.g. "kernel". */
        readonly key: string,
        rows: number,
        cols: number,
        options: { trainable?: boolean; regularize?: boolean } = {},
    ) {
        this.value = new Matrix(rows, cols);
        this.grad = new Matrix(rows, cols);
        this.trainable = options.trainable ?? true;
        this.regularize = options.regularize ?? true;
    }

    get name(): string {
        return `${this.owner.name}/${this.key}`;
    }
}

/**
 * Caches scratch buffers keyed by batch size so steady-state training allocates nothing.
 * Keeps the most recently used `capacity` sizes (a training run typically needs 2–3: the batch
 * size, the last partial batch, and the validation tail).
 */
export class BufferCache<T> {
    private readonly entries = new Map<number, T>();
    private lastRows = -1;
    private last: T | undefined;

    constructor(
        private readonly create: (rows: number) => T,
        private readonly capacity = 4,
    ) {}

    get(rows: number): T {
        if (rows === this.lastRows) return this.last as T;
        let entry = this.entries.get(rows);
        if (entry === undefined) {
            entry = this.create(rows);
            if (this.entries.size >= this.capacity) {
                const oldest = this.entries.keys().next().value as number;
                this.entries.delete(oldest);
            }
        } else {
            this.entries.delete(rows); // re-insert as most recently used
        }
        this.entries.set(rows, entry);
        this.lastRows = rows;
        this.last = entry;
        return entry;
    }

    clear(): void {
        this.entries.clear();
        this.lastRows = -1;
        this.last = undefined;
    }
}

/** Validates a user-supplied layer name. Empty means "let the model name it". */
export function checkLayerName(where: string, name: unknown): string {
    if (name === undefined) return "";
    if (typeof name !== "string") throw new ValidationError(`${where}: "name" must be a string, got ${describeValue(name)}`);
    if (name.includes("/")) {
        throw new ValidationError(`${where}: layer name ${JSON.stringify(name)} must not contain "/" (used in weight names)`);
    }
    return name;
}

/** Throws a ValidationError for config keys a layer does not understand. */
export function checkConfigKeys(type: string, config: LayerConfig, allowed: readonly string[]): void {
    for (const key of Object.keys(config)) {
        if (key === "type" || key === "name" || config[key] === undefined || allowed.includes(key)) continue;
        throw new ValidationError(
            `Layer config for "${type}" has unknown key "${key}". Valid keys: type, name, ${allowed.join(", ")}`,
        );
    }
}

/**
 * Base class for the built-in layers. Subclasses implement {@link BaseLayer.onBuild},
 * `forward`, {@link BaseLayer.propagate} and `getConfig`.
 *
 * Matrices returned by `forward`/`backward` are scratch buffers owned by the layer: they are
 * overwritten by the next call with the same batch size. Copy them if you need to keep them.
 */
export abstract class BaseLayer implements Layer {
    abstract readonly type: string;
    name: string;
    private isBuilt = false;
    private builtInputSize = 0;

    protected constructor(name: string) {
        this.name = name;
    }

    get built(): boolean {
        return this.isBuilt;
    }

    /** Input feature count, or undefined before `build`. */
    get inputSize(): number | undefined {
        return this.isBuilt ? this.builtInputSize : undefined;
    }

    /** Output feature count. Only meaningful after `build` (Dense knows it earlier: its `units`). */
    get outputSize(): number {
        return this.builtInputSize;
    }

    /**
     * Allocates parameters and buffers for `inputSize` input features, drawing any randomness
     * from `rng`. Calling it again with the same size is a no-op.
     */
    build(inputSize: number, rng: Random): void {
        if (!Number.isInteger(inputSize) || inputSize < 1) {
            throw new ValidationError(`${this.label}: input size must be a positive integer, got ${describeValue(inputSize)}`);
        }
        if (this.isBuilt) {
            if (inputSize === this.builtInputSize) return;
            throw new ShapeError(
                `${this.label} is already built for ${this.builtInputSize} input features and cannot be rebuilt ` +
                    `for ${inputSize}; create a new layer instead`,
            );
        }
        this.onBuild(inputSize, rng);
        this.builtInputSize = inputSize;
        this.isBuilt = true;
    }

    abstract forward(input: Matrix, training: boolean): Matrix;

    /** dL/dInput for the last forward batch. Also writes (overwrites) every parameter's gradient. */
    backward(gradOutput: Matrix): Matrix {
        return this.propagate(gradOutput, true) as Matrix;
    }

    /**
     * Like {@link BaseLayer.backward}, but when `inputGradient` is false the layer may skip
     * computing dL/dInput and return null. Models use this for their first layer.
     */
    abstract propagate(gradOutput: Matrix, inputGradient: boolean): Matrix | null;

    parameters(): Parameter[] {
        return [];
    }

    abstract getConfig(): LayerConfig;

    /** Called once by `build`. */
    protected abstract onBuild(inputSize: number, rng: Random): void;

    /** `Dense "dense_1"`-style label for error messages. */
    protected get label(): string {
        const type = this.type.charAt(0).toUpperCase() + this.type.slice(1);
        return this.name ? `${type} layer "${this.name}"` : `${type} layer`;
    }

    /** Throws unless the layer is built and `input` has the expected number of features. */
    protected checkInput(input: Matrix): void {
        if (!this.isBuilt) {
            throw new ValidationError(
                `${this.label} is not built: add it to a model that knows its inputSize, or call build(inputSize, rng)`,
            );
        }
        if (!(input instanceof Matrix)) {
            throw new ValidationError(`${this.label}: forward() expects a Matrix, got ${describeValue(input)}`);
        }
        if (input.cols !== this.builtInputSize) {
            throw new ShapeError(`${this.label} expected input with ${this.builtInputSize} features, got ${input.cols}`);
        }
    }

    /** Throws unless `gradOutput` matches the last forward output shape. */
    protected checkGradient(gradOutput: Matrix, rows: number, cols: number): void {
        if (rows < 0) throw new ValidationError(`${this.label}: backward() called before forward()`);
        if (gradOutput.rows !== rows || gradOutput.cols !== cols) {
            throw new ShapeError(
                `${this.label}: gradient is [${gradOutput.rows}, ${gradOutput.cols}], expected [${rows}, ${cols}] ` +
                    "(the shape of the last forward output)",
            );
        }
    }
}

/** True for layers that support skipping the input gradient (all built-in layers). */
export function isBaseLayer(layer: Layer): layer is BaseLayer {
    return layer instanceof BaseLayer;
}
