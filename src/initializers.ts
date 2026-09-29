/**
 * Weight initializers (Keras-compatible formulas and defaults).
 *
 * Variance-scaling initializers draw with variance `scale / n`, where n is fanIn (he, lecun) or
 * (fanIn + fanOut) / 2 (glorot), floored at 1. Uniform variants sample U(-limit, limit) with
 * limit = sqrt(3·variance); normal variants sample a normal truncated at ±2σ whose σ is divided
 * by 0.87962566103423978 so the truncated distribution still has the requested variance.
 */

import { ValidationError } from "./core/errors.js";
import type { Matrix } from "./core/matrix.js";
import type { Random } from "./core/random.js";
import type {
    Initializer,
    InitializerConfig,
    InitializerIdentifier,
    InitializerName,
    JsonValue,
} from "./core/types.js";

/** Every built-in initializer name. */
export const INITIALIZER_NAMES: readonly InitializerName[] = Object.freeze([
    "zeros",
    "ones",
    "constant",
    "randomUniform",
    "randomNormal",
    "glorotUniform",
    "glorotNormal",
    "heUniform",
    "heNormal",
    "lecunUniform",
    "lecunNormal",
] as const);

/** Standard deviation of a unit normal truncated to [-2, 2]. */
const TRUNCATED_NORMAL_STDDEV = 0.87962566103423978;

function checkFans(name: InitializerName, fanIn: number, fanOut: number): void {
    if (!Number.isFinite(fanIn) || fanIn < 0 || !Number.isFinite(fanOut) || fanOut < 0) {
        throw new ValidationError(
            `Initializer "${name}": fanIn and fanOut must be non-negative finite numbers, got ${fanIn} and ${fanOut}`,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Initializers
// ---------------------------------------------------------------------------------------------

/** Fills every element with one value (zeros, ones, constant). */
class ConstantInitializer implements Initializer {
    constructor(
        readonly name: "zeros" | "ones" | "constant",
        readonly value: number,
    ) {}

    initialize(target: Matrix, fanIn: number, fanOut: number, _rng: Random): void {
        checkFans(this.name, fanIn, fanOut);
        target.data.fill(this.value);
    }

    getConfig(): InitializerConfig {
        return this.name === "constant" ? { name: this.name, value: this.value } : { name: this.name };
    }
}

class RandomUniform implements Initializer {
    readonly name = "randomUniform" as const;

    constructor(
        readonly minval: number,
        readonly maxval: number,
    ) {}

    initialize(target: Matrix, fanIn: number, fanOut: number, rng: Random): void {
        checkFans(this.name, fanIn, fanOut);
        const data = target.data;
        const { minval, maxval } = this;
        for (let i = 0; i < data.length; i++) data[i] = rng.uniform(minval, maxval);
    }

    getConfig(): InitializerConfig {
        return { name: this.name, minval: this.minval, maxval: this.maxval };
    }
}

class RandomNormal implements Initializer {
    readonly name = "randomNormal" as const;

    constructor(
        readonly mean: number,
        readonly stddev: number,
    ) {}

    initialize(target: Matrix, fanIn: number, fanOut: number, rng: Random): void {
        checkFans(this.name, fanIn, fanOut);
        const data = target.data;
        const { mean, stddev } = this;
        for (let i = 0; i < data.length; i++) data[i] = rng.normal(mean, stddev);
    }

    getConfig(): InitializerConfig {
        return { name: this.name, mean: this.mean, stddev: this.stddev };
    }
}

type VarianceScalingName = "glorotUniform" | "glorotNormal" | "heUniform" | "heNormal" | "lecunUniform" | "lecunNormal";

class VarianceScaling implements Initializer {
    constructor(
        readonly name: VarianceScalingName,
        private readonly scale: number,
        private readonly mode: "fanIn" | "fanAvg",
        private readonly distribution: "uniform" | "truncatedNormal",
    ) {}

    initialize(target: Matrix, fanIn: number, fanOut: number, rng: Random): void {
        checkFans(this.name, fanIn, fanOut);
        const n = Math.max(1, this.mode === "fanIn" ? fanIn : (fanIn + fanOut) / 2);
        const variance = this.scale / n;
        const data = target.data;
        if (this.distribution === "uniform") {
            const limit = Math.sqrt(3 * variance);
            for (let i = 0; i < data.length; i++) data[i] = rng.uniform(-limit, limit);
        } else {
            const stddev = Math.sqrt(variance) / TRUNCATED_NORMAL_STDDEV;
            for (let i = 0; i < data.length; i++) data[i] = rng.truncatedNormal(0, stddev);
        }
    }

    getConfig(): InitializerConfig {
        return { name: this.name };
    }
}

// ---------------------------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------------------------

type ConfigParams = { readonly [param: string]: JsonValue | undefined };

const ACCEPTED_PARAMS: Record<InitializerName, readonly string[]> = {
    zeros: [],
    ones: [],
    constant: ["value"],
    randomUniform: ["minval", "maxval"],
    randomNormal: ["mean", "stddev"],
    glorotUniform: [],
    glorotNormal: [],
    heUniform: [],
    heNormal: [],
    lecunUniform: [],
    lecunNormal: [],
};

function isInitializerName(value: unknown): value is InitializerName {
    return typeof value === "string" && Object.hasOwn(ACCEPTED_PARAMS, value);
}

function unknownName(value: unknown): ValidationError {
    return new ValidationError(
        `Unknown initializer ${JSON.stringify(value)}. Valid initializers: ${INITIALIZER_NAMES.join(", ")}`,
    );
}

function numberParam(params: ConfigParams, key: string, fallback: number, name: InitializerName): number {
    const value = params[key];
    if (value === undefined) return fallback;
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new ValidationError(
            `Initializer "${name}": "${key}" must be a finite number, got ${JSON.stringify(value)}`,
        );
    }
    return value;
}

function create(name: InitializerName, params: ConfigParams): Initializer {
    const accepted = ACCEPTED_PARAMS[name];
    for (const key of Object.keys(params)) {
        if (key === "name" || params[key] === undefined || accepted.includes(key)) continue;
        const hint = accepted.length > 0 ? `accepted: ${accepted.join(", ")}` : "it takes no parameters";
        throw new ValidationError(`Initializer "${name}": unknown parameter "${key}" (${hint})`);
    }
    switch (name) {
        case "zeros":
            return new ConstantInitializer("zeros", 0);
        case "ones":
            return new ConstantInitializer("ones", 1);
        case "constant":
            return new ConstantInitializer("constant", numberParam(params, "value", 0, name));
        case "randomUniform": {
            const minval = numberParam(params, "minval", -0.05, name);
            const maxval = numberParam(params, "maxval", 0.05, name);
            if (minval > maxval) {
                throw new ValidationError(
                    `Initializer "randomUniform": minval (${minval}) must not exceed maxval (${maxval})`,
                );
            }
            return new RandomUniform(minval, maxval);
        }
        case "randomNormal": {
            const mean = numberParam(params, "mean", 0, name);
            const stddev = numberParam(params, "stddev", 0.05, name);
            if (stddev < 0) throw new ValidationError(`Initializer "randomNormal": stddev must be >= 0, got ${stddev}`);
            return new RandomNormal(mean, stddev);
        }
        case "glorotUniform":
            return new VarianceScaling(name, 1, "fanAvg", "uniform");
        case "glorotNormal":
            return new VarianceScaling(name, 1, "fanAvg", "truncatedNormal");
        case "heUniform":
            return new VarianceScaling(name, 2, "fanIn", "uniform");
        case "heNormal":
            return new VarianceScaling(name, 2, "fanIn", "truncatedNormal");
        case "lecunUniform":
            return new VarianceScaling(name, 1, "fanIn", "uniform");
        case "lecunNormal":
            return new VarianceScaling(name, 1, "fanIn", "truncatedNormal");
    }
}

/**
 * Resolves an initializer from a name (`"heNormal"`), a config (`{ name: "constant", value: 0.1 }`),
 * or an existing `Initializer` instance (returned as-is).
 *
 * Defaults (Keras): constant `value` = 0; randomUniform `minval` = -0.05, `maxval` = 0.05;
 * randomNormal `mean` = 0, `stddev` = 0.05.
 * @throws ValidationError for unknown names, unknown parameters, or invalid parameter values.
 */
export function getInitializer(id: InitializerIdentifier): Initializer {
    if (typeof id === "string") {
        if (!isInitializerName(id)) throw unknownName(id);
        return create(id, {});
    }
    if (typeof id === "object" && id !== null) {
        if (typeof (id as Partial<Initializer>).initialize === "function") return id as Initializer;
        const config = id as InitializerConfig;
        if (!isInitializerName(config.name)) throw unknownName(config.name);
        return create(config.name, config);
    }
    throw new ValidationError(
        `Invalid initializer identifier ${String(id)}: expected a name, a config object, or an Initializer`,
    );
}
