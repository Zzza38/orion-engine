/**
 * Small internal helpers shared by the layers, the model, callbacks and data utilities.
 * Not part of the public API.
 */
import { ValidationError } from "./core/errors.js";

/** Formats a value for an error message: strings are quoted, everything else is `String()`-ed. */
export function describeValue(value: unknown): string {
    if (typeof value === "string") return JSON.stringify(value);
    if (Array.isArray(value)) return `[${value.length > 8 ? `${value.slice(0, 8).join(", ")}, …` : value.join(", ")}]`;
    return String(value);
}

/** Throws unless `options` is a plain object (or undefined) whose keys are all in `allowed`. */
export function checkOptions(where: string, options: unknown, allowed: readonly string[]): void {
    if (options === undefined) return;
    if (options === null || typeof options !== "object" || Array.isArray(options)) {
        throw new ValidationError(`${where}: options must be an object, got ${describeValue(options)}`);
    }
    for (const key of Object.keys(options)) {
        if (!allowed.includes(key)) {
            throw new ValidationError(`${where}: unknown option "${key}". Valid options: ${allowed.join(", ")}`);
        }
    }
}

/** Returns `value` if it is a positive integer, `fallback` if undefined, and throws otherwise. */
export function positiveInteger(where: string, key: string, value: unknown, fallback?: number): number {
    if (value === undefined && fallback !== undefined) return fallback;
    if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
        throw new ValidationError(`${where}: "${key}" must be a positive integer, got ${describeValue(value)}`);
    }
    return value;
}

/** Returns `value` if it is a non-negative integer, `fallback` if undefined, and throws otherwise. */
export function nonNegativeInteger(where: string, key: string, value: unknown, fallback?: number): number {
    if (value === undefined && fallback !== undefined) return fallback;
    if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
        throw new ValidationError(`${where}: "${key}" must be a non-negative integer, got ${describeValue(value)}`);
    }
    return value;
}

/** Returns a finite number satisfying `test`, `fallback` if undefined, and throws otherwise. */
export function numberOption(
    where: string,
    key: string,
    value: unknown,
    fallback: number,
    requirement = "a finite number",
    test: (v: number) => boolean = () => true,
): number {
    if (value === undefined) return fallback;
    if (typeof value !== "number" || !Number.isFinite(value) || !test(value)) {
        throw new ValidationError(`${where}: "${key}" must be ${requirement}, got ${describeValue(value)}`);
    }
    return value;
}

/** Returns a boolean, `fallback` if undefined, and throws otherwise. */
export function booleanOption(where: string, key: string, value: unknown, fallback: boolean): boolean {
    if (value === undefined) return fallback;
    if (typeof value !== "boolean") {
        throw new ValidationError(`${where}: "${key}" must be a boolean, got ${describeValue(value)}`);
    }
    return value;
}

/** True for promises and other thenables. */
export function isThenable(value: unknown): value is PromiseLike<unknown> {
    return (
        value !== null &&
        (typeof value === "object" || typeof value === "function") &&
        typeof (value as { then?: unknown }).then === "function"
    );
}

/** High-resolution timestamp in milliseconds (falls back to Date.now where `performance` is missing). */
export function now(): number {
    return typeof performance !== "undefined" ? performance.now() : Date.now();
}

/** "batchNormalization" -> "batch_normalization". */
export function snakeCase(value: string): string {
    return value.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
}

/** Formats an integer with thousands separators independent of the host locale: 101770 -> "101,770". */
export function formatCount(value: number): string {
    return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
