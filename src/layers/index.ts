/**
 * Built-in layers, their factory helpers, and the config → layer registry used when loading
 * saved models.
 */
import { ACTIVATION_NAMES } from "../activations.js";
import { ValidationError } from "../core/errors.js";
import type { ActivationIdentifier, Layer, LayerConfig } from "../core/types.js";
import { describeValue } from "../utils.js";
import type { ActivationLayerOptions } from "./activation.js";
import { ActivationLayer } from "./activation.js";
import type { LayerOptions } from "./base.js";
import type { BatchNormalizationOptions } from "./batchNormalization.js";
import { BatchNormalization } from "./batchNormalization.js";
import type { DenseOptions } from "./dense.js";
import { Dense } from "./dense.js";
import { Dropout } from "./dropout.js";

export type { ActivationLayerOptions } from "./activation.js";
export { ActivationLayer } from "./activation.js";
export type { LayerOptions } from "./base.js";
export { BaseLayer, LayerParameter } from "./base.js";
export type { BatchNormalizationOptions } from "./batchNormalization.js";
export { BatchNormalization } from "./batchNormalization.js";
export type { DenseOptions, RegularizerOptions } from "./dense.js";
export { Dense } from "./dense.js";
export type { DropoutOptions } from "./dropout.js";
export { Dropout } from "./dropout.js";

/** Builds a layer from its `getConfig()` output. */
export type LayerFromConfig = (config: LayerConfig) => Layer;

/** Layer types that ship with the library. */
export const LAYER_TYPES: readonly string[] = Object.freeze(["dense", "dropout", "batchNormalization", "activation"]);

const REGISTRY = new Map<string, LayerFromConfig>([
    ["dense", Dense.fromConfig],
    ["dropout", Dropout.fromConfig],
    ["batchNormalization", BatchNormalization.fromConfig],
    ["activation", ActivationLayer.fromConfig],
]);

/** Dense-specific option keys: their presence marks the second `dense()` argument as options. */
const DENSE_KEYS = ["units", "activation", "useBias", "kernelInitializer", "biasInitializer", "kernelRegularizer"];

/**
 * Creates a {@link Dense} layer.
 *
 * The second argument is either the activation (a name, a config such as
 * `{ name: "leakyRelu", alpha: 0.2 }`, or an `Activation`) or the remaining {@link DenseOptions}.
 *
 * @example
 * dense(8, "relu");
 * dense(1, { activation: "sigmoid", name: "output" });
 * dense(64, { activation: "relu", kernelInitializer: "heNormal", kernelRegularizer: { l2: 1e-4 } });
 */
export function dense(units: number, activationOrOptions?: ActivationIdentifier | Omit<DenseOptions, "units">): Dense {
    if (activationOrOptions === undefined || typeof activationOrOptions === "string") {
        return new Dense({ units, activation: activationOrOptions });
    }
    if (typeof activationOrOptions !== "object" || activationOrOptions === null) {
        throw new ValidationError(
            `dense(${describeValue(units)}, …): second argument must be an activation or an options object, ` +
                `got ${describeValue(activationOrOptions)}`,
        );
    }
    if (isDenseOptions(activationOrOptions)) {
        if ("units" in activationOrOptions) {
            throw new ValidationError(
                `dense(): pass "units" as the first argument only, e.g. dense(8, { activation: "relu" })`,
            );
        }
        return new Dense({ ...(activationOrOptions as Omit<DenseOptions, "units">), units });
    }
    return new Dense({ units, activation: activationOrOptions as ActivationIdentifier });
}

function isDenseOptions(value: object): boolean {
    if (typeof (value as { forward?: unknown }).forward === "function") return false; // an Activation
    if (Object.keys(value).some((key) => DENSE_KEYS.includes(key))) return true;
    const name = (value as { name?: unknown }).name;
    // `{ name: "relu" }` is an activation config; `{ name: "output" }` names the layer.
    return !(typeof name === "string" && (ACTIVATION_NAMES as readonly string[]).includes(name));
}

/**
 * Creates a {@link Dropout} layer that zeroes a fraction `rate` of its inputs during training.
 * @example
 * dropout(0.2);
 */
export function dropout(rate: number, options: LayerOptions = {}): Dropout {
    return new Dropout({ ...options, rate });
}

/**
 * Creates a {@link BatchNormalization} layer (momentum 0.99, epsilon 1e-3 by default).
 * @example
 * batchNormalization({ momentum: 0.9 });
 */
export function batchNormalization(options: BatchNormalizationOptions = {}): BatchNormalization {
    return new BatchNormalization(options);
}

/**
 * Creates a standalone {@link ActivationLayer}.
 * @example
 * activation("relu");
 * activation({ name: "leakyRelu", alpha: 0.2 });
 */
export function activation(
    id: ActivationIdentifier,
    options: Omit<ActivationLayerOptions, "activation"> = {},
): ActivationLayer {
    return new ActivationLayer({ ...options, activation: id });
}

/**
 * Registers a custom layer type so {@link layerFromConfig} (and therefore model loading) can
 * rebuild it. `type` must match the `type` field of the layer's `getConfig()` output.
 * Built-in types cannot be replaced.
 */
export function registerLayer(type: string, fromConfig: LayerFromConfig): void {
    if (typeof type !== "string" || type.length === 0) {
        throw new ValidationError(`registerLayer: type must be a non-empty string, got ${describeValue(type)}`);
    }
    if (typeof fromConfig !== "function") {
        throw new ValidationError(`registerLayer("${type}"): fromConfig must be a function (config) => Layer`);
    }
    if (LAYER_TYPES.includes(type)) throw new ValidationError(`registerLayer: "${type}" is a built-in layer type`);
    REGISTRY.set(type, fromConfig);
}

/**
 * Rebuilds a layer (unbuilt, without weights) from its `getConfig()` output.
 * @example
 * const copy = layerFromConfig(dense(4, "relu").getConfig());
 * @throws ValidationError for unknown layer types or invalid configs.
 */
export function layerFromConfig(config: LayerConfig): Layer {
    if (config === null || typeof config !== "object" || Array.isArray(config)) {
        throw new ValidationError(`layerFromConfig: expected a layer config object, got ${describeValue(config)}`);
    }
    const factory = REGISTRY.get(config.type);
    if (factory === undefined) {
        throw new ValidationError(
            `Unknown layer type ${describeValue(config.type)}. Known types: ${[...REGISTRY.keys()].join(", ")}. ` +
                "Register custom layers with registerLayer(type, fromConfig)",
        );
    }
    return factory(config);
}
