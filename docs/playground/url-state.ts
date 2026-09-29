/**
 * Serializes the playground configuration to and from the URL hash so setups are shareable.
 * Format: `#data=circle&noise=5&layers=8:tanh,6:tanh:0.1&...` (readable, order-independent).
 * Unknown keys are ignored and every value is validated, so hand-edited URLs cannot break the app.
 */
import type { ActivationId, DatasetId, FeatureId, OptimizerId, PlaygroundConfig, SpeedId } from "./config.js";
import { defaultConfig, sanitizeConfig } from "./config.js";

export function encodeConfig(config: PlaygroundConfig): string {
    const layers = config.layers
        .map((l) => (l.dropout > 0 ? `${l.units}:${l.activation}:${l.dropout}` : `${l.units}:${l.activation}`))
        .join(",");
    const entries: [string, string | number][] = [
        ["data", config.dataset],
        ["noise", config.noise],
        ["samples", config.samples],
        ["train", config.trainRatio],
        ["classes", config.classes],
        ["dseed", config.dataSeed],
        ["features", config.features.join(",")],
        ["layers", layers],
        ["opt", config.optimizer],
        ["lr", config.learningRate],
        ["batch", config.batchSize],
        ["l2", config.l2],
        ["seed", config.seed],
        ["speed", config.speed],
    ];
    // Values only ever contain [A-Za-z0-9.,:-], all of which are legal unescaped in a URL fragment.
    return entries.map(([key, value]) => `${key}=${String(value).replace(/[^\w.,:-]/g, "")}`).join("&");
}

/**
 * Parses a hash (with or without the leading `#`). Missing or invalid fields fall back to
 * `base` (the defaults when omitted); out-of-range numbers are clamped.
 */
export function decodeConfig(hash: string, base: PlaygroundConfig = defaultConfig()): PlaygroundConfig {
    const params = new Map<string, string>();
    for (const part of hash.replace(/^#/, "").split("&")) {
        if (!part) continue;
        const eq = part.indexOf("=");
        if (eq <= 0) continue;
        let value = part.slice(eq + 1);
        try {
            value = decodeURIComponent(value);
        } catch {
            // Keep the raw value; validation below rejects anything malformed.
        }
        params.set(part.slice(0, eq), value);
    }

    const num = (key: string, fallback: number): number => {
        const raw = params.get(key);
        if (raw === undefined || raw.trim() === "") return fallback;
        const value = Number(raw);
        return Number.isFinite(value) ? value : fallback;
    };
    const str = <T extends string>(key: string, fallback: T): T => (params.get(key) as T | undefined) ?? fallback;

    const config: PlaygroundConfig = {
        dataset: str<DatasetId>("data", base.dataset),
        noise: num("noise", base.noise),
        samples: num("samples", base.samples),
        trainRatio: num("train", base.trainRatio),
        classes: num("classes", base.classes),
        dataSeed: num("dseed", base.dataSeed),
        features: params.has("features")
            ? (params
                  .get("features")!
                  .split(",")
                  .filter((f) => f.length > 0) as FeatureId[])
            : [...base.features],
        layers: params.has("layers") ? parseLayers(params.get("layers")!) : base.layers.map((l) => ({ ...l })),
        optimizer: str<OptimizerId>("opt", base.optimizer),
        learningRate: num("lr", base.learningRate),
        batchSize: num("batch", base.batchSize),
        l2: num("l2", base.l2),
        seed: num("seed", base.seed),
        speed: str<SpeedId>("speed", base.speed),
    };
    return sanitizeConfig(config);
}

function parseLayers(text: string): PlaygroundConfig["layers"] {
    const layers: PlaygroundConfig["layers"] = [];
    for (const chunk of text.split(",")) {
        const [units, activation, dropout] = chunk.split(":");
        const count = Number(units);
        if (units.trim() === "" || !Number.isFinite(count)) continue;
        layers.push({
            units: count,
            activation: (activation || "tanh") as ActivationId,
            dropout: dropout === undefined ? 0 : Number(dropout) || 0,
        });
    }
    return layers;
}
