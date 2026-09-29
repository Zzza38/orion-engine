/**
 * Reads the canvas palette from CSS custom properties and manages the light/dark/system theme.
 */

export type Rgb = [number, number, number];

export interface Palette {
    classes: Rgb[];
    neutral: Rgb;
    /** Strongest blend of a class color into the neutral heatmap background. */
    mix: number;
    ring: string;
    outline: string;
    grid: string;
    axis: string;
    ink: string;
    ink2: string;
    muted: string;
    surface: string;
    seriesTrain: string;
    seriesTest: string;
    weightPos: Rgb;
    weightNeg: Rgb;
    weightNone: string;
    node: string;
    nodeStroke: string;
    accent: string;
}

export function parseColor(value: string): Rgb {
    const text = value.trim();
    const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(text);
    if (hex) {
        let digits = hex[1];
        if (digits.length === 3) digits = [...digits].map((d) => d + d).join("");
        const n = Number.parseInt(digits, 16);
        return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    const rgb = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i.exec(text);
    if (rgb) return [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])];
    return [128, 128, 128];
}

export function rgbCss([r, g, b]: Rgb, alpha = 1): string {
    return alpha >= 1 ? `rgb(${r} ${g} ${b})` : `rgb(${r} ${g} ${b} / ${alpha.toFixed(3)})`;
}

export function readPalette(): Palette {
    const style = getComputedStyle(document.documentElement);
    const v = (name: string) => style.getPropertyValue(name).trim();
    return {
        classes: [v("--class-0"), v("--class-1"), v("--class-2"), v("--class-3")].map(parseColor),
        neutral: parseColor(v("--plot-neutral")),
        mix: Number.parseFloat(v("--plot-mix")) || 0.55,
        ring: v("--plot-ring"),
        outline: v("--plot-outline"),
        grid: v("--grid"),
        axis: v("--axis"),
        ink: v("--ink"),
        ink2: v("--ink-2"),
        muted: v("--muted"),
        surface: v("--surface"),
        seriesTrain: v("--series-train"),
        seriesTest: v("--series-test"),
        weightPos: parseColor(v("--weight-pos")),
        weightNeg: parseColor(v("--weight-neg")),
        weightNone: v("--weight-none"),
        node: v("--node"),
        nodeStroke: v("--node-stroke"),
        accent: v("--accent"),
    };
}

export type ThemePreference = "system" | "light" | "dark";
const STORAGE_KEY = "orion-playground-theme";
const ORDER: readonly ThemePreference[] = ["system", "light", "dark"];

function loadPreference(): ThemePreference {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored === "light" || stored === "dark" || stored === "system") return stored;
    } catch {
        // Storage can be unavailable (private mode, sandboxed frames); fall back to the OS setting.
    }
    return "system";
}

function applyPreference(pref: ThemePreference): void {
    const root = document.documentElement;
    root.dataset.themePref = pref;
    if (pref === "system") delete root.dataset.theme;
    else root.dataset.theme = pref;
}

/**
 * Wires the theme toggle button and calls `onChange` whenever the effective colors change
 * (manual toggle or OS switch).
 */
export function initTheme(button: HTMLButtonElement, onChange: () => void): void {
    let pref = loadPreference();
    applyPreference(pref);
    const label = () => `Color theme: ${pref}`;
    button.setAttribute("aria-label", label());
    button.title = label();
    button.addEventListener("click", () => {
        pref = ORDER[(ORDER.indexOf(pref) + 1) % ORDER.length];
        applyPreference(pref);
        button.setAttribute("aria-label", label());
        button.title = label();
        try {
            localStorage.setItem(STORAGE_KEY, pref);
        } catch {
            // Ignore: the preference simply will not persist.
        }
        onChange();
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
        if (pref === "system") onChange();
    });
}
