/** Canvas helpers shared by the renderers. */

export interface CanvasFrame {
    ctx: CanvasRenderingContext2D;
    /** CSS pixel size. Drawing coordinates are in CSS pixels (the context is pre-scaled). */
    width: number;
    height: number;
    dpr: number;
}

/**
 * Sizes the backing store to the element's CSS size × devicePixelRatio (so lines stay crisp on
 * high-density screens) and returns a context scaled to CSS pixels. Returns null while hidden.
 */
export function prepareCanvas(canvas: HTMLCanvasElement): CanvasFrame | null {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width === 0 || height === 0) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height, dpr };
}

/** Calls `callback` (coalesced to one call per frame) whenever any of `elements` changes size. */
export function observeResize(elements: readonly Element[], callback: () => void): void {
    let pending = false;
    const observer = new ResizeObserver(() => {
        if (pending) return;
        pending = true;
        requestAnimationFrame(() => {
            pending = false;
            callback();
        });
    });
    for (const element of elements) observer.observe(element);
}
