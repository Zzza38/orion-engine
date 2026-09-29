/**
 * Seedable pseudo-random number generator (xoshiro128**, seeded via splitmix32).
 * Every source of randomness in the engine (initialization, shuffling, dropout)
 * goes through an instance of this class so training runs are reproducible.
 */
export class Random {
    readonly seed: number;
    private s0 = 0;
    private s1 = 0;
    private s2 = 0;
    private s3 = 0;
    private spareNormal: number | null = null;

    /** @param seed - 32-bit integer seed. Omit for a non-deterministic seed. */
    constructor(seed?: number) {
        this.seed = seed === undefined ? Math.floor(Math.random() * 0x100000000) >>> 0 : seed >>> 0;
        let state = this.seed;
        const splitmix = () => {
            state = (state + 0x9e3779b9) >>> 0;
            let z = state;
            z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
            z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
            return (z ^ (z >>> 16)) >>> 0;
        };
        this.s0 = splitmix();
        this.s1 = splitmix();
        this.s2 = splitmix();
        this.s3 = splitmix();
    }

    /** Next raw 32-bit unsigned integer. */
    nextUint32(): number {
        const result = Math.imul(rotl(Math.imul(this.s1, 5) >>> 0, 7), 9) >>> 0;
        const t = (this.s1 << 9) >>> 0;
        this.s2 = (this.s2 ^ this.s0) >>> 0;
        this.s3 = (this.s3 ^ this.s1) >>> 0;
        this.s1 = (this.s1 ^ this.s2) >>> 0;
        this.s0 = (this.s0 ^ this.s3) >>> 0;
        this.s2 = (this.s2 ^ t) >>> 0;
        this.s3 = rotl(this.s3, 11);
        return result;
    }

    /** Uniform float in [0, 1). */
    next(): number {
        return this.nextUint32() / 0x100000000;
    }

    /** Uniform float in [min, max). */
    uniform(min = 0, max = 1): number {
        return min + (max - min) * this.next();
    }

    /** Uniform integer in [0, maxExclusive). */
    int(maxExclusive: number): number {
        return Math.floor(this.next() * maxExclusive);
    }

    /** Normally distributed sample (Box-Muller). */
    normal(mean = 0, stddev = 1): number {
        if (this.spareNormal !== null) {
            const value = this.spareNormal;
            this.spareNormal = null;
            return mean + stddev * value;
        }
        let u = 0;
        while (u === 0) u = this.next();
        const v = this.next();
        const radius = Math.sqrt(-2 * Math.log(u));
        const theta = 2 * Math.PI * v;
        this.spareNormal = radius * Math.sin(theta);
        return mean + stddev * radius * Math.cos(theta);
    }

    /** Normal sample re-drawn until it lies within two standard deviations of the mean. */
    truncatedNormal(mean = 0, stddev = 1): number {
        for (;;) {
            const z = this.normal();
            if (Math.abs(z) <= 2) return mean + stddev * z;
        }
    }

    /** Fisher-Yates shuffle, in place. Returns the same array. */
    shuffle<T extends { length: number; [index: number]: unknown }>(array: T): T {
        for (let i = array.length - 1; i > 0; i--) {
            const j = this.int(i + 1);
            const tmp = array[i];
            array[i] = array[j];
            array[j] = tmp;
        }
        return array;
    }

    /** Derives an independent generator from this one's stream. */
    fork(): Random {
        return new Random(this.nextUint32());
    }
}

function rotl(x: number, k: number): number {
    return ((x << k) | (x >>> (32 - k))) >>> 0;
}
