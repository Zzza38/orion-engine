/**
 * Runs every example in this folder, one after another, each in its own Node.js process, and
 * exits with an error if any of them fails. Used by `pnpm examples` and CI.
 *
 *   npx tsx examples/run-all.ts            # all examples
 *   npx tsx examples/run-all.ts iris xor   # only the named ones
 *
 * Expected output: each example's own output under a "▶ <name>" heading, then a summary table
 * with every example marked "ok" and its duration.
 */
import { spawnSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const self = basename(fileURLToPath(import.meta.url));
/** Kill an example that hangs instead of blocking CI forever. */
const TIMEOUT_MS = 60_000;

// Top-level .ts files only: examples/data/ holds datasets, not programs.
const available = readdirSync(here, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".ts") && entry.name !== self)
    .map((entry) => entry.name.slice(0, -".ts".length))
    .sort();
const requested = process.argv.slice(2);
const unknown = requested.filter((name) => !available.includes(name));
if (unknown.length > 0) {
    console.error(`Unknown example(s): ${unknown.join(", ")}. Available: ${available.join(", ")}`);
    process.exit(2);
}
const names = requested.length > 0 ? requested : available;

// Children run TypeScript through tsx's loader, resolved from this package's dependencies.
const tsx = import.meta.resolve("tsx");
const results: { name: string; ok: boolean; ms: number; detail: string }[] = [];
for (const name of names) {
    console.log(`\n▶ ${name}\n`);
    const start = performance.now();
    const child = spawnSync(process.execPath, ["--import", tsx, join(here, `${name}.ts`)], {
        stdio: "inherit",
        timeout: TIMEOUT_MS,
    });
    const ms = performance.now() - start;
    let detail = "";
    if (child.error !== undefined) detail = child.error.message;
    else if (child.signal !== null) detail = `killed by ${child.signal}`;
    else if (child.status !== 0) detail = `exit code ${child.status}`;
    results.push({ name, ok: detail === "", ms, detail });
}

const width = Math.max(...results.map((r) => r.name.length));
console.log("\nSummary");
for (const { name, ok, ms, detail } of results) {
    const time = `${(ms / 1000).toFixed(1)} s`.padStart(7);
    console.log(`  ${name.padEnd(width)}  ${ok ? "ok    " : "FAILED"} ${time}${ok ? "" : `  (${detail})`}`);
}
const failed = results.filter((r) => !r.ok);
if (failed.length > 0) {
    console.error(`\n${failed.length} of ${results.length} example(s) failed.`);
    process.exit(1);
}
console.log(`\nAll ${results.length} examples passed.`);
