# Contributing to Orion Engine

Thanks for your interest in improving Orion Engine! This guide covers local setup, the scripts you'll use, what we
expect from tests, and how releases are made.

## Setup

Requirements:

- **Node.js 22 or newer** (22 and 24 are LTS; CI also runs the current release).
- **pnpm 10** — the exact version is pinned in the `packageManager` field of `package.json`. On Node 22/24 you can run
  `corepack enable` to get it automatically; otherwise `npm install -g pnpm@10`.

```bash
git clone https://github.com/Zzza38/orion-engine.git
cd orion-engine
pnpm install
pnpm test
```

## Project layout

| Path               | What lives there                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------- |
| `src/index.ts`     | Main, browser-safe entry point. Must not import Node built-ins or use Node-only globals.     |
| `src/node.ts`      | Node-only helpers (file system save/load), published as `@zzza38/orion-engine/node`.        |
| `src/**`           | Library modules.                                                                            |
| `tests/**`         | `node:test` suites, named `*.test.ts`.                                                      |
| `examples/`        | Small runnable programs (`pnpm tsx examples/<name>.ts`).                                    |
| `bench/`           | Micro-benchmarks (`pnpm bench`).                                                            |
| `docs/playground/` | Browser demo, bundled with esbuild and deployed to GitHub Pages.                            |

## Scripts

| Script                  | Description                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------- |
| `pnpm build`            | Clean `build/` and compile `src/` with `tsconfig.build.json` (JS, `.d.ts`, maps).     |
| `pnpm typecheck`        | Type-check everything (src, tests, examples, bench, playground) + the browser guard.  |
| `pnpm test`             | Run all `tests/**/*.test.ts` with `node:test` via tsx.                                |
| `pnpm test:coverage`    | Same, with Node's built-in coverage and minimum thresholds for `src/`.                |
| `pnpm lint`             | Lint with Biome.                                                                      |
| `pnpm format`           | Format all files with Biome (writes changes).                                         |
| `pnpm check`            | Biome format + lint + import sorting check (what CI runs).                            |
| `pnpm check:fix`        | Apply all safe Biome fixes and formatting.                                            |
| `pnpm bench`            | Run the benchmarks in `bench/`.                                                       |
| `pnpm examples`         | Run every example in `examples/` (each in its own process; fails if any fails).       |
| `pnpm playground:dev`   | Serve the playground at <http://localhost:8000> with rebuild on change.               |
| `pnpm playground:build` | Bundle the playground into `docs/playground/dist/`.                                   |
| `pnpm pack:check`       | Validate the packed package with publint and Are the Types Wrong (run after build).   |

Before opening a pull request, run:

```bash
pnpm check && pnpm typecheck && pnpm test
```

To run a single test file or filter by name:

```bash
pnpm tsx --test tests/matrix.test.ts
pnpm tsx --test --test-name-pattern="softmax" "tests/**/*.test.ts"
```

## Code style

- Formatting and linting are handled by [Biome](https://biomejs.dev) (4-space indent, double quotes, semicolons,
  120-column lines). Run `pnpm check:fix` before committing, or install the Biome editor extension.
- TypeScript is strict, with `verbatimModuleSyntax`: use `import type` for type-only imports and include the `.js`
  extension in relative imports (`import { Matrix } from "./core/matrix.js"`).
- **Keep the main entry browser-safe.** Anything reachable from `src/index.ts` may only use globals shared by Node and
  browsers (e.g. `TextEncoder`, `console`) — no `node:*` imports, `Buffer` or `process`. `pnpm typecheck` enforces this
  with `tsconfig.browser.json`, and the DOM-free `tsconfig.build.json` keeps browser-only globals out as well.
  Node-specific code belongs in `src/node.ts`.
- Performance matters: prefer typed arrays and plain loops in hot paths, avoid per-element allocations, and back
  optimisations with a benchmark in `bench/`.
- Document public APIs with TSDoc comments.

## Testing expectations

Every change to behaviour needs tests. In particular:

- **Gradient checks.** Every new or changed layer, activation, and loss must include a numerical gradient check that
  compares the analytic backward pass against central finite differences,
  `(f(x + ε) − f(x − ε)) / 2ε` with `ε ≈ 1e-5`, requiring a relative error below about `1e-4`. Check gradients with
  respect to both inputs and parameters, and use small random shapes (not only squares) to catch transposition bugs.
- **Determinism.** Use a seeded random generator in tests; never depend on `Math.random()`. A test must produce the same
  result on every run and every platform.
- **Numerical tolerance.** Compare floating-point results with an explicit tolerance rather than strict equality.
- **Edge cases.** Cover shape mismatches and invalid arguments (the error type and message), empty or single-element
  inputs, and numerically extreme values (large logits, values near zero for `log`).
- **Serialization.** Anything that is saved must round-trip: `load(save(model))` must produce identical predictions.
- **Training smoke tests.** Keep them small and fast (well under a second each) with a fixed seed.

CI runs the suite on Linux with Node 22, 24 and 26, plus Windows and macOS on Node 24, and enforces coverage thresholds
on Node 24.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Adam optimizer
fix(matrix): correct stride in transposed multiply
perf: reuse gradient buffers in Dense.backward
docs: document the model file format
test: add gradient checks for softmax cross-entropy
chore(deps): bump typescript
```

Use `!` (e.g. `feat!:`) or a `BREAKING CHANGE:` footer for breaking changes. Keep the subject line under ~72 characters
and in the imperative mood.

## Changelog

User-facing changes get an entry under `## [Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md), following
[Keep a Changelog](https://keepachangelog.com/). While the major version is `0`, breaking changes bump the minor version.

## Release process

Releases are published from GitHub Actions using npm
[trusted publishing](https://docs.npmjs.com/trusted-publishers) (OIDC), so no npm token is stored in the repository and
every release carries a provenance attestation.

1. Make sure `main` is green in CI.
2. Update the version in `package.json` (e.g. `npm version 0.2.0 --no-git-tag-version`).
3. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [0.2.0] - YYYY-MM-DD` and add a fresh empty `Unreleased`
   section above it.
4. Commit: `chore(release): v0.2.0`.
5. Tag and push:

   ```bash
   git tag v0.2.0
   git push origin main --tags
   ```

6. The **Publish** workflow checks that the tag matches `package.json`, runs lint, typecheck, tests, build and package
   checks, publishes to npm (pre-release versions such as `0.2.0-beta.1` go to the `next` dist-tag), and creates a GitHub
   Release using the matching `CHANGELOG.md` section as release notes.

The playground is redeployed to GitHub Pages automatically whenever `main` changes.
