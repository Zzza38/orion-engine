# Orion Engine

Orion Engine is an ESM neural network framework for Node.js.

## Install

```bash
pnpm add @zzza38/orion-engine
```

## Usage

```typescript
import { NeuralNetwork, writeNetworkToFile, loadNetworkFromFile } from "@zzza38/orion-engine";

const network = new NeuralNetwork();
network.addLayer(2, "linear");
network.addLayer(4, "relu");
network.addLayer(1, "sigmoid");

network.train(
    [[0, 0], [0, 1], [1, 0], [1, 1]],
    [[0], [1], [1], [0]],
    10000,
    0.3,
    "crossEntropy",
);

console.log(network.runNetwork([1, 0]));

writeNetworkToFile(network, "model.onn");
const loaded = loadNetworkFromFile("model.onn");
```

## Development

```bash
pnpm install
pnpm dev    # run XOR training + save/load demo
pnpm build  # compile to build/
pnpm test   # run tests
```

See [docs/files.md](docs/files.md) for the `.onn` model file format.
