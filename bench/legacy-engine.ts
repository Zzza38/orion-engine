// Verbatim copy of the pre-0.1 engine (src/classes.ts + src/fileHandler.ts), kept only as a
// benchmark baseline. Only the imports were changed (merged into one file, node:fs).
// biome-ignore-all lint: verbatim legacy code, intentionally left as-is
// biome-ignore-all format: verbatim legacy code, intentionally left as-is
import * as fs from "node:fs";

// Neurons
export interface NeuralNetworkNeuron {
    value: number;
    preActivationValue: number;
    weights: number[];
    bias: number;
}

// Activation Functions
export type NeuralNetworkActivationFunction =
    "linear"
    | "sigmoid"
    | "tanh"
    | "relu"
    | "leakyRelu"
    | "elu"
    | "softmax"
    | "swish";

export class Activation {
    static sigmoid(x: number) {
        return 1 / (1 + Math.exp(-x));
    }

    static tanh(x: number) {
        return Math.tanh(x);
    }

    static relu(x: number) {
        return Math.max(0, x);
    }

    static leakyRelu(x: number, alpha = 0.01) {
        return x > 0 ? x : alpha * x;
    }

    static elu(x: number, alpha = 1.0) {
        return x > 0 ? x : alpha * (Math.exp(x) - 1);
    }

    static swish(x: number) {
        return x / (1 + Math.exp(-x));
    }

    static softmax(arr: number[]) {
        let max = -Infinity;
        for (let i = 0; i < arr.length; i++) {
            const v = arr[i];
            if (v > max) max = v;
        }
        const exp = arr.map(v => Math.exp(v - max));
        let sum = 0;
        for (let i = 0; i < exp.length; i++) sum += exp[i];
        if (!Number.isFinite(sum) || sum <= 0) throw new Error("Softmax sum invalid");
        return exp.map(v => v / sum);
    }


    static use(
        activation: NeuralNetworkActivationFunction,
        value: number | number[],
    ): number | number[] {
        if (activation === "softmax") {
            if (!Array.isArray(value))
                throw new Error("Softmax requires an array input");
            return this.softmax(value);
        }

        if (Array.isArray(value)) return value.map(v => this.use(activation, v) as number);

        switch (activation) {
            case "linear":
                return value;
            case "sigmoid":
                return this.sigmoid(value);
            case "tanh":
                return this.tanh(value);
            case "relu":
                return this.relu(value);
            case "leakyRelu":
                return this.leakyRelu(value);
            case "elu":
                return this.elu(value);
            case "swish":
                return this.swish(value);
            default:
                throw new Error(`Unknown activation function: ${activation}`);
        }
    }
}

export class ActivationDerivative {
    static linear(x: number) {
        return 1;
    }

    static sigmoid(x: number) {
        const sig = Activation.sigmoid(x);
        return sig * (1 - sig);
    }

    static tanh(x: number) {
        const t = Activation.tanh(x);
        return 1 - t * t;
    }

    static relu(x: number) {
        return x > 0 ? 1 : 0;
    }

    static leakyRelu(x: number, alpha = 0.01) {
        return x > 0 ? 1 : alpha;
    }

    static elu(x: number, alpha = 1.0) {
        return x > 0 ? 1 : alpha * Math.exp(x);
    }

    static swish(x: number) {
        const s = Activation.swish(x);
        return s + (1 - s) * Activation.sigmoid(x);
    }

    static softmax(arr: number[]) {
        // Calculate softmax output values (as in the base Activation.softmax)
        const max = Math.max(...arr);
        const exp = arr.map(v => Math.exp(v - max));
        const sum = exp.reduce((acc, v) => acc + v, 0);
        if (!Number.isFinite(sum) || sum <= 0) throw new Error("Softmax sum invalid");
        const softmax = exp.map(v => v / sum);
        // The derivative of softmax for a vector is a Jacobian matrix, but to match the output of regular softmax,
        // we will return the element-wise (vector) derivative, i.e., the gradient for each element with respect to itself:
        // grad_i = softmax_i * (1 - softmax_i)
        return softmax.map(s => s * (1 - s));
    }

    static use(activation: NeuralNetworkActivationFunction, value: number | number[]): number | number[] {
        if (activation === "softmax") {
            if (!Array.isArray(value)) throw new Error("Softmax derivative requires array input");
            return this.softmax(value);
        }
        if (Array.isArray(value)) return value.map(v => this.use(activation, v) as number);

        switch (activation) {
            case "linear":
                return this.linear(value);
            case "sigmoid":
                return this.sigmoid(value);
            case "tanh":
                return this.tanh(value);
            case "relu":
                return this.relu(value);
            case "leakyRelu":
                return this.leakyRelu(value);
            case "elu":
                return this.elu(value);
            case "swish":
                return this.swish(value);
            default:
                throw new Error(`Unknown activation function: ${activation}`);
        }
    }
}

// Layers
export type NeuralNetworkLayerType = "hidden";

export interface NeuralNetworkLayer {
    neurons: NeuralNetworkNeuron[];
    type: NeuralNetworkLayerType;
    activation: NeuralNetworkActivationFunction;
}

// Model
export interface NeuralNetworkModel {
    layers: NeuralNetworkLayer[];
}
// Loss
export type NeuralNetworkLossType = "mse" | "mae" | "crossEntropy";
export class Loss {
    /** Mean Squared Error (good for regression) */
    static mse(pred: number[], target: number[]) {
        if (pred.length !== target.length) throw new Error("MSE: Shape mismatch");
        let sum = 0;
        for (let i = 0; i < pred.length; i++) {
            const diff = pred[i] - target[i];
            sum += diff * diff;
        }
        return sum / pred.length;
    }

    /** Mean Absolute Error (less sensitive to outliers) */
    static mae(pred: number[], target: number[]) {
        if (pred.length !== target.length) throw new Error("MAE: Shape mismatch");
        let sum = 0;
        for (let i = 0; i < pred.length; i++) {
            sum += Math.abs(pred[i] - target[i]);
        }
        return sum / pred.length;
    }

    /** Cross Entropy (for classification) */
    static crossEntropy(pred: number[], target: number[]) {
        if (pred.length !== target.length) throw new Error("CrossEntropy: Shape mismatch");
        let loss = 0;
        for (let i = 0; i < pred.length; i++) {
            const p = Math.max(pred[i], 1e-9); // avoid log(0)
            loss += -target[i] * Math.log(p);
        }
        return loss / pred.length;
    }

    /** dLoss/dPrediction for backpropagation */
    static gradient(type: NeuralNetworkLossType, pred: number[], target: number[]): number[] {
        if (pred.length !== target.length) throw new Error("Loss gradient: Shape mismatch");
        const n = pred.length;
        switch (type) {
            case "mse":
                return pred.map((p, i) => (2 * (p - target[i])) / n);
            case "mae":
                return pred.map((p, i) => {
                    if (p > target[i]) return 1 / n;
                    if (p < target[i]) return -1 / n;
                    return 0;
                });
            case "crossEntropy":
                return pred.map((p, i) => {
                    const clamped = Math.max(p, 1e-9);
                    return -target[i] / (clamped * n);
                });
            default:
                throw new Error(`Unknown loss type: ${type}`);
        }
    }
}

export class
    NeuralNetwork {
    private model: NeuralNetworkModel = {
        layers: []
    };
    strict: boolean = false;

    constructor(strict?: boolean) {
        this.strict = !!strict;
    }

    /**
     * Adds a layer to the neural network
     * @param neuronCount - How many neurons in that specific layer
     * @param activation - The type of activation the layer experiences (default: ReLU)
     */
    addLayer(neuronCount: number, activation: NeuralNetworkActivationFunction = "relu"): boolean {
        if (neuronCount < 1) return false;
        const layer: NeuralNetworkLayer = {
            type: "hidden",
            neurons: [],
            activation: activation
        };
        layer.neurons = Array.from({ length: neuronCount }, () => ({
            value: 0,
            preActivationValue: 0,
            weights: [],
            bias: (Math.random() - 0.5) * 0.2,
        }));
        this.model.layers.push(layer);
        this.fixWeights();
        return true;
    }

    /**
     * Fixes the weight mappings with neurons, useful if a layer was inserted.
     * Will delete weight values that are unused and create more if needed (random init)
     */
    fixWeights() {
        for (let layerIdx = 1; layerIdx < this.model.layers.length; layerIdx++) {
            const layer = this.model.layers[layerIdx];
            const lastLayer = this.model.layers[layerIdx - 1];
            if (!lastLayer || !lastLayer.neurons) continue;

            const fanIn = lastLayer.neurons.length;
            for (const neuron of layer.neurons) {
                neuron.weights = neuron.weights.slice(0, fanIn);
                const diff = fanIn - neuron.weights.length;
                for (let i = 0; i < diff; i++) {
                    neuron.weights.push(this.randomWeight(fanIn));
                }
            }
        }
    }

    private randomWeight(fanIn: number): number {
        const limit = Math.sqrt(6 / fanIn);
        return (Math.random() * 2 - 1) * limit;
    }

    /**
     * Loads in the weights and biases for a specific layer
     * @param layer - The layer index in which to update the neurons in
     * @param weights - The new updated weights
     * @param biases - The new updated biases
     */
    loadWeightsAndBiases(layer: number, weights: number[][], biases: number[]) {
        if (layer === 0) throw new Error("Modifying the weights and biases of the input layer is disallowed.");
        const neuronCount = this.model.layers[layer].neurons.length;
        const prevSize = layer === 0 ? 0 : this.model.layers[layer - 1].neurons.length;

        for (let i = 0; i < neuronCount; i++) {
            if (!Array.isArray(weights[i]) || weights[i].length !== prevSize) {
                if (this.strict) throw new Error("Bad weight shape");
                console.warn("Bad weight shape at row", i);
                return; // abort the entire load to avoid half-state
            }
            if (typeof biases[i] !== "number") {
                if (this.strict) throw new Error("Bad bias");
                console.warn("Bad bias at row", i);
                return;
            }
        }

        for (let i = 0; i < neuronCount; i++) {
            this.model.layers[layer].neurons[i].weights = weights[i];
            this.model.layers[layer].neurons[i].bias = biases[i];
        }
    }

    /**
     * Run the neural network
     * @param input - The number array to be inputted into the network
     */
    runNetwork(input: number[]): number[] {
        if (this.model.layers.length === 0) {
            throw new Error("No layers defined.");
        } else if (this.model.layers.length === 1) {
            throw new Error("No output layer defined.");
        }
        if (input.length !== this.model.layers[0].neurons.length) {
            if (this.strict) {
                throw new Error("Input length is not the same as input layer length");
            }
            console.warn("Input length is not the same as input layer length; if input length is less than input layer length, input layer will be filled with 0s for not filled values");
        }

        // TODO: Improve input layer loading
        for (let i = 0; i < this.model.layers[0].neurons.length; i++) {
            if (typeof input[i] !== "number") {
                this.model.layers[0].neurons[i].value = 0;
                if (this.strict) throw new Error("Input contains a value that is not a number.")
                console.warn("Input contains a value that is not a number. Filling with 0...");
                continue;
            }
            this.model.layers[0].neurons[i].value = input[i];
        }
        // Run the model
        for (let layerIndex = 1; layerIndex < this.model.layers.length; layerIndex++) {
            const layer = this.model.layers[layerIndex];
            for (const neuron of layer.neurons) {
                let val = 0;
                if (neuron.weights.length !== this.model.layers[layerIndex - 1].neurons.length) {
                    throw new Error(`On layer ${layerIndex}, weights do not match up with the neurons.`);
                }
                for (let lastLayerNeuronIndex = 0; lastLayerNeuronIndex < this.model.layers[layerIndex - 1].neurons.length; lastLayerNeuronIndex++) {
                    const lastLayerNeuron = this.model.layers[layerIndex - 1].neurons[lastLayerNeuronIndex];
                    val += lastLayerNeuron.value * neuron.weights[lastLayerNeuronIndex];
                }
                val += neuron.bias;
                neuron.value = val;
            }
            const values = layer.neurons.map(n => n.value);
            const activationResults = Activation.use(layer.activation, values);
            if (!Array.isArray(activationResults) || activationResults.length !== layer.neurons.length) {
                throw new Error("Activation output shape mismatch.");
            }
            for (let i = 0; i < layer.neurons.length; i++) {
                layer.neurons[i].preActivationValue = layer.neurons[i].value;
                layer.neurons[i].value = activationResults[i];
            }

        }
        return this.model.layers[this.model.layers.length - 1].neurons.map(
            neuron => neuron.value
        );
    }

    /**
     * Calculates the loss of the network given a target
     * @param input - The input to the neural network
     * @param target - The expected output
     * @param type - The type of loss calculation (Default: MSE)
     */
    calculateLoss(input: number[], target: number[], type: "mse" | "mae" | "crossEntropy" = "mse") {
        const predicted = this.runNetwork(input);
        switch (type) {
            case "mse":
                return Loss.mse(predicted, target);
            case "mae":
                return Loss.mae(predicted, target);
            case "crossEntropy":
                return Loss.crossEntropy(predicted, target);
            default:
                throw new Error(`Unknown loss type: ${type}`);
        }
    }
    private lossFromPredicted(predicted: number[], target: number[], type: NeuralNetworkLossType): number {
        switch (type) {
            case "mse":
                return Loss.mse(predicted, target);
            case "mae":
                return Loss.mae(predicted, target);
            case "crossEntropy":
                return Loss.crossEntropy(predicted, target);
            default:
                throw new Error(`Unknown loss type: ${type}`);
        }
    }

    private activationDerivativeAt(
        activation: NeuralNetworkActivationFunction,
        neuronIndex: number,
        layerNeurons: NeuralNetworkNeuron[],
    ): number {
        if (activation === "softmax") {
            const preActivations = layerNeurons.map(n => n.preActivationValue);
            const derivs = ActivationDerivative.use("softmax", preActivations);
            if (!Array.isArray(derivs)) throw new Error("Softmax derivative output shape mismatch.");
            return derivs[neuronIndex];
        }
        const deriv = ActivationDerivative.use(activation, layerNeurons[neuronIndex].preActivationValue);
        if (typeof deriv !== "number") throw new Error("Activation derivative output shape mismatch.");
        return deriv;
    }

    private outputDeltas(
        predicted: number[],
        target: number[],
        lossType: NeuralNetworkLossType,
        outputLayer: NeuralNetworkLayer,
    ): number[] {
        if (lossType === "crossEntropy") {
            const act = outputLayer.activation;
            if (act === "sigmoid" || act === "softmax") {
                return predicted.map((p, i) => (p - target[i]) / predicted.length);
            }
        }

        const lossGrads = Loss.gradient(lossType, predicted, target);
        return lossGrads.map((grad, i) =>
            grad * this.activationDerivativeAt(outputLayer.activation, i, outputLayer.neurons)
        );
    }

    private applyLayerUpdate(layerIdx: number, deltas: number[], learningRate: number) {
        const layer = this.model.layers[layerIdx];
        const prevLayer = this.model.layers[layerIdx - 1];

        for (let j = 0; j < layer.neurons.length; j++) {
            const neuron = layer.neurons[j];
            const delta = deltas[j];
            for (let i = 0; i < neuron.weights.length; i++) {
                neuron.weights[i] -= learningRate * delta * prevLayer.neurons[i].value;
            }
            neuron.bias -= learningRate * delta;
        }
    }

    /**
     * Trains the network on a single example via backpropagation.
     * @param input - The input to the neural network
     * @param target - The expected output
     * @param learningRate - How fast the network should learn (setting too high will cause instability)
     * @param lossType - The loss function to optimize (default: MSE)
     * @returns The loss after the forward pass
     */
    backpropagate(
        input: number[],
        target: number[],
        learningRate: number,
        lossType: NeuralNetworkLossType = "mse",
    ): number {
        const predicted = this.runNetwork(input);
        const loss = this.lossFromPredicted(predicted, target, lossType);

        const outputLayerIdx = this.model.layers.length - 1;
        const outputLayer = this.model.layers[outputLayerIdx];
        let deltas = this.outputDeltas(predicted, target, lossType, outputLayer);
        this.applyLayerUpdate(outputLayerIdx, deltas, learningRate);

        for (let layerIdx = outputLayerIdx - 1; layerIdx >= 1; layerIdx--) {
            const layer = this.model.layers[layerIdx];
            const nextLayer = this.model.layers[layerIdx + 1];
            const newDeltas: number[] = [];

            for (let j = 0; j < layer.neurons.length; j++) {
                let error = 0;
                for (let k = 0; k < nextLayer.neurons.length; k++) {
                    error += deltas[k] * nextLayer.neurons[k].weights[j];
                }
                newDeltas.push(
                    error * this.activationDerivativeAt(layer.activation, j, layer.neurons)
                );
            }

            this.applyLayerUpdate(layerIdx, newDeltas, learningRate);
            deltas = newDeltas;
        }

        return loss;
    }

    /**
     * Trains the network over multiple epochs on a dataset.
     * @param inputs - List of input vectors
     * @param targets - List of expected output vectors (same length as inputs)
     * @param epochs - How many times to iterate over the full dataset
     * @param learningRate - How fast the network should learn
     * @param lossType - The loss function to optimize (default: MSE)
     * @returns The average loss across the dataset on the final epoch
     */
    train(
        inputs: number[][],
        targets: number[][],
        epochs: number,
        learningRate: number,
        lossType: NeuralNetworkLossType = "mse",
    ): number {
        if (inputs.length !== targets.length) {
            throw new Error("Train: inputs and targets must have the same length");
        }
        if (inputs.length === 0) {
            throw new Error("Train: dataset cannot be empty");
        }
        if (epochs < 1) {
            throw new Error("Train: epochs must be at least 1");
        }

        let avgLoss = 0;
        const order = inputs.map((_, i) => i);
        for (let epoch = 0; epoch < epochs; epoch++) {
            this.shuffleInPlace(order);
            let epochLoss = 0;
            for (const i of order) {
                epochLoss += this.backpropagate(inputs[i], targets[i], learningRate, lossType);
            }
            avgLoss = epochLoss / inputs.length;
        }

        return avgLoss;
    }

    private shuffleInPlace(indices: number[]) {
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
    }

    /**
     * Returns a reference to the layers in the model. Useful for debugging
     */
    get layers() {
        return this.model.layers;
    }
}

// ---- src/fileHandler.ts ----


export function loadNetwork(src: string): NeuralNetwork {
    const network = new NeuralNetwork(true);
    const [layerStructureString, weightsAndBiasesString] = src.split("\n");

    const layerStructure: number[] = [];
    const layerStruct = layerStructureString.split(":");

    for (let i = 0; i < layerStruct.length; i += 2) {
        const layerSize = Number(layerStruct[i]);
        const activation = layerStruct[i + 1] as NeuralNetworkActivationFunction;
        if (Number.isNaN(layerSize)) throw new Error("Cannot parse model file..");
        network.addLayer(layerSize, activation);
        layerStructure.push(layerSize);
    }

    let weights: number[][] = [];
    let biases: number[] = [];
    let layer = 1; // weights start from layer 1 (input -> hidden)
    let neuron = 0;

    for (const neuronString of weightsAndBiasesString.split("|")) {
        if (!neuronString.trim()) continue; // skip empties
        const neuronData = neuronString.split(":");
        const w = neuronData.slice(0, -1).map(Number);
        const b = Number(neuronData.at(-1));
        if (w.some(isNaN) || Number.isNaN(b)) {
            throw new Error(`Invalid weight or bias at layer ${layer}, neuron ${neuron}`);
        }

        weights.push(w);
        biases.push(b);
        neuron++;

        if (neuron === layerStructure[layer]) {
            network.loadWeightsAndBiases(layer, weights, biases);
            weights = [];
            biases = [];
            neuron = 0;
            layer++;
        }
    }

    // Catch any leftovers
    if (weights.length && layer < layerStructure.length) {
        network.loadWeightsAndBiases(layer, weights, biases);
    }

    return network;
}

export function loadNetworkFromFile(path: string): NeuralNetwork {
    if (!fs.existsSync(path)) throw new Error("File does not exist");
    const src = fs.readFileSync(path).toString();
    return loadNetwork(src);
}

export function writeNetwork(network: NeuralNetwork) {
    const layers = network.layers;
    let stringifiedModel = "";
    for (const layer of layers) {
        stringifiedModel += `${layer.neurons.length}:${layer.activation}:`;
    }
    stringifiedModel = stringifiedModel.slice(0, -1) + "\n";
    for (const layer of layers) {
        for (const neuron of layer.neurons) {
            if (neuron.weights.length === 0) continue;
            neuron.weights.forEach(weight => stringifiedModel += `${weight}:`);
            stringifiedModel += neuron.bias + '|';
        }
    }
    return stringifiedModel.slice(0, -1);
}

export function writeNetworkToFile(network: NeuralNetwork, path: string): boolean {
    const src = writeNetwork(network);
    fs.writeFileSync(path, src);
    return fs.readFileSync(path).toString() === src;
}