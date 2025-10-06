// Neurons
export interface NeuralNetworkNeuron {
    value: number;
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
        layer.neurons = Array.from({length: neuronCount}, () => ({
            value: 0,
            weights: [],
            bias: 0,
        }));
        this.model.layers.push(layer);
        this.fixWeights();
        return true;
    }

    /**
     * Fixes the weight mappings with neurons, useful if a layer was inserted.
     * Will delete weight values that are unused and create more if needed (.fill(1))
     */
    fixWeights() {
        for (let layerIdx = 1; layerIdx < this.model.layers.length; layerIdx++) {
            const layer = this.model.layers[layerIdx];
            const lastLayer = this.model.layers[layerIdx - 1];
            if (!lastLayer || !lastLayer.neurons) continue;

            for (const neuron of layer.neurons) {
                // Trim extra weights
                neuron.weights = neuron.weights.slice(0, lastLayer.neurons.length);
                // Add missing weights
                const diff = lastLayer.neurons.length - neuron.weights.length;
                if (diff > 0) neuron.weights.push(...Array(diff).fill(1));
            }
        }
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

    /**
     * Returns a reference to the layers in the model. Useful for debugging
     */
    get layers() {
        return this.model.layers;
    }
}