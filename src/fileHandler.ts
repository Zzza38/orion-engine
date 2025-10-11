import * as fs from "fs";
import {NeuralNetwork, NeuralNetworkActivationFunction, NeuralNetworkLayer} from "./classes.js";

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