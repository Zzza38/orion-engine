import { NeuralNetwork } from "./classes.js";

const network = new NeuralNetwork(true);
network.addLayer(1);
network.addLayer(1);
console.log(network.runNetwork([2]));