import {loadNetwork, writeNetwork} from "./fileHandler.js";

const network = loadNetwork(`2:relu:2:swish
0.71:-0.2:0.19|-1.82:0.95:0.97`);
console.log(writeNetwork(network));