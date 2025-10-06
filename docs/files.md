# File
## File Extension
OrionEngine's neural network file extension is `.onn`

## File Structure
OrionEngine's file structure follows a basic format:
* Dump model structure—layer size and layer activation function seperated by a colon (`:`)
* Newline (`\n`)
* Dump neuron weights—numbers separated by another colon (`:`)
* Dump neuron bias at the end of one neuron weights with a colon
* Separate neurons with a vertical pipe (`|`), and join the rest of the neurons

This file structure is HIGHLY inefficient and is only a starting base.
In the future, a binary format like what `.h5` and `.pt` are would be better.

Format:
```text
[layer_count_1]:[layer_activation_1]...
[neuron_1_weight_1]...:[neuron_1_bias]...
```
Example: 
```text
2:relu:2:swish
0.71:-0.2:0.19|-1.82:0.95:0.97
```