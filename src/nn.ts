import { activation, ActivationFunction } from './activation';
import { initList, multiply, minus, add } from './matrix';
import { listRand, listRandNormal } from './random';

/** Use uniformly or normally distributed numbers in the initialization of the weights? */
const useRandNormalInit = true;

/** Returns a weight matrix with given dimension */
const createWeightMatrix = useRandNormalInit ?
    (rows: number, cols: number) => listRandNormal(0, rows ** -0.5)(rows * cols):
    (rows: number, cols: number) => listRand(rows * cols);

export class NeuralNetwork {

    readonly layerSizes: number[];

    readonly weights: number[][];

    readonly activation: ActivationFunction;

    constructor(layerSizes: number[]) {
        this.layerSizes = layerSizes;
        this.weights = [];
        this.activation = activation.sigmoid;
        for (let i = 0, len = layerSizes.length - 1; i < len; i++) {
            this.weights.push(createWeightMatrix(layerSizes[i], layerSizes[i + 1]));
        }
    }

    /** Map inputs through all layers and return a list of all layer inputs and outputs */
    forward(inputs: number[]) {
        let outputs: number[][] = [inputs];
        for (let i = 0, len = this.layerSizes.length - 1; i < len; i++) {
            // Calculate next layer inputs
            inputs = multiply(this.weights[i], inputs, this.layerSizes[i + 1], this.layerSizes[i], 1);
            outputs.push(inputs);
            // Calculate next layer outputs
            inputs = inputs.map(this.activation.map);
            outputs.push(inputs);
        }
        return outputs;
    }

    /** Calculate the outputs from the inputs */
    map(inputs: number[]) {
        return this.forward(inputs)[(this.layerSizes.length - 1) * 2];
    }

}
