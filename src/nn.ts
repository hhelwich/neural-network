import { activation } from './activation';
import { initList, multiply, minus, add } from './matrix';
import { listRand, listRandNormal } from './random';

/** Use uniformly or normally distributed numbers in the initialization of the weights? */
const useRandNormalInit = true;

/** Returns a weight matrix with given dimension */
const createWeightMatrix = useRandNormalInit ?
    (rows: number, cols: number) => listRandNormal(0, rows ** -0.5)(rows * cols):
    (rows: number, cols: number) => listRand(rows * cols);

const activationFn = activation.sigmoid;

/**
 * Calculate the weight delta from the gradients and the previous delta.
 * The learningRate is used to scale the gradients and the momentum to scale the previous delta.
 */
const nextDelta = (learningRate: number, momentum: number, gradients: number[], lastDelta?: number[]) => {
    let delta = gradients.map(x => x * learningRate);
    if (lastDelta != null) {
        delta = add(delta, lastDelta.map(d => d * momentum));
    }
    return delta;
};

export class NeuralNetwork {

    readonly layerSizes: number[];

    readonly weights: number[][];

    private gradients: number[][];
    private lastDelta: number[][];

    constructor(layerSizes: number[]) {
        this.layerSizes = layerSizes;
        this.weights = [];
        this.lastDelta = [];
        for (let i = 0, len = layerSizes.length - 1; i < len; i++) {
            this.weights.push(createWeightMatrix(layerSizes[i], layerSizes[i + 1]));
        }
    }

    /** Map inputs through all layers and return a list of all layer inputs and outputs */
    private forward(inputs: number[]) {
        let outputs: number[][] = [inputs];
        for (let i = 0, len = this.layerSizes.length - 1; i < len; i++) {
            // Calculate next layer inputs
            inputs = multiply(this.weights[i], inputs, this.layerSizes[i + 1], this.layerSizes[i], 1);
            outputs.push(inputs);
            // Calculate next layer outputs
            inputs = inputs.map(activationFn.map);
            outputs.push(inputs);
        }
        return outputs;
    }

    /** Calculate the outputs from the inputs */
    map(inputs: number[]) {
        return this.forward(inputs)[(this.layerSizes.length - 1) * 2];
    }

    /** Train the neural network */
    train(inputs: number[], targets: number[]) {
        // Forward
        let outputs = this.forward(inputs);
        console.log('input transformation', JSON.stringify(outputs, null, 2));
        // Backward
        let outDeriv: number[];
        let grads: number[][] = [];
        for (let i = this.layerSizes.length - 1; i > 0; i--) {
            const size = this.layerSizes[i];
            const ins = outputs[i * 2 - 2];
            const out = outputs[i * 2 - 1];
            const outAct = outputs[i * 2];
            let errors: number[];
            // calculate error
            if (i === this.layerSizes.length - 1) { // errors for output layer
                errors = minus(targets, outAct, size); // .map(e => e * 2 / size);
            } else { // errors for hidden layer
                errors = multiply(outDeriv, this.weights[i], 1, this.layerSizes[i + 1], this.layerSizes[i]);
            }
            console.log('errors', JSON.stringify(errors, null, 2));
            // apply activation derivative on outputs and multiply with error
            outDeriv = [];
            for (let j = 0; j < size; j++) {
                outDeriv.push(activationFn.derivative(out[j], outAct[j]) * errors[j]);
            }
            console.log('derivative * errors', JSON.stringify(outDeriv, null, 2));
            // multiply with error
            grads[i - 1] = multiply(outDeriv, ins, size, 1, ins.length);
            console.log('gradients', JSON.stringify(grads[i - 1], null, 2));
        }
        if (this.gradients == null) {
            this.gradients = grads;
        } else {
            for (let i = 0; i < grads.length; i++) {
                this.gradients[i] = add(this.gradients[i], grads[i], grads[i].length);
                console.log('new gradients', JSON.stringify(this.gradients[i], null, 2));
            }
        }
    }

    update(learningRate: number, momentum: number) {
        const deltas: number[][] = [];
        // Calculate deltas
        for (let i = 0; i < this.layerSizes.length - 1; i++) {
            deltas.push(nextDelta(learningRate, momentum, this.gradients[i], this.lastDelta[i]));
        }
        // Update weights
        for (let i = 0; i < this.layerSizes.length - 1; i++) {
            this.weights[i] = add(this.weights[i], deltas[i]);
            console.log('new weights', JSON.stringify(this.weights[i], null, 2));
        }
        this.gradients = null;
        this.lastDelta = deltas;
    }

}
