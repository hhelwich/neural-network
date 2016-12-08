import { activation } from './activation';

const { random, log, sqrt, cos, PI, exp } = Math;

/** Returns a pseudo random uniformly distributed number in the interval [-1, 1) */
const rand = () => random() * 2 - 1;

/** Returns a pseudo random normally distributed number with given mean and standard deviation */
const randNormal = (mean: number = 0, standardDeviation: number = 1) =>
    sqrt(-2 * log(1 - random())) * cos(2 * PI * random()) * standardDeviation + mean;

/** Returns a list with given size where all elements are created by the given function */
const initList = <T>(createElement: () => T) => (size: number) => {
    const list = <T[]>[];
    for (let i = 0; i < size; i++) {
        list.push(createElement());
    }
    return list;
};

/** Returns a list of zeros */
const listZeros = initList(() => 0);

/** Returns a list of pseudo random uniformly distributed numbers in the interval [-1, 1) */
const listRand = initList(rand);

/** Returns a list of pseudo random normally distributed numbers with given mean and standard deviation */
const listRandNormal = (mean: number = 0, standardDeviation: number = 1) =>
    initList(() => randNormal(mean, standardDeviation));

/** Use uniformly or normally distributed numbers in the initialization of the weights? */
const useRandNormalInit = true;

/** Returns a weight matrix with given dimension */
const createWeightMatrix = useRandNormalInit ?
    (rows: number, cols: number) => listRandNormal(0, rows ** -0.5)(rows * cols):
    (rows: number, cols: number) => listRand(rows * cols);

/** Returns matrix multiplication A * B */
const multiply = (A: number[], B: number[], rowsA: number, colsA: number, colsB: number) => {
    const C = listZeros(rowsA * colsB); // Init with zeros
    for (let i = 0; i < rowsA; i++) { // Iterate rows of C / A
        for (let k = 0; k < colsB; k++) { // Iterate columns of C / B
            for (let j = 0; j < colsA; j++) { // Iterate columns of A / rows of B
                C[i * colsB + k] += A[i * colsA + j] * B[j * colsB + k];
            }
        }
    }
    return C;
};

/** Combines two lists of the same size element wise with the given function */
const elementOp = (op: (a: number, b: number) => number) => (A: number[], B: number[], size: number = A.length) => {
    const result: number[] = [];
    for (let i = 0; i < size; i++) {
        result.push(op(A[i], B[i]));
    }
    return result;
};

/** Adds two lists */
const add = elementOp((a, b) => a + b);

/** Subtract list b from list a */
const minus = elementOp((a, b) => a - b);

const activationFn = activation.sigmoid;

export class NeuralNetwork {

    readonly layerSizes: number[];

    readonly weights: number[][];

    constructor(layerSizes: number[]) {
        this.layerSizes = layerSizes;
        this.weights = [];
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
    train(inputs: number[], targets: number[], learningRate: number) {
        // Forward
        let outputs = this.forward(inputs);
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
            // apply activation derivative on outputs and multiply with error
            outDeriv = [];
            for (let j = 0; j < size; j++) {
                outDeriv.push(activationFn.derivative(out[j], outAct[j]) * errors[j]);
            }
            // multiply with error
            grads[i - 1] = multiply(outDeriv, ins, size, 1, ins.length);
        }
        // Update weights
        for (let i = 0; i < this.layerSizes.length - 1; i++) {
            this.weights[i] = add(this.weights[i], grads[i].map(x => x * learningRate),
                this.layerSizes[i] * this.layerSizes[i + 1]);
        }
    }

}
