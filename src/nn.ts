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

const fooobar = elementOp((a, b) => a * b * (1 - b));

const fooo = (weights: number[], inputs: number[], inputSize: number, outputSize: number) => {
    return multiply(weights, inputs, outputSize, inputSize, 1).map(sigmoid);
};

const weightInc = (errors: number[], finalOutputs: number[], hiddenOutputs: number[],
        outputSize: number, hiddenSize: number, learningRate: number) => {
    const foo = fooobar(errors, finalOutputs);
    const bar = foo.map(f => learningRate * f);
    const baz = multiply(bar, hiddenOutputs, outputSize, 1, hiddenSize);
    return baz;
};

/** Sigmoid function */
const sigmoid = (x: number) => 1 / (1 + exp(-x));

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

    private mapLayer(inputs: number[], layerIndex: number) {
        return fooo(this.weights[layerIndex], inputs, this.layerSizes[layerIndex], this.layerSizes[layerIndex + 1]);
    }

    /** Calculate the outputs from the inputs */
    map(inputs: number[]) {
        for (let i = 0, len = this.layerSizes.length - 1; i < len; i++) {
            inputs = this.mapLayer(inputs, i);
        }
        return inputs;
    }

    /** Train the neural network */
    train(inputs: number[], targets: number[], learningRate: number) {
        // Calculate output of each layer
        let outputs: number[][] = [inputs];
        for (let i = 0, len = this.layerSizes.length - 1; i < len; i++) {
            outputs.push(this.mapLayer(outputs[i], i));
        }

        let n = this.layerSizes.length - 1;

        // calculate error
        let outputErrors = minus(targets, outputs[n], this.layerSizes[n]);
        // hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        const baz = weightInc(outputErrors, outputs[n], outputs[n - 1], this.layerSizes[n], this.layerSizes[n - 1],
            learningRate);
        this.weights[n-1] = add(this.weights[n-1], baz);


        for (let i = n-1; i > 0; i--) {
            const hiddenErrors = multiply(outputErrors, this.weights[n-1], 1, this.layerSizes[n], this.layerSizes[n - 1]);
            n -= 1;
            const bazz = weightInc(hiddenErrors, outputs[n], outputs[n-1], this.layerSizes[n], this.layerSizes[n-1],
                learningRate);
            this.weights[n-1] = add(this.weights[n-1], bazz);
        }
    }

}
