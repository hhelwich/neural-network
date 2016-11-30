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

interface NeuralNetworkConfig {
    inputSize: number,
    hiddenSize: number,
    outputSize: number,
    learningRate: number,
    weights?: {
        inputHidden?: number[],
        hiddenOutput?: number[],
    },
}

export class NeuralNetwork {

    readonly inputSize: number;
    readonly hiddenSize: number;
    readonly outputSize: number;

    readonly learningRate: number;

    private weightsInputHidden: number[];
    private weightsHiddenOutput: number[];

    constructor(config: NeuralNetworkConfig) {
        this.inputSize = config.inputSize;
        this.hiddenSize = config.hiddenSize;
        this.outputSize = config.outputSize;
        this.learningRate = config.learningRate;

        if (config.weights && config.weights.inputHidden) {
            this.weightsInputHidden = config.weights.inputHidden;
        } else {
            this.weightsInputHidden = createWeightMatrix(config.hiddenSize, config.inputSize);
        }
        if (config.weights && config.weights.hiddenOutput) {
            this.weightsHiddenOutput = config.weights.hiddenOutput;
        } else {
            this.weightsHiddenOutput = createWeightMatrix(config.outputSize, config.hiddenSize);
        }
    }

    /** Calculate the hidden outputs from the inputs */
    private hiddenOutputs(inputs: number[]) {
        return fooo(this.weightsInputHidden, inputs, this.inputSize, this.hiddenSize);
    }

    /** Calculate the final outputs from the hidden outputs */
    private finalOutputs(hidden: number[]) {
        return fooo(this.weightsHiddenOutput, hidden, this.hiddenSize, this.outputSize);
    }

    /** Calculate the outputs from the inputs */
    map(inputs: number[]) {
        return this.finalOutputs(this.hiddenOutputs(inputs));
    }


    /** Train the neural network */
    train(inputs: number[], targets: number[]) {
        const hiddenOutputs = this.hiddenOutputs(inputs);
        const finalOutputs = this.finalOutputs(hiddenOutputs);

        // calculate error
        const outputErrors = minus(targets, finalOutputs, this.outputSize);
        // hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        const hiddenErrors = multiply(outputErrors, this.weightsHiddenOutput, 1, this.outputSize, this.hiddenSize);
        const baz = weightInc(outputErrors, finalOutputs, hiddenOutputs, this.outputSize, this.hiddenSize, this.learningRate);
        this.weightsHiddenOutput = add(this.weightsHiddenOutput, baz);

        const bazz = weightInc(hiddenErrors, hiddenOutputs, inputs, this.hiddenSize, this.inputSize, this.learningRate);
        this.weightsInputHidden = add(this.weightsInputHidden, bazz);
    }

}
