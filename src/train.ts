import { NeuralNetwork } from './nn';
import { add, minus, multiply } from './matrix';
import { randRange } from './random';

interface TrainingData {
    input: number[],
    target: number[],
}

interface TrainerConfig {
    net: NeuralNetwork,
    data: TrainingData[],
    learningRate: number,
    momentum: number,
    batchSize: number,
}

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

export class Trainer {

    readonly net: NeuralNetwork;
    readonly data: TrainingData[];
    learningRate: number;
    momentum: number;
    batchSize: number;

    private gradients: number[][];
    private lastDelta: number[][];
    private dataOrder: number[];

    /** Number of training iterations performed on the current training data batch */
    private batchIndex: number;

    constructor(config: TrainerConfig) {
        this.net = config.net;
        this.data = config.data;
        this.learningRate = config.learningRate;
        this.momentum = config.momentum;
        this.batchSize = config.batchSize;

        this.lastDelta = [];
        this.dataOrder = [];
        this.batchIndex = 0;
    }

    private nextData(): TrainingData {
        if (this.dataOrder.length === 0) {
            this.dataOrder = randRange(this.data.length);
        }
        return this.data[this.dataOrder.pop()];
    }

    /** Train the neural network */
    train() {
        if (this.data.length === 0) {
            return;
        }
        // Get next input / target pair
        const { input, target } = this.nextData();
        // Forward
        let outputs = this.net.forward(input);
        console.log('input transformation', JSON.stringify(outputs, null, 2));
        // Backward
        let outDeriv: number[];
        let grads: number[][] = [];
        for (let i = this.net.layerSizes.length - 1; i > 0; i--) {
            const size = this.net.layerSizes[i];
            const ins = outputs[i * 2 - 2];
            const out = outputs[i * 2 - 1];
            const outAct = outputs[i * 2];
            let errors: number[];
            // calculate error
            if (i === this.net.layerSizes.length - 1) { // errors for output layer
                errors = minus(target, outAct, size); // .map(e => e * 2 / size);
            } else { // errors for hidden layer
                errors = multiply(outDeriv, this.net.weights[i], 1, this.net.layerSizes[i + 1], this.net.layerSizes[i]);
            }
            console.log('errors', JSON.stringify(errors, null, 2));
            // apply activation derivative on outputs and multiply with error
            outDeriv = [];
            for (let j = 0; j < size; j++) {
                outDeriv.push(this.net.activation.derivative(out[j], outAct[j]) * errors[j]);
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
        // Update weights if batch is finished
        this.batchIndex += 1;
        if (this.batchIndex >= this.batchSize) {
            this.update();
            this.batchIndex = 0;
        }
    }

    private update() {
        const deltas: number[][] = [];
        // Calculate deltas
        for (let i = 0; i < this.net.layerSizes.length - 1; i++) {
            deltas.push(nextDelta(this.learningRate, this.momentum, this.gradients[i], this.lastDelta[i]));
        }
        // Update weights
        for (let i = 0; i < this.net.layerSizes.length - 1; i++) {
            this.net.weights[i] = add(this.net.weights[i], deltas[i]);
            console.log('new weights', JSON.stringify(this.net.weights[i], null, 2));
        }
        this.gradients = null;
        this.lastDelta = deltas;
    }

}