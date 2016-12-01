import { NeuralNetwork } from '../src/nn'

/** Create some test neural network with fixed dimension and weights */
const nnWithConstantWeights = () => new NeuralNetwork({
    inputSize: 4,
    hiddenSize: 3,
    outputSize: 2,
    weights: {
        inputHidden: [ // 3x4
            1/16, -2/16, 3/16, -4/16,
            5/16, -9/16, 7/16, -8/16,
            2/16, -1/16, 8/16, -7/16,
        ],
        hiddenOutput: [ // 2x3
            5/8, 4/8, -7/8,
            1/8, -2/8, 6/8,
        ],
    },
});

describe('NeuralNetwork', () => {

    it('maps input to correct output', () => {
        // GIVEN Neural network
        const nn = nnWithConstantWeights();
        // WHEN Maps input
        const out = nn.map([1/4, 2/4, -2/4, 3/4]);
        // THEN Returns expected hand verified output
        const x = expect(out)
        x.toBeAbout([
            0.5256248130827825, 0.5607451182402815
        ], 1e-16);
    });

    it('can learn input output relation', () => {
        // GIVEN Some neural network and input / output
        const nn = nnWithConstantWeights();
        const input = [1/4, 2/4, -2/4, 3/4];
        const output = [0.3, 0.7];
        // WHEN Trained for single input / output
        for (let i = 0; i < 1000; i++) {
            nn.train(input, output, 0.05);
        }
        // THEN Learns input / output relation
        expect(nn.map(input)).toBeAbout(output, 0.02);
    });

});
