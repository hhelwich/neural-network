import { NeuralNetwork } from '../src/net';
import { Trainer } from '../src/train';

const { round } = Math;

describe('NeuralNetwork examples', () => {

    it('can learn XOR', () => {
        // GIVEN Neural network and training data for XOR
        const net = new NeuralNetwork([2, 6, 1]);
        const trainer = new Trainer({
            net,
            data: [
                { input: [0, 0], target: [0] },
                { input: [0, 1], target: [1] },
                { input: [1, 0], target: [1] },
                { input: [1, 1], target: [0] },
            ],
            batchSize: 50,
            learningRate: 0.3,
            momentum: 0.9,
        });
        const xor = (a: number, b: number) => round(net.map([a, b])[0]); 
        // WHEN Training for XOR
        for (let i = 1;; i++) {
            trainer.train();
            // THEN At some point the net learns XOR
            if (i % 100 === 0 &&
                    xor(0, 0) === 0 && xor(0, 1) === 1 && xor(1, 0) === 1 && xor(1, 1) === 0) {
                console.info('learned XOR after ' + i + ' iterations');
                expect(true).toBe(true);
                return;
            }
            if (i > 100000) throw Error('too much iterations');
        }
    });

    it('can learn to count', () => {
        // GIVEN Neural network and training data for incrementing a 3 bit integer
        const net = new NeuralNetwork([3, 10, 3]);
        const trainer = new Trainer({
            net,
            data: [
                { input: [0, 0, 0], target: [0, 0, 1] },
                { input: [0, 0, 1], target: [0, 1, 0] },
                { input: [0, 1, 0], target: [0, 1, 1] },
                { input: [0, 1, 1], target: [1, 0, 0] },
                { input: [1, 0, 0], target: [1, 0, 1] },
                { input: [1, 0, 1], target: [1, 1, 0] },
                { input: [1, 1, 0], target: [1, 1, 1] },
                { input: [1, 1, 1], target: [0, 0, 0] },
            ],
            batchSize: 50,
            learningRate: 0.3,
            momentum: 0.9,
        });
        const inc = (n: number) => parseInt(net.map([(n&4)>>2, (n&2)>>1, n&1]).map(round).join(''), 2);
        // WHEN Train counting
        for (let i = 1;; i++) {
            trainer.train();
            // THEN At some point the net learns to count
            if (i % 100 === 0 &&
                    inc(0) === 1 && inc(1) === 2 && inc(2) === 3 && inc(3) === 4 &&
                    inc(4) === 5 && inc(5) === 6 && inc(6) === 7 && inc(7) === 0){
                console.info('learned counting after ' + i + ' iterations');
                expect(true).toBe(true);
                return;
            }
            if (i > 100000) throw Error('too much iterations');
        }
    });

});
