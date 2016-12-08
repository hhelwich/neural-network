const { exp, max } = Math;

export interface ActivationFunction {
    map: (x: number) => number,
    derivative: (x: number, mapX: number) => number,
}

export const activation = {
    linear: <ActivationFunction>{
        map: x => x,
        derivative: () => 1,
    },
    sigmoid: <ActivationFunction>{
        map: x => 1 / (1 + exp(-x)),
        derivative: (_, sigmoidX) => sigmoidX * (1 - sigmoidX),
    },
    tanh: <ActivationFunction>{
        map: Math.tanh,
        derivative: (_, tanhX) => 1 - tanhX * tanhX,
    },
    relu: <ActivationFunction>{
        map: x => max(0, x),
        derivative: (_, reluX) => reluX === 0 ? 0 : 1,
    },
    softmax: (xs: number[]) => {
        const result: number[] = [];
        const len = xs.length;
        let sum = 0;
        // Calculate e^x for all x and also the sum of all e^x
        for (let i = 0; i < len; i++) {
            let expX = exp(xs[i]);
            result.push(expX);
            sum += expX;
        }
        // Calculate e^x / (sum of all e^x) for all x
        for (let i = 0; i < len; i++) {
            result[i] /= sum;
        }
        return result;
    },
};
