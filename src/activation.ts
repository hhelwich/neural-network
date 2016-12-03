const { exp, max } = Math;

// Some activation functions

export const linear = (x: number) => x;

export const step = (x: number) => x >= 0.5 ? 1 : 0;

export const sigmoid = (x: number) => 1 / (1 + exp(-x));

export const tanh = Math.tanh;

export const relu = (x: number) => max(0, x);

export const softmax = (xs: number[]) => {
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
};

// Derivatives of activation functions

export const sigmoidDerivativeFromSigmoid = (sigmoidX: number) => sigmoidX * (1 - sigmoidX);

export const sigmoidDerivative = (x: number) => sigmoidDerivativeFromSigmoid(sigmoid(x));
