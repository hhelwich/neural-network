/** Returns a list with given size where all elements are created by the given function */
export const initList = <T>(createElement: () => T) => (size: number) => {
    const list = <T[]>[];
    for (let i = 0; i < size; i++) {
        list.push(createElement());
    }
    return list;
};

/** Returns a list of zeros */
const listZeros = initList(() => 0);

/** Returns matrix multiplication A * B */
export const multiply = (A: number[], B: number[], rowsA: number, colsA: number, colsB: number) => {
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
export const elementOp = (op: (a: number, b: number) => number) => (A: number[], B: number[], size: number = A.length) => {
    const result: number[] = [];
    for (let i = 0; i < size; i++) {
        result.push(op(A[i], B[i]));
    }
    return result;
};

/** Adds two lists */
export const add = elementOp((a, b) => a + b);

/** Subtract list b from list a */
export const minus = elementOp((a, b) => a - b);
