import { initList } from './matrix';

const { random, sqrt, log, cos, PI, floor } = Math;

/** Returns a pseudo random uniformly distributed number in the interval [-1, 1) */
export const rand = () => random() * 2 - 1;

/** Returns a pseudo random normally distributed number with given mean and standard deviation */
export const randNormal = (mean: number = 0, standardDeviation: number = 1) =>
    sqrt(-2 * log(1 - random())) * cos(2 * PI * random()) * standardDeviation + mean;

/** Returns a list of pseudo random uniformly distributed numbers in the interval [-1, 1) */
export const listRand = initList(rand);

/** Returns a list of pseudo random normally distributed numbers with given mean and standard deviation */
export const listRandNormal = (mean: number = 0, standardDeviation: number = 1) =>
    initList(() => randNormal(mean, standardDeviation));

/** Returns a list with numbers 0…(size-1) in random order */
export const randRange = (size: number) => {
    const range: number[] = [];
    for (let i = 0; i < size; i++) {
        range.push(i);
    }
    for (let i = size - 1; i > 0; i--) {
        const j = floor(random() * (i + 1)); // Random integer 0 <= j <= i
        // Exchange list elements with indices i and j
        let tmp = range[j];
        range[j] = range[i];
        range[i] = tmp;
    }
    return range;
};
