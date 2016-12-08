import { initList } from './matrix';

const { random, sqrt, log, cos, PI } = Math;

/** Returns a pseudo random uniformly distributed number in the interval [-1, 1) */
const rand = () => random() * 2 - 1;

/** Returns a pseudo random normally distributed number with given mean and standard deviation */
const randNormal = (mean: number = 0, standardDeviation: number = 1) =>
    sqrt(-2 * log(1 - random())) * cos(2 * PI * random()) * standardDeviation + mean;

/** Returns a list of pseudo random uniformly distributed numbers in the interval [-1, 1) */
export const listRand = initList(rand);

/** Returns a list of pseudo random normally distributed numbers with given mean and standard deviation */
export const listRandNormal = (mean: number = 0, standardDeviation: number = 1) =>
    initList(() => randNormal(mean, standardDeviation));
