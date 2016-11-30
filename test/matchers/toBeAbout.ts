const { abs } = Math;

/** If given input is a number, wrap it in a list. Otherwise identity */
const assureNbrList = (a: number|number[]) => typeof a === 'number' ? [a] : a;

/** Returns true if the difference between the given numbers is lower or equal to maxDifference */
const isAboutNbr = (a: number, b: number, maxDifference: number) => abs(a - b) <= maxDifference;

/**
 * Returns true if the differences of all numbers with the same indices in the given lists are lower or equal than
 * maxDifference
 */
const isAbout = (a: number[]|number, b: number[]|number, maxDifference: number) => {
    a = assureNbrList(a);
    b = assureNbrList(b);
    if (a.length !== b.length) {
        return false;
    }
    for (let i = 0, len = a.length; i < len; i++) {
        if (!isAboutNbr(a[i], b[i], maxDifference)) {
            return false;
        }
    }
    return true;
};

export const toBeAbout: jasmine.CustomMatcherFactory = (util, customEqualityTesters) => {
    return {
        compare: (actual: number|number[], expected: number|number[], maxDifference: number = 0) => {
            const pass = isAbout(actual, expected, maxDifference);
            const message = 'Expected difference to be ' + (pass ? '> ' : '<= ') + maxDifference;
            return { pass, message };
        }
    };
};

declare global {
    namespace jasmine {
        interface Matchers {
            toBeAbout(expected: number|number[], maxDifference: number)
        }
    }
}
