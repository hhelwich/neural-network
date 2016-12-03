const { abs, max } = Math;

/** If given input is a number, wrap it in a list. Otherwise identity */
const assureNbrList = (a: number|number[]) => typeof a === 'number' ? [a] : a;

/** Returns the maximum difference for all number pairs */
const maxDiff = (a: number[]|number, b: number[]|number) => {
    a = assureNbrList(a);
    b = assureNbrList(b);
    if (a.length !== b.length) {
        throw Error('Compared lists must have the same length');
    }
    const diff = a.map((value, i) => abs(value - b[i]));
    return max(0, ...diff);
};

export const toBeAbout: jasmine.CustomMatcherFactory = (util, customEqualityTesters) => {
    return {
        compare: (actual: number|number[], expected: number|number[], maxDifference: number = 0) => {
            const diff = maxDiff(actual, expected);
            const pass = diff <= maxDifference;
            const message = 'Expected max difference to be ' + (pass ? '> ' : '<= ') + maxDifference + ' but was ' + diff;
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
