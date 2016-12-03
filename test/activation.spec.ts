import { softmax } from '../src/activation'

describe('activation functions', () => {

    describe('softmax', () => {

        it('returns expected result', () => {
            // GIVEN Some values
            const xs = [1, 2, 3, 4, 1, 2, 3];
            // WHEN Calculate softmax
            const ys = softmax(xs);
            // THEN Returns expected result
            expect(ys).toBeAbout([0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175], 1e-3);
        });

        it('can handle empty list', () => {
            // WHEN Calculate softmax on empty list
            const ys = softmax([]);
            // THEN Returns empty list
            expect(ys).toEqual([]);
        });

    });

});
