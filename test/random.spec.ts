import { randRange } from '../src/random';

describe('random', () => {

    describe('randRange', () => {

        it('creates empty list for size 0', () => {
            // WHEN Called with size 0
            const range = randRange(0);
            // THEN Creates empty list
            expect(range).toEqual([]);
        });

        it('creates list [0] for size 1', () => {
            // WHEN Called with size 1
            const range = randRange(1);
            // THEN Creates single list with element 0
            expect(range).toEqual([0]);
        });

        it('creates list [0, 1] and [1, 0] for size 2', () => {
            let set = 0;
            for (;;) {
                // WHEN Creating ranges for size 2
                const range = randRange(2);
                // THEN Returns either [0, 1] or [1, 0] and both occur
                expect(range.length).toBe(2);
                if (range[0] === 0 && range[1] === 1) {
                    set |= 1;
                } else {
                    expect(range).toEqual([1, 0]);
                    set |= 2;
                }
                if (set === 3) {
                    return;
                }
            }
        });

        it('creates list with all numbers 0…19 for size 20', () => {
            // WHEN Called with size 20
            const range = randRange(20);
            // THEN Returns list which contains all numbers 0…19
            expect(range.length).toBe(20);
            for (let i = 0; i < 20; i++) {
                expect(range.indexOf(i)).not.toBe(-1);
            }
        });

    });

});
