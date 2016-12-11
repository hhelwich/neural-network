import { NeuralNetwork } from '../../net';
import { Trainer } from '../../train';

import { run } from '@cycle/xstream-run';
import { makeDOMDriver, div, button, input } from '@cycle/dom';
import xs from 'xstream';
import { Stream } from 'xstream';

import { makeCanvasDriver } from './canvas';
import { makeNetDriver } from './net';


const main = ({ DOM }) => {
    const foo$ = DOM.select('.learningRateRange').events('input').map(ev => ev.target.value).startWith('');
    //foo$.debug('foooo');
    console.log(foo$);
    //.debug(ev => {console.log(ev);})
    const add$ = DOM.select('.add').events('click').map(ev => 1);
    const count$ = add$.fold((total, change) => total + change, 0);

    return {
        Canvas: xs.create({
            start: listener => {

            },
            stop: () => {},
        }),
        DOM: foo$.map(count => 
            div('.foooo', [
                div('.bar', [
                    'Learning rate',
                    input('.learningRateRange', {attrs: {type: 'range', min: '0.01', max: '1', step: '0.01'}}),
                    count
                ]),
                button('.add', 'Add'),
                button('.foo', 'fooo'), 
                'Toggle me'
            ])
        ),
    };
}



const sources = {
    DOM: makeDOMDriver('.app'),
    Canvas: makeCanvasDriver('.canvas-view'),
    Net: makeNetDriver({}),
};

run(main, sources);
