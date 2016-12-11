import { NeuralNetwork } from '../../net';
import { Trainer } from '../../train';

import { run } from '@cycle/xstream-run';
import { makeDOMDriver, div, button, input } from '@cycle/dom';
//import xs from 'xstream';
import { Stream } from 'xstream';



export interface SinusPredictionViewModel {

}

/*
const step = (time: number) => {
    requestAnimationFrame(step);
};
requestAnimationFrame(step);*/

export const makeCanvasDriver = (selector: string) => {

    const canvas = document.createElement('canvas');
    document.querySelector(selector).appendChild(canvas);

    return (drawThis$: Stream<SinusPredictionViewModel>) => {
        drawThis$.addListener({
            next: viewModel => {
                
            },
            error: () => {},
            complete: () => {},
        });
        // Driver has no output => return nothing
    };
};
