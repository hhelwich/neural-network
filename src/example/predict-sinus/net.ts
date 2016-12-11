import { NeuralNetwork } from '../../net';
import { Trainer } from '../../train';

import { queue } from './queue';

import { run } from '@cycle/xstream-run';
import { makeDOMDriver, div, button, input } from '@cycle/dom';
import xs from 'xstream';
import { Stream } from 'xstream';

export interface NetModel {

}

export interface Config {

}

export const makeNetDriver = (config: Config) => {
    const net = new NeuralNetwork([3, 3, 1]);
    const trainer = new Trainer({
        net,
        data: [],
        batchSize: 1,
        learningRate: 0.1,
        momentum: 0.9,
    });
    queue(() => {
        console.log('fooo');
    });

    return (drawThis$: Stream<NetModel>) => {
        drawThis$.addListener({
            next: viewModel => {
                
            },
            error: () => {},
            complete: () => {},
        });
        return xs.create({
            start: listener => {

            },
            stop: () => {},
        });
    };
};
