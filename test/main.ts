// Set up custom matchers

import { toBeAbout } from './matchers/toBeAbout'

beforeEach(() => {
    jasmine.addMatchers({ toBeAbout });
});

// Run top level specs

import './random.spec';
import './activation.spec';
import './net.spec';
import './net-examples.spec';
