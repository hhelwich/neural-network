// Set up custom matchers

import { toBeAbout } from './matchers/toBeAbout'

beforeEach(() => {
    jasmine.addMatchers({ toBeAbout });
});

// Run top level specs

import './nn.spec';