import { NeuralNetwork } from '../src/nn'

/** Create some test neural network with fixed dimension and weights */
const nnWithConstantWeights = () => {
    const nn = new NeuralNetwork([4, 3, 2]);
    nn.weights[0] = [ // (input -> hidden) 3x4
        1/16, -2/16, 3/16, -4/16,
        5/16, -9/16, 7/16, -8/16,
        2/16, -1/16, 8/16, -7/16,
    ];
    nn.weights[1] = [ // (hidden -> output) 2x3
        5/8, 4/8, -7/8,
        1/8, -2/8, 6/8,
    ];
    return nn;
};

describe('NeuralNetwork', () => {

    it('maps input to correct output', () => {
        // GIVEN Neural network
        const nn = nnWithConstantWeights();
        // WHEN Maps input
        const out = nn.map([1/4, 2/4, -2/4, 3/4]);
        // THEN Returns expected hand verified output
        const x = expect(out)
        x.toBeAbout([
            0.5256248130827825, 0.5607451182402815
        ], 1e-16);
    });

    it('trains the correct weights', () => {
        /* 
        Detailed calculation so it can be verified by hand:
        Input
        i = [0.25, 0.5, -0.5, 0.75]
        W0 = [0.0625, -0.125 , 0.1875, -0.25  ,
              0.3125, -0.5625, 0.4375, -0.5   ,
              0.125 , -0.0625, 0.5   , -0.4375]
        W1 = [0.625,  0.5 , -0.875,
              0.125, -0.25,  0.75 ]

        Forward
        a = W0 * i = [-0.328125, -0.796875, -0.578125]
        b = sigmoid(a) = [0.4186969093556867, 0.31069438321455395, 0.35936414516010196]
        c = W1 * b = [0.1025891329394919, 0.2441866267358988]
        d = sigmoid(c) = [0.5256248130827825, 0.5607451182402815]

        Backward change of weights
        e2 = [0.3, 0.7] - d = [-0.22562481308278254, 0.13925488175971845]
        f2 = e2 .* d .* (1 .- d) = [-0.05625805101378413, 0.034299874188824596]
        g2 = l * f2 = [-0.005625805101378414, 0.0034299874188824598]
        h3 = g2 * b^t = [-0.0023555072085845977, -0.0017479060460580574, -0.0020217126410941944,
                          0.0014361251314149752,  0.0010656778255433658,  0.0012326144966965998]
        W1' = W1 + h3 = [0.6226444927914154 ,  0.49825209395394193, -0.8770217126410942,
                         0.12643612513141497, -0.24893432217445663,  0.7512326144966966]


        e1 = W1'^t * e2 = [-0.1228771996477549, -0.14708335516680277, 0.30249066888997034]
        f1 = e1 .* b .* (1 .- b) = [-0.029907057962469022, -0.031499868992212646, 0.06963987256819562]
        g1 = l * f1 = [-0.0029907057962469024, -0.0031499868992212647, 0.006963987256819563]
        h1 = g1 * i^t = [-0.0007476764490617256, -0.0014953528981234512,  0.0014953528981234512, -0.002243029347185177 ,
                         -0.0007874967248053162, -0.0015749934496106324,  0.0015749934496106324, -0.0023624901744159488,
                          0.0017409968142048907,  0.0034819936284097813, -0.0034819936284097813,  0.005222990442614672 ]
        W0' = W0 + h1 = [0.06175232355093827, -0.12649535289812344, 0.18899535289812344, -0.2522430293471852 ,
                         0.3117125032751947 , -0.5640749934496107 , 0.43907499344961065, -0.5023624901744159 ,
                         0.1267409968142049 , -0.05901800637159022, 0.4965180063715902 , -0.43227700955738535]

        Forward again
        a' = W0' * i = [-0.3314895440207778, -0.8004187352616239, -0.570290514336078]
        b' = sigmoid(a') = [0.41787823835227145, 0.30993595446572014, 0.3611697928869481]
        c' = W1' * b' = [0.09786207175963835, 0.24700373627927819]
        d' = sigmoid(c') = [0.5244460111296307, 0.561438881398655]
        */
        // GIVEN Some neural network and input / output
        const nn = nnWithConstantWeights();
        const input = [1/4, 2/4, -2/4, 3/4];
        const output = [0.3, 0.7];
        // WHEN Trained for single input / output
        nn.train(input, output, 0.1);
        // THEN Learns input / output relation
        expect(nn.map(input)).toBeAbout([0.5244460111296307,0.561438881398655], 1e-16);
    });

    it('can learn input output relation', () => {
        // GIVEN Some neural network and input / output
        const nn = nnWithConstantWeights();
        const input = [1/4, 2/4, -2/4, 3/4];
        const output = [0.3, 0.7];
        // WHEN Trained for single input / output
        for (let i = 0; i < 1000; i++) {
            nn.train(input, output, 0.05);
        }
        // THEN Learns input / output relation
        expect(nn.map(input)).toBeAbout(output, 0.02);
    });

});
