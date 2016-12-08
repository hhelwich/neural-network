import { NeuralNetwork } from '../src/nn'

/** Create some fixed random neural network */
const nnWithConstantWeights = () => {
    const nn = new NeuralNetwork([5, 4, 3, 2]);
    nn.weights[0] = [ // (input -> first hidden) 4x5
         0.5619, 0.4189, -1.9511, -0.0332,  1.4653,
         1.3524, 0.8693, -1.0953, -1.1414,  0.7953,
        -0.3169, 0.3326, -0.9113, -0.7058, -1.0113,
         0.6004, 0.2971,  1.6375,  0.5248, -1.7437,
    ];
    nn.weights[1] = [ // (first hidden -> second hidden) 3x4
        -1.3182, -0.2256, -1.6695,  0.0026,
         1.8931,  0.5505,  1.5227,  1.5385,
         0.0212,  0.1345,  0.7466, -0.3201,
    ];
    nn.weights[2] = [ // (second hidden -> output) 2x3
        0.4546, -0.0013, -0.5559,
        0.9146, -0.9927, -3.3979,
    ];
    return nn;
};

describe('NeuralNetwork', () => {

    it('maps inputs to expected outputs', () => {
        // GIVEN Neural network and inputs
        const nn = nnWithConstantWeights();
        const input = [1.4485, -0.9515, -1.6606, -1.4564, 2.7283];
        // WHEN Neural network maps input
        const output = nn.map(input);
        // THEN Returns expected verified output
        expect(output).toEqual([0.4319690615865325, 0.05581886824957501]);
    });

    it('trains the weights as expected', () => {
        // GIVEN Neural network, inputs and target outputs
        const nn = nnWithConstantWeights();
        const input = [1.4485, -0.9515, -1.6606, -1.4564, 2.7283];
        const target = [-1.2261, -0.9901];
        // WHEN Neuraln network is trained for single input / target
        nn.train(input, target, 0.2);
        // THEN Weights are adapted correctly
        expect(nn.weights[0]).toEqual([
             0.5619052651108594 , 0.4188965414201017 , -1.95110603606703  , -0.033205293826341443,  1.4653099170189559,
             1.3524067581247021 , 0.8692955606795623 , -1.0953077476989164, -1.141406794982959   ,  0.7953127291623229,
            -0.31001342637257234, 0.32807630320573183, -0.919194956275945 , -0.7127241324342324  , -0.9983288996702031,
             0.6003961925596188 , 0.2971025010559356 ,  1.637504364953743 ,  0.5248038282058484  , -1.7437071714460421,
        ]);
        expect(nn.weights[1]).toEqual([
            -1.3231623329584876  , -0.23055895789437617, -1.6708416358612126,  0.0025976469553844897,
             1.8936687201488696  ,  0.5510683333414989 ,  1.522853761416878 ,  1.538500269676359    ,
             0.041218830375240614,  0.1545052148366387 ,  0.7520123697373062, -0.3200905074485293   ,
        ]);
        expect(nn.weights[2]).toEqual([
            0.4448502373545206, -0.07823757983099752, -0.603773268379824,
            0.9132790051165085, -1.0031242690822928 , -3.40438634688713 ,
        ]);
    });

    it('maps to expected outputs after training', () => {
        // GIVEN Some neural network, inputs and target outputs
        const nn = nnWithConstantWeights();
        const input = [1.4485, -0.9515, -1.6606, -1.4564, 2.7283];
        const target = [-1.2261, -0.9901];
        // WHEN Trained for single inputs and target outputs pair
        nn.train(input, target, 0.2);
        // THEN Neural network maps random different inputs to verified output
        expect(nn.map([-0.1499, 1.138, 1.4641, -0.5351, -0.9625])).toEqual([0.4278389776217402, 0.07158063410506454]);
    });

});
