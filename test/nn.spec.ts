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
        const target = [0.2261, 0.9901];
        // WHEN Neuraln network is trained for single input / target
        nn.train(input, target)
        nn.update(0.2);
        // THEN Weights are adapted correctly
        expect(nn.weights[0]).toEqual([
             0.5618988821239669 , 0.41890073431760133, -1.9510987184363544, -0.03319887602716283,  1.4652978944417114,
             1.3523978894463964 , 0.869301386394031  , -1.095297580403649 , -1.1413978779356104 ,  0.7952960246990702,
            -0.31877903219499076, 0.333834310758394  , -0.9091458261215039, -0.7039107197177876 , -1.0148392223248832,
             0.6004009515055418 , 0.2970993749689175 ,  1.637498909168034 ,  0.5247990433050252 , -1.7436982078063035,
        ]);
        expect(nn.weights[1]).toEqual([
            -1.3177346763381326 , -0.22513499282177118, -1.669374193264125 ,  0.0026002206476965946,
             1.8925975221940148 ,  0.5499978639475457 ,  1.5225641481235515,  1.5384997617345095   ,
             0.01445890582046595,  0.12776349068513088,  0.7447774512571541, -0.32010319649960906  ,
        ]);
        expect(nn.weights[2]).toEqual([
            0.4533894506127533, -0.010852718718115814, -0.5618440375945524,
            0.915779996491363 , -0.9833883810862168  , -3.3921059831841682,
        ]);
    });

    it('double trains the weights as expected', () => {
        // GIVEN Neural network, inputs and target outputs
        const nn = nnWithConstantWeights();
        const input1 = [1.4485, -0.9515, -1.6606, -1.4564, 2.7283];
        const target1 = [0.2261, 0.9901];
        const input2 = [0.1145, 1.2747, 0.2471, 0.129, -0.8889];
        const target2 = [0.1962, 0.1614];
        // WHEN Neuraln network is trained for single input / target
        nn.train(input1, target1);
        nn.train(input2, target2);
        nn.update(0.2);
        // THEN Weights are adapted correctly
        expect(nn.weights[0]).toEqual([
             0.561913737078255  , 0.4190661108261694 , -1.9510666602773625, -0.03318213987779021,  1.4651825706961508,
             1.352403681017639  , 0.8693658625151047 , -1.095285081737675 , -1.1413913529339483 ,  0.7952510628852925,
            -0.31874621968958894, 0.33419960421372774, -0.9090750141556978, -0.7038737519169419 , -1.0150939562641994, 0.6003986634238226  , 0.29707390232466074,  1.637493971307835 ,  0.5247964654662325 , -1.7436804446985292,
        ]);
        expect(nn.weights[1]).toEqual([
            -1.3178737487883867  , -0.22545379969168694, -1.6698067646042891,  0.0020502627905242448,
             1.892588294711716   ,  0.5499767110539031 ,  1.5225354469366739,  1.5384632719309377   ,
             0.014615805645343503,  0.12812316466777873,  0.7452654729134544, -0.31948274082112904  ,
        ]);
        expect(nn.weights[2]).toEqual([
            0.45145012677379576, -0.02231145860391669, -0.5687466979193039,
            0.9159647624862414 , -0.9822966678650817 , -3.391448343198215 ,
        ]);
    });

    it('maps to expected outputs after training', () => {
        // GIVEN Some neural network, inputs and target outputs
        const nn = nnWithConstantWeights();
        const input = [1.4485, -0.9515, -1.6606, -1.4564, 2.7283];
        const target = [-1.2261, -0.9901];
        // WHEN Trained for single inputs and target outputs pair
        nn.train(input, target);
        nn.update(0.2);
        // THEN Neural network maps random different inputs to verified output
        expect(nn.map([-0.1499, 1.138, 1.4641, -0.5351, -0.9625])).toEqual([0.4278389776217402, 0.07158063410506454]);
    });

});
