function [nn, histData] = updateWeights(nn, dW, dBiases, histData)
% updateWeights - update weights 

% INPUTS:
% nn: neural network structure, see manual for details
% dW: 1 x N cell array where N is the number of layers. Each
% cell contains the weight gradients for the corresponding layer 
% dBiases: same as above but each layer contains the gradients for the
% biases
% histData: structure which contains the previous updates for weights,  sum
% of past gradients, sum of squares of past gradients, sum of squares of past updates

% OUTPUTS:
% nn: neural network structure, see manual for details
% histData: updated structure, see above
        
switch nn.trainingMethod        

    case 1 % SGD
        
        [nn, histData] = updateNNweightsSGD(nn, dW, dBiases, histData);

    case 2 % SGD with momentum
        
        [nn, histData] = updateNNweightsSGD_momentum(nn, dW, dBiases, histData);

    case 3 % SGD with nesterov momentum
       
        [nn, histData] = updateNNweightsSGD_momentum(nn, dW, dBiases, histData);
        
    case 4 % Adagrad
        
        [nn, histData] = updateNNweightsAdagrad(nn, dW, dBiases, histData);
    case 5 % Adadelta
         
        [nn, histData] = updateNNweightsAdadelta(nn, dW, dBiases, histData);
    case 6 % RMSprop
        
        [nn, histData] = updateNNweightsRMSprop(nn, dW, dBiases, histData);
    case 7 % Adam

        [nn, histData] = updateNNweightsAdam(nn, dW, dBiases, histData);
end
        