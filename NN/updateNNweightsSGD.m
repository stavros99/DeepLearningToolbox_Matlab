function [nn, histData] = updateNNweightsSGD(nn, dW, dBiases, histData)
% updateNNweightsSGD - update weights using SGD

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
% histData: structure which contains the previous updates for weights,  sum
% of past gradients, sum of squares of past gradients, sum of squares of past updates


lr = nn.trParams.lrParams.lr;

for i = 1 : nn.noLayers
        
    gradW = dW{i};
    gradBiases = dBiases{i};
    weights = nn.W{i};
    biases = nn.biases{i};
    
    if nn.weightConstraints.weightPenaltyL1 > 0 || nn.weightConstraints.weightPenaltyL2 > 0
       
        gradW = applyWeightConstraints(gradW, weights, nn.weightConstraints.weightPenaltyL1, nn.weightConstraints.weightPenaltyL2);
   
    end
    
    updateW = -lr * gradW;
    
    updateBiases = -lr * gradBiases;
       
    newW = weights + updateW;
  
    if nn.weightConstraints.maxNormConstraint > 0
    
        newW = applyMaxNormRegularisation(newW, nn.weightConstraints.maxNormConstraint);
        updateW = newW - weights; % some weight vectors may have been rescaled
    end
    
    nn.W{i} = newW;
    nn.biases{i} = biases + updateBiases;
    
    histData.vW{i} = updateW;
    histData.vBiases{i} = updateBiases;
    
end





        

