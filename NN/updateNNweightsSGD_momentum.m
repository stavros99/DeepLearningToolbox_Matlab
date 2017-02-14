function [nn, histData] = updateNNweightsSGD_momentum(nn, dW, dBiases, histData)
% updateNNweightsSGD_momentum - update weights using SGD with momentum

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

lr = nn.trParams.lrParams.lr;
momentum = nn.trParams.momParams.momentum;

for i = 1 : nn.noLayers
        
    gradW = dW{i};
    gradBiases = dBiases{i};
    weights = nn.W{i};
    biases = nn.biases{i};
    
    vW = histData.vW{i};
    vBiases = histData.vBiases{i};
    
    if nn.weightConstraints.weightPenaltyL1 > 0 || nn.weightConstraints.weightPenaltyL2 > 0
       
        gradW = applyWeightConstraints(gradW, weights, nn.weightConstraints.weightPenaltyL1, nn.weightConstraints.weightPenaltyL2);
   
    end
    
    updateW = -lr * gradW;
    
    updateBiases = -lr * gradBiases;
        
    
    if nn.trParams.momParams.scaleLR == 0
            
            updateW = momentum * vW + updateW; % m*DW(t-1) - lr*dE/dW
            updateBiases = momentum * vBiases + updateBiases;
            
        
    elseif nn.trParams.momParams.scaleLR == 1
            

            updateW = momentum * vW + (1-momentum) * updateW; % m*DW(t-1) - (1-m)*lr*dE/dW
            updateBiases = momentum * vBiases + (1-momentum) * updateBiases;
            
     
    end
    
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





        

