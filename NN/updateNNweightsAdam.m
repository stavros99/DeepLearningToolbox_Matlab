function [nn, histData] = updateNNweightsAdam(nn, dW, dBiases, histData)
% UPDATE COMMENTS

% INPUTS:
% nn: neural network structure, see manual for details
% dW: 1 x N cell array where N is the number of layers. Each
% cell contains the weight gradients for the corresponding layer 
% dBiases: same as above but each layer contains the gradients for the
% biases


% OUTPUTS:
% nn: neural network structure, see manual for details

lr = nn.trParams.lrParams.lr;


epsilon = nn.trParams.adam.epsilon;
b1 = nn.trParams.adam.b1;
b2 = nn.trParams.adam.b2;


for i = 1 : nn.noLayers
        
    gradW = dW{i};
    gradBiases = dBiases{i};
    weights = nn.W{i};
    biases = nn.biases{i};
    
    mW = histData.sum_gradW{i};
    mBiases = histData.sum_gradBiases{i};
    
    uW = histData.sum_gradW_sq{i};
    uBiases = histData.sum_gradBiases_sq{i};
    
    if nn.weightConstraints.weightPenaltyL1 > 0 || nn.weightConstraints.weightPenaltyL2 > 0
       
        gradW = applyWeightConstraints(gradW, weights, nn.weightConstraints.weightPenaltyL1, nn.weightConstraints.weightPenaltyL2);
   
    end
      
    mW = b1 * mW + (1-b1) * gradW;
    mBiases = b1 * mBiases + (1-b1) * gradBiases;
    
    uW = b2 * uW + (1-b2) * gradW .^ 2;
    uBiases = b2 * uBiases + (1-b2) * gradBiases .^ 2;
    
    mW_hat = mW / (1 - b1);
    mBiases_hat = mBiases / (1 - b1);
    
    uW_hat = uW / (1 - b2);
    uBiases_hat = uBiases / (1 - b2);
    
  
    updateW = - lr .* mW_hat ./ (sqrt(uW_hat) + epsilon);
    
    updateBiases = - lr .* mBiases_hat ./ (sqrt(uBiases_hat) + epsilon);
    
    
    newW = weights + updateW;
  
    if nn.weightConstraints.maxNormConstraint > 0
    
        newW = applyMaxNormRegularisation(newW, nn.weightConstraints.maxNormConstraint);
        updateW = newW - weights; % some weight vectors may have been rescaled
    end
    
    nn.W{i} = newW;
    nn.biases{i} = biases + updateBiases;
    
    histData.sum_gradW{i} = mW;
    histData.sum_gradBiases{i} = mBiases;
    
    histData.sum_gradW_sq{i} = uW;
    histData.sum_gradBiases_sq{i} = uBiases;
    
    histData.vW{i} = updateW;
    histData.vBiases{i} = updateBiases;
    
end





        

