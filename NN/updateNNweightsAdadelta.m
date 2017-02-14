function [nn, histData] = updateNNweightsAdadelta(nn, dW, dBiases, histData)
% updateNNweightsAdadelta - Update weights using Adadelta

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

epsilon = nn.trParams.adadelta.epsilon;
gamma = nn.trParams.adadelta.gamma;



for i = 1 : nn.noLayers
        
    gradW = dW{i};
    gradBiases = dBiases{i};
    weights = nn.W{i};
    biases = nn.biases{i};
    
    histGradW = histData.sum_gradW_sq{i};
    histGradBiases = histData.sum_gradBiases_sq{i};
    
    hist_vW = histData.sum_vW_sq_{i};
    hist_vBiases = histData.sum_vBiases_sq{i};
    
    if nn.weightConstraints.weightPenaltyL1 > 0 || nn.weightConstraints.weightPenaltyL2 > 0
       
        gradW = applyWeightConstraints(gradW, weights, nn.weightConstraints.weightPenaltyL1, nn.weightConstraints.weightPenaltyL2);
   
    end
      
    histGradW = gamma * histGradW + (1-gamma) * gradW .^ 2;
    histGradBiases = gamma * histGradBiases + (1-gamma) * gradBiases .^ 2;
    
    RMS_gradW = sqrt(histGradW + epsilon);
    RMS_gradBiases = sqrt(histGradBiases + epsilon);
    
    RMS_vW = sqrt(hist_vW + epsilon);
    RMS_vBiases = sqrt(hist_vBiases + epsilon);
    
    adaptedLR_w = RMS_vW ./ RMS_gradW;
    adaptedLR_biases = RMS_vBiases ./ RMS_gradBiases;
  
    updateW = -adaptedLR_w .* gradW;
    
    updateBiases = -adaptedLR_biases .* gradBiases;
        
    
    newW = weights + updateW;
  
    if nn.weightConstraints.maxNormConstraint > 0
    
        newW = applyMaxNormRegularisation(newW, nn.weightConstraints.maxNormConstraint);
        updateW = newW - weights; % some weight vectors may have been rescaled
    end
    
    nn.W{i} = newW;
    nn.biases{i} = biases + updateBiases;
    
    hist_vW = gamma * hist_vW + (1-gamma) * updateW .^ 2;
    hist_vBiases = gamma * hist_vBiases + (1-gamma) * updateBiases .^ 2;
    
    histData.sum_vW_sq_{i} = hist_vW;
    histData.sum_vBiases_sq{i} = hist_vBiases;
    
    histData.sum_gradW_sq{i} = histGradW;
    histData.sum_gradBiases_sq{i} = histGradBiases;
    
    histData.vW{i} = updateW;
    histData.vBiases{i} = updateBiases;
    
end





        

