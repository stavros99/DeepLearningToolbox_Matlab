function gradW = applyWeightConstraints(gradW, weights, weightPenaltyL1, weightPenaltyL2)
% applyWeightConstrains - apply L1, L2 weight constraints

% INPUTS
% gradW: weight gradients, noWeights in previous layer X no weights in next layer
% weights: weight matrix, noWeights in previous layer X no weights in next layer
% weightPenaltyL1, value for L1 regulariser
% weightPenaltyL2, value for L2 regulariser

% OUTPUTS
% gradW: regularised weight gradients, noWeights in previous layer X no weights in next layer

    if weightPenaltyL2 > 0
    
        
       
        gradW = gradW + weightPenaltyL2 * weights;
        
    end
   
    if weightPenaltyL1 > 0
    
        
        gradW = gradW + weightPenaltyL1 * sign(weights);
        
    end