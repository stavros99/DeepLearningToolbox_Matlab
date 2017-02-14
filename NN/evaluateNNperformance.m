function [stats, output, e, L] = evaluateNNperformance( nn, data_x, data_y)
% evaluateNNperformance - evaluate performance of network

% INPUTS
% nn: nn structure, check manual
% data_x: input data, noExamples x inputDimensionality
% data_y: targets, noExamples x noClasses
% 
% OUTPUTS
% stats: structure containing performance measure for classification
% output: network predictions, noExamples x noClasses
% e: error matrix, noExamples x noClasses
% L: scalar, loss value


[output, layerActivations] = simulateNN(nn, data_x);
[e, L] = computeLoss(data_y, output, nn.activation_functions{end});
   
stats = [];

if nn.type == 2%i.e., classification
        
    [m, targetsVec] = max(data_y,[], 2);
    
    [m2, predVec] = max(output,[], 2);
    
    nClasses = nn.layersSize(end);
    
    [matrix] = create_confusion_matrix(predVec,targetsVec,nClasses);
    
    stats = computeStatsFromConfMat(matrix);
    stats.confusionMatrix = matrix;
   
  
end


  

