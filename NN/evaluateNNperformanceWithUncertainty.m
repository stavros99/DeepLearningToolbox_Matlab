function [stats, output, e, L] = evaluateNNperformanceWithUncertainty( nn, data_x, data_y)

[noEx, inputDim] = size(data_x);

for i = 1:100
    dropoutMasks = createDropoutMasks(nn.layersSize(1:end-1), noEx, nn.dropoutParams.dropoutPresentProbHid, nn.dropoutParams.dropoutType);
    dropoutMaskInput = createDropoutMasks(inputDim, noEx, nn.dropoutParams.dropoutPresentProbVis, nn.dropoutParams.dropoutType);
    data_x2 = data_x .* dropoutMaskInput{1};
    nn.dropoutParams.dropoutMasks = dropoutMasks;
    [outputTemp, layerActivations] = simulateNN(nn, data_x2);
    outputMatrix(:,:,i) = outputTemp;
    [tempErr(:,:,i), tempL(i)] = computeLoss(data_y, outputTemp, nn.activation_functions{end});
    i
end

output = mean(outputTemp,3);
L = mean(tempL);
e = mean(tempErr,3);


stats = [];

if nn.type == 2%i.e., classification
        
    [m, targetsVec] = max(data_y,[], 2);
    
    [m2, predVec] = max(output,[], 2);
    
    nClasses = nn.layersSize(end);
    
    [matrix,p_row,p_col] = create_confusion_matrix(predVec,targetsVec,nClasses);
    
    stats = computeStatsFromConfMat(matrix);
    stats.confusionMatrix = matrix;
   
  
end


  

