function [meanRatios, stdRatios] = computeWeightRatios(ratioUpdateWoverW)
% computeWeightRatios - Compute Ratio of weight updates norm of weight
% vector norm for each layer

% INPUTS
% ratioUpdateWoverW: batchSize x NoLayers, matrix which contains the ratio
% of the norm of the weight updates over the norm of all the weights in each layer.
% Each row contains the ratios for each example in the mini-batch, columns
% contains the ratios of each layer

% OUTPUTS
% meanRatios: 1xNoLayers vector, where each entry contains the mean ratio
% over the mini-batch for the corresponding layer
% stdRatios: 1xNoLayers vector, where each entry contains the standard
% deviation of ratios over the mini-batch for the corresponding layer

noLayers = size(ratioUpdateWoverW,2);
meanRatios = mean(ratioUpdateWoverW);
stdRatios = std(ratioUpdateWoverW);

disp(' ')
disp('Weight Statistics - Ratio should be between 0.01 and 0.0001')
    
for i = 1:noLayers
        
    disp(['Layer ',num2str(i),' Ratio of weight updates over weights magnitudes, mean / st.dev. over all mini-batches ',num2str(meanRatios(i)),' / ', num2str(stdRatios(i)) ])

end