function [meanActiv, stdActiv] = computeActivationStats(layerActivations, activFunctions)
% computeActivationStats - Compute the mean and standard deviation of
% activations per layer

% INPUTS
% layerActivations: 1xNoLayers, cell array which contains the neuron
% activations per layer
% activFunctions: 1xNoLayers, cell array which contains name of the
% activation function per layer

% OUTPUTS
% meanActiv: 1xNoLayers, vector which contains the mean activation of all
% neurons per layer (for all examples in a mini-batch)
% stdActiv: 1xNoLayers, vector which contains the standard deviation of activations of all
% neurons per layer (for all examples in a mini-batch)

noLayers = length(layerActivations);

disp(' ')
disp('Activation Statistics - Mean values should be different than 0')

for i = 1:noLayers
   
    meanActiv(i) = mean(layerActivations{i}(:));
    stdActiv(i) = std(layerActivations{i}(:));  
    
    disp(['Layer ',num2str(i),' (',activFunctions{i},') mean / st.dev of activations on training set ', num2str(meanActiv(i)), ' / ', num2str(stdActiv(i))])
    
end