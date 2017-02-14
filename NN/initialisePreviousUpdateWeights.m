function [vW, vBiases] = initialisePreviousUpdateWeights(w, biases)
% initialisePreviousUpdateWeights - Initialises variables that hold the
% previous weight update. Here they are initialised to zero.

% INPUTS:
% w: 1 x N cell array where N is the number of layers. Each
% cell contains the  weights for the corresponding layer. 
% biases: same as above but each layer contains the 
% biases. 

% OUTPUTS:
% vW: 1 x N cell array where N is the number of layers. Each
% cell contains the previous weight updates for the corresponding layer. 
% Used only if momentum is applied.
% vBiases: same as above but each layer contains the previous updates for the
% biases. Used only if momentum is applied.
% Here both vW and vBiases are initialised to zero matrices.

noLayers = length(w);

for i = 1:noLayers
   
    [r, c] = size(w{i});
    [rb, cb] = size(biases{i});
    
    vW{i} = zeros(r, c);
    vBiases{i} = zeros(rb, cb);
    
end