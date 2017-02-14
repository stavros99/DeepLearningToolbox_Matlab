function [w, b] = updateWeights4Nesterov(w, b, momentum, vW, vBiases)
% updateWeights4Nesterov - update weights when using Nesterov momentum
% update weihgts so gradients can be computed from w(t) + m*Dw(t-1)

%INPUTS

% w: 1xN cell array containing the weights of each layer
% b: 1xN cell array containing the biases of each layer
% momentum: scalar, momentum value
% vW: 1 x N cell array where N is the number of layers. Each
% cell contains the previous weight updates for the corresponding layer. 
% vBiases: same as above but each layer contains the previous updates for the
% biases. 

%OUTPUTS
% w: 1xN cell array containing the updated weights of each layer
% b: 1xN cell array containing the updated biases of each layer

noLayers = length(w);


for i = 1:noLayers
   
    w{i} = w{i} + momentum * vW{i};
    b{i} = b{i} + momentum * vBiases{i};
    
end