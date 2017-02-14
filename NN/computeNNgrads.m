function [dW,dBiases, dX] = computeNNgrads(nn, x, e, layerActivations)
% computeGradients - Computes NN gradients

% INPUTS:
% nn: neural network structure, see manual for details
% x: matrix containing input examples (noExamples x InputDimensionality)
% e: error Matrix, error per example and per output, noExamples x targetDimensiolaity
% layerActivations: 1 x N cell array where N is the number of layers. Each
% cell contains the activations of the corresponding layer and the last
% cell is the output of the network

% OUTPUTS: 
% dW: 1 x N cell array where N is the number of layers. Each
% cell contains the weight gradients for the corresponding layer 
% dBiases: same as above but each layer contains the gradients for the
% biases
% dX: contains the gradients wrt inputs, size: noInputs x batchSize
    
n = nn.noLayers;
    
outputActivFcn = nn.activation_functions{end};
    
if strcmpi(outputActivFcn, 'softmax')
   
    d = e;
else
    
    d_act = computeActivFcnDeriv(layerActivations{end}, outputActivFcn);
    d = e .* d_act;
    
end

N = size(x, 1);

for i = n : -1 : 2
    
    a = layerActivations{i-1};
    aBias = ones(N, 1);

    dW{i} = (a' * d) / N; % divide by batchsize
    dBiases{i} = (aBias' * d) / N;
        
    
    w = nn.W{i};
    d_act = computeActivFcnDeriv(a, nn.activation_functions{i-1});
        
    % Backpropagate first derivatives        
    d_prevLayer = (d * w')  .* d_act; 

    % for bernoulli dropout "kill" the error in the neurons that have
    % been dropped
    if nn.dropoutParams.dropoutType == 1 
        d_prevLayer = d_prevLayer .* nn.dropoutParams.dropoutMasks{i - 1};
    end
    
    d = d_prevLayer;
        
end

dX = nn.W{1}*d';

aBias = ones(N, 1);
dW{1} = (x' * d) / N;
dBiases{1} = (aBias' * d) / N;
       
    
