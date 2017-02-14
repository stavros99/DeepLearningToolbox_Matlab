function [output, layerActivations] = simulateNN(nn, x)
% simulateNN - Simulate neural network

% INPUTS
% nn: neural network structure, see manual for details
% x: matrix containing input examples, noExamples x InputDimensionality

% OUTPUTS
% output of the network: noExamples x TargetDimensionality
% layerActivations: 1 x N-1 cell array where N is the number of layers. Each
% cell contains the activations of the corresponding layer.



weights = nn.W;
biases = nn.biases;
activation_functions = nn.activation_functions;
layerActivations = [];

noLayers = nn.noLayers;
N = size(x,1);

x = [x ones(N,1)];

for layerId = 1:noLayers

    w = [weights{layerId}; biases{layerId}];
    inp2Hid = x * w; 
    
    % if gpu is available the following piece of code can be used instead
    % of line 28
%     g = gpuDevice(1);
%     xGpu = gpuArray(x);
%     wGpu = gpuArray(w);
%     inp2HidGpu = xGpu * wGpu;
%     
%     inp2Hid = gather(inp2HidGpu);
%     reset(g)
      
    a = computeActivations(activation_functions{layerId}, inp2Hid);

    % if dropout is used AND layerId~=noLayers (i.e., for hidden layers only) AND nn is not used for
    % testing, i.e. we are in the training phase
    if  nn.dropoutParams.dropoutType ~= 0 && layerId ~= noLayers && nn.testing == 0
            
        %dropout
        a = a .* nn.dropoutParams.dropoutMasks{layerId};
            
    end
    
    % save activations only when NN is in training phase or in diagnostics
    % mode
    if (nn.testing == 0 || nn.diagnostics == 1) 
        layerActivations{layerId} = a;
            
    end
    
    a = [a  ones(N,1)];
    x = a;   
   
end

output = a;
output(:,end) = []; % remove last column, it's only ones
  
   
   

   
 
   