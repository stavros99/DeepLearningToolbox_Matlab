function dropoutMasks = createDropoutMasks(layersSize, batchSize, dropoutPresentProb, dropoutType)
% createDropoutMasks - Creates masks for dropout

% INPUTS: 
% layersSize - 1 x N vector, where N is the number of layers. Contains the
% size of each layer.
% batchSize: scalar, size of mini-batch
% dropoutPresentProb: probability that any neuron is present
% dropoutType: type of dropout, 1 = Bernoulli

% OUTPUTS:
% dropoutMasks - 1 x N cell array where N is the number of layers. Each
% cell contains the dropout mask for the corresponding layer (matrix with
% size [size of layer i, size of layer i+1]

noLayers = length(layersSize);

for i = 1:noLayers
    
    maskSize = [batchSize layersSize(i)];
    
    if dropoutType == 1 % Bernoulli
      
        dropoutMasks{i} = rand(maskSize) > (1 - dropoutPresentProb);
        
    end
    
    
    
end