function states = computeStates(layerType, probs, data)
% computeStates - Computes states of hidden/visible layer of an RBM

% INPUTS
% layerType: activation function of given layer, e.g. 'sigm', 'linear',
% 'ReLu'

% probs: activation matrix, noExamples x noNeurons

% data: data matrix, it's the input to the neurons, noExamples x noNeurons

% OUTPUTS
% states: states matrix, noExamples x noNeurons


[numExamples,numHid] = size(probs);

if strcmpi(layerType,'sigm')
  
      states = probs > rand(numExamples,numHid);
  
  elseif strcmpi(layerType,'linear')
      
%     Compute states for linear layers similarly to Hinton's code http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
      states = probs + randn(numExamples,numHid); 
      
  elseif strcmpi(layerType,'ReLu')
      
%     Compute states for ReLu layers similarly to Nair&Hinton, Rectified
%     Linear Units Improve Restricted Boltzmann Machines, ICML 2010

      sigma = 1./(1 + exp(-data));
      noise = sigma .* randn(numExamples, numHid);
      states =  max(0,data + noise); 
      
 
      
end


