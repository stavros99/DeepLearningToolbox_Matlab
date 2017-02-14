function d_act = computeActivFcnDeriv(a, activationFunction)
% computeActivFcnDeriv - Compute derivatives of activation functions

%INPUTS
% a: batchSize x noNeurons, contains the activations of the neurons
% activationFunction: string which contains the activation function which
% is used

%OUTPUTS:
% d_act: batchSize x noNeurons, contains the derivatives of the activations
% Derivative of the activation function
           
switch activationFunction 
          
    case 'sigm'
    
        d_act = a .* (1- a);
        
    case 'tanh'
    
        d_act = 1 - a.^2;
        
    case 'linear'
        
        d_act = 1;
        
    case 'ReLu'
        
        
        d_act = zeros(size(a));
        d_act(a > 0) = 1;
        
    case 'leakyReLu'
        
        d_act = 0.01*ones(size(a));
        d_act(a > 0) = 1;
        
  
        
      
end
        
