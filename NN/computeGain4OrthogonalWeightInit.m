function g = computeGain4OrthogonalWeightInit(activ_function)
% computeGain4OrthogonalWeightInit - compute gain parameter for orthogonal weight
% initialisation

% INPUTS
% activ_function: name of activation function

% OUTPUTS
% g: scalar, gain parameter

%as suggested by Andrew Saxe on https://plus.google.com/+SoumithChintala/posts/RZfdrRQWL6u
switch activ_function
   
    case 'ReLu'
        
        g = sqrt(2); % lasagne (http://lasagne.readthedocs.org/en/latest/modules/init.html#lasagne.init.Glorot)
        % uses this gain for ReLu
        
    case 'tanh'
        
        g = 2; % suggested a value > 1 by Saxe
        
    otherwise
        
        g = 1; 
end