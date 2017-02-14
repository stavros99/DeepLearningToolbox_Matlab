function g = computeGain4GlorotWeightInit(activ_function)
% computeGain4GlorotWeightInit - compute gain parameter for glorot weight
% initialisation

% INPUTS
% activ_function: name of activation function

% OUTPUTS
% g: scalar, gain parameter

switch activ_function
   
    case 'ReLu'
        
        g = sqrt(2); % lasagne (http://lasagne.readthedocs.org/en/latest/modules/init.html#lasagne.init.Glorot)
        % uses this gain for ReLu
        
    case 'sigm'
        
        g = 4; % this is what suggested at the deep learning tutorials
        %http://deeplearning.net/tutorial/deeplearning.pdf, p. 37
        
    otherwise
        
        g = 1; %as suggested at the at the deep learning tutorials
        %http://deeplearning.net/tutorial/deeplearning.pdf, p. 37
        % we have no info for other functions so we use gain of 1, i.e., we
        % don't modify the standard formula. 
end