function [e, L] = computeLoss(targetOut, netOut, activFunction)
% computeLoss - Computes loss of NN

% INPUTS:
% targetOut: contains the desired targets, noExamples x
% targetDimensionality
% netOut: contains the output of the network, noExamples x
% targetDimensionality
% activFunction: activation function of output layer

% OUTPUTS:
% e: error Matrix, error per example and per output, noExamples x targetDimensionality
% L: scalar, average loss of the network over all examples and outputs
     
 N = size(targetOut,1);

 e = netOut - targetOut;
 
switch activFunction
     
    case 'softmax'   
        
        L = -sum(sum(targetOut .* log(netOut))) / N; % negative log-likelihood loss function
        
    otherwise
        
        L = 0.5 * sum(sum(e .^ 2)) / N; %MSE loss function
end






