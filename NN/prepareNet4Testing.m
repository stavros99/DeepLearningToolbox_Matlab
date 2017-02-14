function nn = prepareNet4Testing(nn)
% prepareNet4Testing - sets the network in test mode and rescales the weights in case dropout is
% used

% INPUTS 
% nn: nn structure, check manual

% OUTPUTS
% nn: nn structure, check manual

nn.testing = 1;

    %if bernoulli dropout then rescale the weights for testing
    if nn.dropoutParams.dropoutType == 1
        
        p =  [nn.dropoutParams.dropoutPresentProbVis ones(1,nn.noLayers - 1) * nn.dropoutParams.dropoutPresentProbHid];
        nn.W = rescaleWeights4Dropout(nn.W, p);
        
    end