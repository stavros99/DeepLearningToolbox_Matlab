function dispDropoutParams(dropoutParams)
% dispDropoutParams - displays on command line if dropout is used

% INPUTS
% dropoutParams: strucure which specifies the dropout parameters

if dropoutParams.dropoutType == 1
    
    disp(['Bernoulli Dropout '])
    disp(['Retain Probability for Hidden Neurons: ',num2str(dropoutParams.dropoutPresentProbHid)])
    disp(['Retain Probability for Inputs: ',num2str(dropoutParams.dropoutPresentProbVis)])
    
elseif dropoutParams.dropoutType == 2
    
     disp(['Gaussian Dropout '])
     
else
    
    disp('No Dropout used')
    
end



