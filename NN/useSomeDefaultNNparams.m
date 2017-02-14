function nn = useSomeDefaultNNparams(nn)
% useSomeDefaultNNparams - Use some default parameters if dropout is used
% as suggested by the papers below

% INPUTS
% nn: neural network structure, see manual for details

% OUTPUTS
% nn: neural network structure, see manual for details


%if pretraining and dropout then use no weight constraints and a small
%learning rate according the paper above (sections A1, A2)
if nn.pretraining == 1 && nn.dropoutParams.dropoutType ~= 0

        %according to the above paper no weight constraints should be used
        %and a small learning rate
        nn.trParams.lrParams.initialLR = 0.1; 
        nn.trParams.lrParams.scalingFactor  = 1; %i.e., constant lr
        nn.trParams.lrParams.schedulingType = 2;
        
        nn.trParams.momParams.initialMomentum = 0.5;
        nn.trParams.momParams.finalMomentum = 0.99;
        nn.trParams.momParams.momentumEpochLowerThres = 1;
        nn.trParams.momParams.momentumEpochUpperThres = 500;
        nn.trParams.momParams.schedulingType     = 1;
    
        %or
        %nn.trParams.sgd.momParams.momentum = 0.9;
        %nn.trParams.sgd.momParams.scalingFactor = 1; %i.e. constant
        %nn.trParams.sgd.momParams.schedulingType = 2;
        
        nn.weightConstraints.weightPenaltyL1 = 0;
        nn.weightConstraints.weightPenaltyL2 = 0;
        nn.weightConstraints.maxNormConstraint = 0;
        
        nn.epochs = 3000;
 
%if NO pretrainig and dropout then use preferably max-norm constraint and a
%high learning rate + momentum with scheduling
elseif nn.pretraining == 0 && nn.dropoutParams.dropoutType ~= 0
    

%     e.g.
    nn.trParams.lrParams.initialLR = 10;
    nn.trParams.lrParams.scalingFactor = 0.998;
    nn.trParams.lrParams.schedulingType = 2;
    
    nn.trParams.momParams.initialMomentum = 0.5;
    nn.trParams.momParams.finalMomentum = 0.99;
    nn.trParams.momParams.momentumEpochLowerThres = 1;
    nn.trParams.momParams.momentumEpochUpperThres = 500;
    nn.trParams.momParams.schedulingType     = 1;
    nn.epochs = 3000;
    
%     or 
%     from “Dropout: A simple way to prevent neural networks from overfitting” by Srivastava at al. JMLR 2014 
%     nn.trParams.sgd.momParams.momentum = 0.95;
%     nn.trParams.sgd.momParams.scalingFactor = 1; %i.e. constant
%     nn.trParams.sgd.momParams.schedulingType = 2;
    
    
    nn.weightConstraints.weightPenaltyL1 = 0;
    nn.weightConstraints.weightPenaltyL2 = 0;
    nn.weightConstraints.maxNormConstraint = 3;
end


