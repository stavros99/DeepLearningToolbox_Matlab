function nn = paramsNNinit(newLayers, activation_functions)
% paramsNNinit - Initialise parameters for neural networks

% INPUTS
% newLayers: 1xN vector, where N is the number of layers (hidden + output),
% each entry contains the size of the corresponding layer

% activation_functions: 1xN cell array, where N is the number of layers (hidden + output),
% each cell contains the activation function of the corresponding layer

% OUTPUTS

% nn: neural network structure, see manual for details
   
nn.layersSize   = newLayers;
nn.noLayers      = length(newLayers);
 
nn.epochs = 1000;
nn.batchsize = 100;
nn.activation_functions              = activation_functions; %  Activation functions of hidden layers
    
nn.trParams.lrParams.lr             = 0.01; %  learning rate 
nn.trParams.lrParams.scalingFactor  = 0.999; %  Scaling factor for the learning rate (each epoch)
nn.trParams.lrParams.lrEpochThres   = 50;
nn.trParams.lrParams.schedulingType = 1; % 1 = lr*T / max(currentEpoch, T), 2 = scaling, 3 = lr / (1 + currentEpoch/T), where T = lrEpochThres
nn.trParams.lrParams.initialLR      = 0.01; 
    
nn.trParams.momParams.momentum                = 0.5;  %  Momentum
nn.trParams.momParams.initialMomentum         = 0.5; 
nn.trParams.momParams.finalMomentum           = 0.9; 
nn.trParams.momParams.momentumEpochLowerThres = 50; 
nn.trParams.momParams.momentumEpochUpperThres = 100; 
nn.trParams.momParams.schedulingType          = 1; %1 = constant (=initialMomentum) until momentumEpochLowerThres then linear increase until momentumEpochUpperThres and then constant again (=momentumEpochUpperThres)
nn.trParams.momParams.scaleLR                 = 0; %if true (1) then the update rule is m*DW(t-1) + (1-m)(-lr*dE/dW), if false (0) then update rule m*DW(t-1) + (-lr*dE/dW)

nn.trParams.adagrad.epsilon = 10^(-6);

nn.trParams.adadelta.epsilon = 10^(-6);
nn.trParams.adadelta.gamma = 0.95;

nn.trParams.rmsprop.epsilon = 10^(-6);
nn.trParams.rmsprop.gamma = 0.9;

nn.trParams.adam.b1 = 0.9;
nn.trParams.adam.b2 = 0.999;
nn.trParams.adam.epsilon = 10^(-8);

nn.weightInitParams.type             = 8;% check manual for details
nn.weightInitParams.sigma            = 0.1;% st. dev. of gaussian used to initialse weights, applies to type 1 only
nn.weightInitParams.mean             = 0; % mean of gaussian used to initialse weights, applies to type 1 only  
nn.weightInitParams.upperLimit       = 1; %lower limit of uniform distribution, applies to type 2 only
nn.weightInitParams.lowerLimit       = -1; %upper limit of uniform distribution, applies to type 2 only
nn.weightInitParams.sparsity         = 0.8; % percentage of incoming weights to be set to 0, applies to type 9 only
nn.weightInitParams.biasType         = 2; % 2 = constant, 1 = a gaussian is used with mean and stdev sigma defined by the parameters below
nn.weightInitParams.biasConstant     = 0;
    
    
nn.weightConstraints.weightPenaltyL2  = 0; % L2 regularization coefficient
nn.weightConstraints.weightPenaltyL1  = 0; % L1 regularisation coefficient
nn.weightConstraints.maxNormConstraint= 0; % maximum norm allowed, if 0 then it's disabled
    
nn.dropoutParams.dropoutType         = 0;% 0 = no dropout, 1 = bernoulli droput, 2 = gaussian dropout 
nn.dropoutParams.dropoutPresentProbVis            = 0.8;% present probability for a visible node, use 1 if no input layer droput is needed
nn.dropoutParams.dropoutPresentProbHid            = 0.5;% present probability for a hidden node
    
nn.inputZeroMaskedFraction          = 0;%Used for Denoising AutoEncoders, percentage of input units that will be set to 0

nn.earlyStopping                    = 0; %early stopping flag, 1 = yes, 0 = no
nn.max_fail                         = 10; %max number of increases in validation set, then training stops

nn.trainingMethod                   = 2; % 1 = SGD, 2 = SGD with momentum, 3 = SGD with nesterov momentum, 4 = adagrad, 5 = adadelta, 6 = rmsprop, 7 = adam
nn.pretraining                      = 0; % it's 1 if pretraining is used, otherwise 0

nn.testing                          = 0; % if it's 0 then network is in training mode, if 1 then netword in testing mode
nn.diagnostics                      = 0; %1 for yes, 0 for no
nn.showDiagnostics                  = 10;

nn.showPlot                         = 1; % show plot of training and validation loss
        

