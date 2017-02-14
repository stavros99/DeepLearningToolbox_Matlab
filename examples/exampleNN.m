
type = 2; % 1 is autoencoder (AE), 2 is classifier

% load MNIST data
load('mnist_uint8.mat')

% validation inputs
val_x = double(train_x(50001:60000,:));
% validation targets
val_y = double(train_y(50001:60000,:));

%if no validation exists then
% val_x = [];
% val_y = [];

% training inputs
train_x = double(train_x(1:50000,:));
% training targets
train_y = double(train_y(1:50000,:));

% test inputs
test_x  = double(test_x);
% test targets
test_y  = double(test_y);

inputSize = size(train_x,2);

if type == 1 % AE
   outputSize  = inputSize; % in case of AE it should be equal to the number of inputs

   %if type = 1, i.e., AE then the last layer should be linear and usually a
   % series of decreasing layers are used
    hiddenActivationFunctions = {'sigm','sigm','sigm','linear'}; 
    hiddenLayers = [1000 500 250 50 250 500 1000 outputSize]; 
    
elseif type == 2 % classifier
    outputSize = size(train_y,2); % in case of classification it should be equal to the number of classes

    hiddenActivationFunctions = {'ReLu','ReLu','ReLu','softmax'};
    hiddenLayers = [500 500 1000 outputSize]; 
    
end


% parameters used for visualisation of first layer weights
visParams.noExamplesPerSubplot = 50; % number of images to show per row
visParams.noSubplots = floor(hiddenLayers(1) / visParams.noExamplesPerSubplot);
visParams.col = 28;% number of image columns
visParams.row = 28;% number of image rows 

inputActivationFunction = 'sigm'; %sigm for binary inputs, linear for continuous input

% normalise data
% we assume that data are images so each image is z-normalised. If other
% types of data are used then each feature should be z-normalised on the
% training set and then mean and standard deviation should be applied to
% validation and test sets.
train_x = normaliseData(inputActivationFunction, train_x, []);
val_x = normaliseData(inputActivationFunction, val_x, []);
test_x = normaliseData(inputActivationFunction, test_x, []);

%initialise NN params
nn = paramsNNinit(hiddenLayers, hiddenActivationFunctions);

% Set some NN params
%-----
nn.epochs = 20;

% set initial learning rate
nn.trParams.lrParams.initialLR = 0.01; 
% set the threshold after which the learning rate will decrease (if type
% = 1 or 2)
nn.trParams.lrParams.lrEpochThres = 10;
% set the learning rate update policy (check manual)
% 1 = initialLR*lrEpochThres / max(lrEpochThres, T), 2 = scaling, 3 = lr / (1 + currentEpoch/lrEpochThres)
nn.trParams.lrParams.schedulingType = 1;

nn.trParams.momParams.schedulingType = 1;
%set the epoch where the learning will begin to increase
nn.trParams.momParams.momentumEpochLowerThres = 10;
%set the epoch where the learning will reach its final value (usually 0.9)
nn.trParams.momParams.momentumEpochUpperThres = 15;

% set weight constraints
nn.weightConstraints.weightPenaltyL1 = 0;
nn.weightConstraints.weightPenaltyL2 = 0;
nn.weightConstraints.maxNormConstraint = 4;

% show diagnostics to monnitor training  
nn.diagnostics = 1;
% show diagnostics every "showDiagnostics" epochs
nn.showDiagnostics = 5;

% show training and validation loss plot
nn.showPlot = 1;

% use bernoulli dropout
nn.dropoutParams.dropoutType = 0;

% if 1 then early stopping is used
nn.earlyStopping = 0;
nn.max_fail = 10;

nn.type = type;

% set the type of weight initialisation (check manual for details)
nn.weightInitParams.type = 8;

% set training method
% 1: SGD, 2: SGD with momentum, 3: SGD with nesterov momentum, 4: Adagrad, 5: Adadelta,
% 6: RMSprop, 7: Adam
nn.trainingMethod = 2;
%-----------

% initialise weights
[W, biases] = initWeights(inputSize, nn.weightInitParams, hiddenLayers, hiddenActivationFunctions);

nn.W = W;
nn.biases = biases;

% if dropout is used then use max-norm constraint and a
%high learning rate + momentum with scheduling
% see the function below for suggested values
% nn = useSomeDefaultNNparams(nn);

if type == 1 % AE
    [nn, Lbatch, L_train, L_val]  = trainNN(nn, train_x, train_x, val_x, val_x);
elseif type == 2 % classifier
    [nn, Lbatch, L_train, L_val, clsfError_train, clsfError_val]  = trainNN(nn, train_x, train_y, val_x, val_y);
 end

nn = prepareNet4Testing(nn);

% visualise weights of first layer
figure()
visualiseHiddenLayerWeights(nn.W{1},visParams.col,visParams.row,visParams.noSubplots);


if type == 1 % AE
    [stats, output, e, L] = evaluateNNperformance( nn, test_x, test_x);
elseif type == 2 % classifier
    [stats, output, e, L] = evaluateNNperformance( nn, test_x, test_y);
 end




