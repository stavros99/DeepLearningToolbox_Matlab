
type = 2; % 1 is autoencoder (AE), 2 is classifier 

% load mnist_uint8;
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
    hiddenLayers = [1000 500 250 50]; 
   
elseif type == 2 % classifier
    outputSize = size(train_y,2); % in case of classification it should be equal to the number of classes

    hiddenActivationFunctions = {'ReLu','ReLu','ReLu'};%{'sigm','sigm','sigm'};%{'ReLu','ReLu','ReLu','ReLu'};%
    hiddenLayers = [500 500 1000 ]; % hidden layers sizes, does not include input or output layers

end


% parameters used for visualisation of first layer weights
visParams.noExamplesPerSubplot = 50; % number of images to show per row
visParams.noSubplots = floor(hiddenLayers(1) / visParams.noExamplesPerSubplot);
visParams.col = 28;% number columns of image
visParams.row = 28;% number rows of image



dbnParams = dbnParamsInit(type, hiddenActivationFunctions, hiddenLayers);
dbnParams.inputActivationFunction = 'sigm'; %sigm for binary inputs, linear for continuous input
dbnParams.rbmParams.epochs = 10;

% normalise data
train_x = normaliseData(dbnParams.inputActivationFunction, train_x,[]);
val_x = normaliseData(dbnParams.inputActivationFunction, val_x,[]);
test_x = normaliseData(dbnParams.inputActivationFunction, test_x,[]);

% train Deep Belief Network
[dbn, errorPerBatch errorPerSample] = trainDBN(train_x, dbnParams);

% visualise weights of first layer
figure()
visualiseHiddenLayerWeights(dbn.W{1},visParams.col,visParams.row,visParams.noSubplots);

nn = unfoldDBNtoNN(dbnParams, dbn, outputSize);

% Set some NN params
%-----
nn.epochs = 20;

% set initial learning rate
nn.trParams.lrParams.initialLR = 0.01; 
% set the threshold after which the learning rate will decrease (if type
% =1)
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

% show diagnostics to check if training proceeds 
nn.diagnostics = 1;
% show diagnostics every "showDiagnostics" epochs
nn.showDiagnostics = 5;

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




