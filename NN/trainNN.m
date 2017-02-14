function [nn, L_perBatch, L, L_val, clsfError_train, clsfError_val ]  = trainNN(nn, train_x, train_y, val_x, val_y)
% trainNN - trains a neural network

% INPUTS
% nn: neural network structure, see manual for details
% train_x: matrix containing training input examples (noExamples x
% InputDimensionality)
% train_y: matrix containing training targets (noExamples x
% TargetDimensionality). Each target  (row) is represented by a vector
% where all entries are 0 except for the one that corresponds to the target
% which is one, e.g. for class 4 [0 0 0 1 0 0 0 0 0 0]
% val_x: same as train_x but contains the validation training examples
% val_y: same as val_y but contains the validation targets


% OUTPUTS
% nn: neural network structure, see manual for details
% L_perBatch: noEpochs x batchSize, matrix where each row contains the loss
% of all the mini-batches in that epoch. Each row corresponds to one epoch.
% L: 1xNoEpochs, vector which contains the training loss per epoch
% L_val: 1xNoEpochs, vector which contains the validation loss per epoch
% clsfError_train: 1xNoEpochs, vector which contains the classification error on the training set per epoch
% clsfError_val: 1xNoEpochs, vector which contains the classification error on the validation set per epoch

validation = 0;

if ~isempty(val_x) && ~isempty(val_y)
    validation = 1;
    minError = Inf;
    max_fail_counter = 0;
end

[m, inputDim] = size(train_x);

batchsize = nn.batchsize;
numepochs = nn.epochs;

numbatches = ceil(m / batchsize);

% loss per epoch on the training set
L = zeros(1, numepochs); 
% loss per epoch on the validation set
L_val = zeros(1, numepochs); 
% loss per epoch, batch on the training set
L_perBatch = zeros(numepochs, numbatches);

clsfError_train = zeros(1, numepochs);
clsfError_val = zeros(1, numepochs);


if nn.showPlot == 1;
    fhandle = figure();
    h = axes;
    set(h, 'Xlim',[0,numepochs])
    xlabel('Number of epochs');
    ylabel('Loss');
    hold on
else
    fhandle = [];
end

%initialise matrices for previous updates - used in SGD with momentum
[histData.vW, histData.vBiases] = initialisePreviousUpdateWeights(nn.W, nn.biases); 
%initialise matrices for sum of past gradients - used in adam
[histData.sum_gradW, histData.sum_gradBiases] = initialisePreviousUpdateWeights(nn.W, nn.biases);
%initialise matrices for sum of squares of past gradients - used in
%adagrad, adadelta, adam and rmsprop
[histData.sum_gradW_sq, histData.sum_gradBiases_sq] = initialisePreviousUpdateWeights(nn.W, nn.biases);
%initialise matrices for sum of squares of past updates - used in adadelta
[histData.sum_vW_sq_, histData.sum_vBiases_sq] = initialisePreviousUpdateWeights(nn.W, nn.biases);


for i = 1 : numepochs
    tic;
    
    % Update learning rate
    lr = updateLR(nn.trParams.lrParams, i);

    % Update momentum
    if nn.trainingMethod == 1 || nn.trainingMethod == 2 || nn.trainingMethod == 3
        momentum = updateMomentum(nn.trParams.momParams, i);
    end
    
    
    nn.trParams.lrParams.lr = lr;
    nn.trParams.momParams.momentum = momentum;

    randomorder = randperm(m);
    
    % create mini-batches
    for batch = 1 : numbatches
         
        if batch == numbatches
            
            batch_x = train_x(randomorder((batch - 1) * batchsize + 1 : end), :);
            batch_y = train_y(randomorder((batch - 1) * batchsize + 1 : end), :);
            
        else
            
            batch_x = train_x(randomorder((batch - 1) * batchsize + 1 : batch * batchsize), :);
            batch_y = train_y(randomorder((batch - 1) * batchsize + 1 : batch * batchsize), :);
        end
        
        tempBatchsize = size(batch_x, 1); % last batch may have different size than the usual batchsize
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            inputMask = rand(size(batch_x)) > nn.inputZeroMaskedFraction;
            batch_x = batch_x .* inputMask;
        end
        
       
        if nn.dropoutParams.dropoutType ~= 0
             %dropout for hidden layers
            dropoutMasks = createDropoutMasks(nn.layersSize(1:end-1), tempBatchsize, nn.dropoutParams.dropoutPresentProbHid, nn.dropoutParams.dropoutType);
            nn.dropoutParams.dropoutMasks = dropoutMasks;
            
            %dropout for input layer
            dropoutMaskInput = createDropoutMasks(inputDim, tempBatchsize, nn.dropoutParams.dropoutPresentProbVis, nn.dropoutParams.dropoutType);
            nn.dropoutParams.dropoutMaskInput = dropoutMaskInput;
            
             %apply dropout to input
             batch_x = batch_x .* dropoutMaskInput{1}; % same as inputZeroMaskedFraction above?
        end
        
        % compute NN output
        [output, layerActivations] = simulateNN(nn, batch_x);
     
        % compute loss
        [e, Lbatch] = computeLoss(batch_y, output, nn.activation_functions{end});
        
        if nn.trainingMethod == 3 % i.e. SGD with nesterov momentum
            % update weihgts so gradients can be computed in w(t) +
            % m*Dw(t-1) if nesterov momentum is used
            currentW = nn.W;
            currentBiases = nn.biases;
            [newW, newBiases] = updateWeights4Nesterov(currentW, currentBiases, nn.trParams.momParams.momentum, histData.vW, histData.vBiases); 
            nn.W = newW;
            nn.biases = newBiases;
                     
        end
        
        % compute gradients
        [dW,dBiases] = computeNNgrads(nn, batch_x, e, layerActivations);
               
        
        
        if nn.trainingMethod == 3 % i.e. SGD with nesterov momentum
            % put back original weights if nesterov momentum is used
            nn.W = currentW;
            nn.biases = currentBiases;
        end
        
        % compute norm of weights (if we wish to monitor training, i.e. diagnostics =1)
        if nn.diagnostics == 1 && mod(i, nn.showDiagnostics) == 0
            normPerLayerW = computeNormPerLayer(nn.W);
        end
        
        [nn, histData] = updateWeights(nn, dW, dBiases, histData);
        
        % compute norm of weight updates and ration of norm(UpdateW) /
        % norm(W)
        % (if we wish to monitor training, i.e. diagnostics =1) 
        if nn.diagnostics == 1 && mod(i, nn.showDiagnostics) == 0
            normPerLayerUpdateW = computeNormPerLayer(histData.vW);
            ratioUpdateWoverW(batch,:) = normPerLayerUpdateW ./ normPerLayerW;
        end
        
        L_perBatch(i,batch) = Lbatch;
        
    end
     
    
    % Test NN after each epoch on the entire trainings + validation sets
    %-------------------------
    originalW = nn.W;
    nn = prepareNet4Testing(nn);
        
    [statsTrain, output, e, L(i)] = evaluateNNperformance( nn, train_x, train_y);
    
    if nn.type == 2%i.e., classification. If regression then the last layer will be linear so the MSE will be the loss, i.e. L
        
        clsfError_train(i) = 100*(1 - statsTrain.clsfRate);
    end
    
    
    if validation == 1  
        
            [statsVal, output, e_val, L_val(i)] = evaluateNNperformance( nn, val_x, val_y);
    
            if nn.type == 2%i.e., classification
        
                clsfError_val(i) = 100*(1 - statsVal.clsfRate);
    
            end
            
        % check if loss is lower than the minimum loss (used when early
        % stopping is enabled)
        if L_val(i) < minError
           
            bestW = originalW;
            bestBiases = nn.biases;
            minError = L_val(i);
            max_fail_counter = 0;
        else
            max_fail_counter = max_fail_counter + 1;
        end
        
    end
    
    nn.testing = 0;
    nn.W = originalW;
    %-------------------------   
    
    t = toc;
    disp(' ')
    disp(['epoch ' num2str(i) '/' num2str(numepochs) '. Took ' num2str(t) ' seconds'])

    % display nn parameters during training
    dispParams(nn);
    
    disp(' ')
    disp(['Mini-batch mean error on training set is ' num2str(mean(L_perBatch(i,:)))]);
    disp(['Full-batch train err = ', num2str(L(i))])
    
    if validation == 1
        disp(['Full-batch val err = ',  num2str(L_val(i))]);
    end
    
    if nn.type == 2
        disp(['Classification Error on Training Set ',num2str(clsfError_train(i))])
        disp(['Classification Error on Validation Set ',num2str(clsfError_val(i))])
    end
    
    disp(' ')
    
    % display statistics about training, i.e., mean/st.dev. of neurons
    % activations per layer and ratios of norm(UpdateW) / norm(W)
    if nn.diagnostics == 1 && mod(i, nn.showDiagnostics) == 0
            
        [meanActiv, stdActiv] = computeActivationStats(layerActivations, nn.activation_functions);
        [meanRatios, stdRatios] = computeWeightRatios(ratioUpdateWoverW);
     end
    
    disp(' ')
    
    % plot training and validation loss 
    if nn.showPlot == 1;

        updatefigure(fhandle,L(1:i), L_val(1:i), validation, numepochs);
    
    end
    
    % when early stopping is used and validation error has increased more than the maximum allowed numner
    if nn.earlyStopping == 1 && max_fail_counter >= nn.max_fail 
        
        nn.biases = bestBiases;
        nn.W = bestW;
        disp(['Validation error has increased ',num2str(max_fail_counter),' times since the minimun validation error observed so training will stop'])
        break
        
    end
    

    
end
 


