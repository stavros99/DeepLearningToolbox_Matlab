function [W, biases] = initWeights(inputSize, weightInitParams, layers, activ_functions)
% initWeights - initialise NN weights

% INPUTS
% inputSize: scalar, size of input layer
% weightInitParams: structure containing parameters for weight
% initialisation, see manual for details
% layers: 1xN vector containing the size of each layer, where N is the number of hidden layers + output layer
% activ_functions: 1xN cell array containing the activation function for
% each layer

% OUTPUTS
% W: 1xN cell array containing the weights of each layer
% biases: 1xN cell array containing the biases of each layer

type = weightInitParams.type;            
sigma = weightInitParams.sigma;   
mean = weightInitParams.mean;   
upperLimit = weightInitParams.upperLimit;
lowerLimit = weightInitParams.lowerLimit;
biasType = weightInitParams.biasType;   
biasConstant = weightInitParams.biasConstant;
sparsity = weightInitParams.sparsity;

layerSize_prev = inputSize;
noLayers = length(layers);    

    for i = 1 : noLayers
        
        layerSize_current = layers(i);
        
        % weights 
        if type == 1 % normal distribution  
            
          W{i} = mean + sigma * randn(layerSize_prev, layerSize_current);
          disp(['Layer ',num2str(i),' (',num2str(layerSize_prev), ' ', num2str(layerSize_current) ,') initialised with weights drawn from a normal distribution with mean ',...
              num2str(mean), ' and st. dev. ', num2str(sigma)])
          
        elseif type == 2 % uniform distribution  
            
            W{i} =  lowerLimit + (upperLimit - lowerLimit) .* rand(layerSize_prev, layerSize_current);
            disp(['Layer ',num2str(i),' (',num2str(layerSize_prev), ' ', num2str(layerSize_current) ,') initialised with weights drawn from a uniform distribution with range [',...
              num2str(lowerLimit), ' ', num2str(upperLimit), ']'])
          
        elseif type == 3 % lecun uniform
            %from Efficient BackProp by LeCun et al.
            sigma = 1 / sqrt(layerSize_prev);
            lowerLimit = - sqrt(3) * sigma;
            upperLimit = sqrt(3) * sigma;
            W{i} =  lowerLimit + (upperLimit - lowerLimit) .* rand(layerSize_prev, layerSize_current);
            disp(['Layer ',num2str(i),' (',num2str(layerSize_prev), ' ', num2str(layerSize_current) ,') initialised with LeCuns method where weights are drawn from a uniform distribution with range [',...
              num2str(lowerLimit), ' ', num2str(upperLimit), ']'])
        
        elseif type == 4 % lecun normal
            %from Efficient BackProp by LeCun et al.
            sigma = 1 / sqrt(layerSize_prev);
            W{i} =  mean + sigma * randn(layerSize_prev, layerSize_current);
            disp(['Layer ',num2str(i),' (',num2str(layerSize_prev), ' ', num2str(layerSize_current) ,') initialised with LeCuns method where weights are drawn from a normal distribution with mean ',...
              num2str(mean), ' and st. dev. ', num2str(sigma)])
        
        elseif type == 5 % glorot uniform
            %from Understanding the difficulty of training deep feedforward
            %neural networks by Glorot and Bengio
            
            g = computeGain4GlorotWeightInit(activ_functions{i});
            
            sigma = g * sqrt(2 / (layerSize_prev + layerSize_current));
            lowerLimit = - sqrt(3) * sigma; 
            upperLimit = sqrt(3) * sigma;
            W{i} =  lowerLimit + (upperLimit - lowerLimit) .* rand(layerSize_prev, layerSize_current);
            disp(['Layer ',num2str(i),' (',num2str(layerSize_prev), ' ', num2str(layerSize_current) ,') initialised with Glorots method where weights are drawn from a uniform distribution with range [ ',...
              num2str(lowerLimit), ' ', num2str(upperLimit), ' ]'])
            
        elseif type == 6 % glorot normal
            %from Understanding the difficulty of training deep feedforward
            %neural networks by Glorot and Bengio
            
            g = computeGain4GlorotWeightInit(activ_functions{i});
            
            sigma = g * sqrt(2 / (layerSize_prev + layerSize_current));
            
            W{i} = mean + sigma * randn(layerSize_prev, layerSize_current);
            disp(['Layer ',num2str(i),' (',num2str(layerSize_prev), ' ', num2str(layerSize_current) ,') initialised with Glorots method where weights are drawn from a normal distribution with mean ',...
              num2str(mean), ' and st. dev. ', num2str(sigma)])
            
        elseif type == 7 % He uniform
            %from Delving Deep into Rectifiers: Surpassing Human-level
            %Performance on ImageNet Classification by He et al.
            sigma = sqrt(2 / layerSize_prev); 
            lowerLimit = - sqrt(3) * sigma;
            upperLimit = sqrt(3) * sigma;
            W{i} =  lowerLimit + (upperLimit - lowerLimit) .* rand(layerSize_prev, layerSize_current);
            disp(['Layer ',num2str(i),' (',num2str(layerSize_prev), ' ', num2str(layerSize_current) ,') initialised with Hes method where weights are drawn from a uniform distribution with range [ ',...
              num2str(lowerLimit), ' ', num2str(upperLimit), ' ]'])
            disp('This method was specifically designed for ReLu activation functions')
            
        elseif type == 8 % He normal
            %from Delving Deep into Rectifiers: Surpassing Human-level
            %Performance on ImageNet Classification by He et al.
            sigma = sqrt(2 / layerSize_prev); 
            
            W{i} =  mean + sigma * randn(layerSize_prev, layerSize_current);
            disp(['Layer ',num2str(i),' (',num2str(layerSize_prev), ' ', num2str(layerSize_current) ,') initialised with Hes method where weights are drawn from a normal distribution with mean ',...
              num2str(mean), ' and st. dev. ', num2str(sigma)])
            disp('This method was specifically designed for ReLu activation functions')
            
        elseif type == 9 % sparse
            %from Deep Learning via Hessian-free optimisation by Martens
            noConnections2Keep = round((1 - sparsity) * layerSize_prev);
            tempW = zeros(layerSize_prev, layerSize_current);
            for neuronID = 1:layerSize_current
               r = randperm(layerSize_prev,noConnections2Keep); 
               tempW(r,neuronID) = mean + sigma * randn(noConnections2Keep,1);
            end
            
            W{i} = tempW;
          
          disp(['Layer ',num2str(i),' (',num2str(layerSize_prev), ' ', num2str(layerSize_current) ,')  initialised with sparse initialisation where ',num2str(noConnections2Keep) ,' incoming weights are kept for each neuron '])  
          
        elseif type == 10 % orthogonal
%             from Exact solutions to the nonlinear dynamics of learning in
%             deep linear neural networks by Saxe et al.

            g = computeGain4OrthogonalWeightInit(activ_functions{i});
            % note that ortho_init returns a cell array
            W(i) = ortho_init([layerSize_prev, layerSize_current],g);
            
            disp(['Layer ',num2str(i),' (',num2str(layerSize_prev), ' ', num2str(layerSize_current) ,')  initialised with orthogonal initialisation' ])  
          
        end
      
        if biasType == 1
            sigma = weightInitParams.sigma; % mean and sigma may change during weight initialisation so we read the values again
            mean = weightInitParams.mean;   
            
            biases{i} =  mean + sigma * randn(1, layerSize_current);
            disp(['Layer ',num2str(i),' (', num2str(layerSize_current) ,') biases initialised from a normal distribution with mean ',...
              num2str(mean), ' and st. dev. ', num2str(sigma)])
          disp(' ' )
           
        elseif biasType == 2
            disp(['Layer ',num2str(i),' (', num2str(layerSize_current) ,') biases initialised to a constant value ',...
              num2str(biasConstant)])
            biases{i} = biasConstant * ones(1, layerSize_current); 
            disp(' ')
        end


        
        layerSize_prev = layerSize_current;
    end
    

    
