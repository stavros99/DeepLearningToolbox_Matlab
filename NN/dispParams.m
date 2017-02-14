function dispParams(nn)



disp(['Training NN with ',num2str(nn.noLayers), ' layers'])
 
disp(['Topology = ',num2str(nn.layersSize)])

disp(['Activation Functions: ']), disp(nn.activation_functions)
 
disp(['Batchsize = ', num2str(nn.batchsize)])
 
dispTrainingMethod(nn.trainingMethod)
    
  
if nn.trainingMethod ~= 5 % i.e. if it's other than AdaDelta which doesn't need a learning rate
 
    disp(['Global learning rate = ',num2str(nn.trParams.lrParams.lr)]);
 
end


if nn.trainingMethod == 1 || nn.trainingMethod == 2 || nn.trainingMethod == 3
        
  
    disp(['momentum = ', num2str(nn.trParams.momParams.momentum)]); 
  
end


disp(' ')
  
dispWeightConstraints(nn.weightConstraints)
  
disp(' ')
   
dispDropoutParams(nn.dropoutParams)