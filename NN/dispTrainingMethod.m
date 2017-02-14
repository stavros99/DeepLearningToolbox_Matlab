function dispTrainingMethod(methodNo)
%dispTrainingMethod - Displays the training method used

%INPUTS
% methodNo: scalar which indicates which training method is used
        
switch methodNo        

    case 1 % SGD
        disp('Training method: SGD')
        

    case 2 % SGD with momentum
        disp('Training method: SGD with momentum')
        

    case 3 % SGD with nesterov momentum
       disp('Training method: SGD with Nesterov momentum')
      
        
    case 4 % Adagrad
        disp('Training method: Adagrad')
        
    case 5 % Adadelta
         disp('Training method: Adadelta')
         disp('Learning rate is not needed for Adadelta')
        
    case 6 % RMSprop
        disp('Training method: RMSprop')
       
    case 7 % Adam
        disp('Training method: Adam')
        
end
        