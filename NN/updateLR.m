function lr = updateLR(lrParams, currentEpoch)
% updateLR - Updates learning rate

% INPUTS:

% lrParams: structure containing the learning rate parameters, check manual
% for details
% currentEpoch: scalar, epoch number

% OUTPUTS:

% lr: scalar, new learning rate

if lrParams.schedulingType == 1 % lr remains constant until lrEpochThres and then decreases according to the formula below
    
      
    lr = lrParams.initialLR * lrParams.lrEpochThres / max(currentEpoch, lrParams.lrEpochThres ); %from 'Practical Recommendations 
    %for Gradient-Based Training of Deep Architectures' by Y. Bengio
    
elseif lrParams.schedulingType == 2 % lr is multiplied by the scaling factor after each epoch
       
      
    if  currentEpoch <= lrParams.lrEpochThres
    
        lr = lrParams.initialLR;
        
    else
        currentLR = lrParams.lr;   
        lr = currentLR * lrParams.scalingFactor; % from 'Improving neural networks by preventing co-adaptation of feature detectors' by Hinton et al. (arxiv)
    end
    
    
        
elseif lrParams.schedulingType == 3 % lr constantly decreases according to the formula below
    
       lr = lrParams.initialLR / (1 + (currentEpoch - 1) / (lrParams.lrEpochThres - 1)); % from 'Dropout:  A Simple Way to Prevent Neural Networks from
%Overfitting' by Srivastava et al.

    
    
    
end