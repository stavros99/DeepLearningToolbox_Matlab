function momentum = updateMomentum(momParams, currentEpoch)
% updateMomentum - Updates momentum

% INPUTS:

% momParams: structure containing the momentum parameters, check manual
% for details
% currentEpoch: scalar, epoch number

% OUTPUTS:

% momentum: scalar, new momentum
   
if momParams.schedulingType == 1 % first constant then linear increase
    
    deltaEpochThres = momParams.momentumEpochUpperThres - momParams.momentumEpochLowerThres;
    
    if currentEpoch < momParams.momentumEpochLowerThres
        
        momentum = momParams.initialMomentum;
        
    elseif currentEpoch >= momParams.momentumEpochLowerThres && currentEpoch <= momParams.momentumEpochUpperThres
    % linear increase between initial and final values
        momentum = ( (currentEpoch - momParams.momentumEpochLowerThres)  / deltaEpochThres) * momParams.finalMomentum + ...
            ( 1 - (currentEpoch - momParams.momentumEpochLowerThres) / deltaEpochThres) * momParams.initialMomentum;
        
    elseif currentEpoch > momParams.momentumEpochUpperThres
        
        momentum = momParams.finalMomentum;
        
    end
        
end