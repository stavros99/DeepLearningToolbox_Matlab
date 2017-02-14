function dispWeightConstraints(weightConstraints)
% dispWeightConstraints - displays on command line which weight contrainsts
% are used

% INPUTS
% weightConstrains: strucure which specifies which weight constraints to
% use

flag = 0;

if weightConstraints.weightPenaltyL2 > 0
    
    disp(['L2 regularisation is applied with lamda = ', num2str(weightConstraints.weightPenaltyL2)])
    flag = 1;
end

if weightConstraints.weightPenaltyL1 > 0
    
    disp(['L1 regularisation is applied with lamda = ', num2str(weightConstraints.weightPenaltyL1)])
    flag = 1;
end

if weightConstraints.maxNormConstraint > 0
    
    disp(['Max norm constraint is applied with max norm = ', num2str(weightConstraints.maxNormConstraint)])
    flag = 1;
end

if flag == 0
   disp('No Weight Constraints Used') 
end
