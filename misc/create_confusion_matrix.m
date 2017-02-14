function [matrix] = create_confusion_matrix(outputs,targets,nClasses)
% create_confusion_matrix - create confusion matrix

% INPUTS
% outputs: noExamples x 1, contains the predicted targets
% targets: noExampples x 1, actual targets
% nClasses: scalar, number of classes

% OUTPUTS
% matrix: confusion matrix, noClasses x noClasses


[samples,dim] = size(targets);
matrix = zeros(nClasses,nClasses);

for i = 1:samples
%     row = targets(i);
%     col = outputs(i);
%     matrix(row,col) = matrix(row,col) + 1;
% The above is clearer but the line below is quicker
      matrix(targets(i),outputs(i)) = matrix(targets(i),outputs(i)) + 1;
end

