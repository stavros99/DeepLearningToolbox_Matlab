function clasfRate = confMat2ClasfRate(confMat)

% Divide the trace  by the total sum of the matrix
clasfRate = sum(diag(confMat)) / sum(confMat(:));