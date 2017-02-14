function [recal,prec] = confMat2recalprec(confMat)

% convert a confusion matrix to recal and precision rates


[rows,col] = size(confMat);

for i = 1:rows
    recal(i) = confMat(i,i)/(sum(confMat(i,:)));%+eps);
    prec(i) = confMat(i,i)/(sum(confMat(:,i)));%+eps);
end

