function stats = computeStatsFromConfMat(confMat) 
% computeStatsFromConfMat - compute performance measures from confusion
% matrix

%INPUTS 
% confMat: confusion matrix, noClasses x noClasses

%OUTPUTS
% stats: structure which contains performance measures

[rec,prec] = confMat2recalprec(confMat);
clsfRate = confMat2ClasfRate(confMat);
f1 = calculateFmeasure(rec ,prec, 1);
UAR = mean(rec);

stats.recall = rec;
stats.prec = prec;
stats.clsfRate = clsfRate;
stats.f1 = f1;
stats.UAR = UAR;