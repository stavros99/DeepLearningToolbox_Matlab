function f = calculateFmeasure(recall ,precision, a)


f = (1 + a)*(precision.*recall) ./ ((a*precision + recall) + eps);
