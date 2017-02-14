function regW = applyMaxNormRegularisation(w, c)
% applyMaxNormRegularisation - Applies max norm regularasation

% INPUTS: 
% w: weight matrix, (size of layer i) x (size of layer + 1)
% c: scalar, maximum allowed norm for weight vector connected to a neuron
% in layer i + 1

% OUTPUTS:
% regW: weight matrix after applying max norm constraint, (size of layer i) x (size of layer + 1)

regW = w;

w2 = w .^ 2;

normW = sqrt(sum(w2));

ind = find(normW > c);

for i = 1:length(ind)
    colInd = ind(i);
    
    scaledW = regW(:,colInd) * (c / normW(colInd) ); 
    regW(:,colInd) = scaledW; 
    
end
