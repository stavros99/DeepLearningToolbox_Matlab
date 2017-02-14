function w = rescaleWeights4Dropout(w, p)
% rescaleWeights4Dropout - Rescales weights for dropout. During testing
% weights should be multiplied with the dropout present probability

% INPUTS:
% w: 1 x N cell array, where N is the number of layers. Each cell contains
% a matrix with the weights of the corresponding layer.
% p: 1 x N, where N is the number of layers. Each entry corresponds to the
% present probability used during training with dropoout.

% OUTPUTS:
%  w: same as above. Weights have been adjusted.

for i =1:length(p)
   
    rescaleConstant = p(i);
    w{i} =  rescaleConstant * w{i};

end


