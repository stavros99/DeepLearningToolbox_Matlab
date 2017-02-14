function normPerLayerW = computeNormPerLayer(w)
% computeNormPerLayer - Computers the norm of all the weights in each layer

%INPUTS:
% w: 1xNoLayers, cell array which contains the weights per layer

%OUTPUTS
% normPerLayerW: 1xNoLayers vector, contains the norm of all the weights in
% each layer

noLayers = length(w);

for i = 1:noLayers
   
    normPerLayerW(i) = norm(w{i}(:));
  
end 