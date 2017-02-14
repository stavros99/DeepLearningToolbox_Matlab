function W = ortho_init(sz,g)
% sz: list of layer sizes (any sizes allowed)
% g: scale factor to apply to each layer
%
% g should be set based on the activation function:
% linear activations     g = 1 (or greater)
% tanh activations       g > 1
% ReLU activations       g = sqrt(2) (or greater)
%
% Author: Andrew Saxe, 2014
 
Nl = length(sz);
for i = 1:Nl
    [R{i},tmp] = qr(randn(sz(i),sz(i)));
end
    
for i = 1:Nl-1
    dim = min(size(R{i+1},2),size(R{i},1));
%     W{i} = g*R{i+1}(:,1:dim)*R{i}(:,1:dim)'; oriinal
    W{i} = [g*R{i+1}(:,1:dim)*R{i}(:,1:dim)']';
end