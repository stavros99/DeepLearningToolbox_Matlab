function [data,PS] = normaliseData(trFcn, data, PS)
% normaliseData - Normalise data

% INPUTS
% trFcn: type of input layer, 'linear' (for continuous data) or 'sigm' for 
% binary data

% data: input data, noExamples x noFeatures

% PS: structure which contains mean and standard deviation values for each
% feature. If it's empty then the data are normalised to 0 mean and st.
% dev. 1. If not, then the mean from PS is subtracted from each feature
% and then we divide by the st. dev. from PS

% OUTPUTS
% data: normalised data
% PS: normalisation structure, see above


% in case of linear visible layer (i.e. data are continuous) it is recommended 
% by Hinton in "A practical guide to training RBMs" to make each dimension 
% of the feature vector to have zero mean and unit standard deviation.
if strcmpi(trFcn, 'linear')
    
    if isempty(PS)
        ymean = 0;
        ystd = 1;
        [data,PS] = mapstd(data,ymean,ystd);
    else
        data = mapstd('apply',data,PS);
        
    end

% in case the activation function of the visible layer is "sigm" i.e. data
% are binary, then simply divide by the max value so the data are in the
% range [0, 1].
elseif strcmpi(trFcn, 'sigm')
    data = data/max(data(:));
end