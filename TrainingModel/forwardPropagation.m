function [Z, A] = forwardPropagation(dataX, theta, transferFunctions, options)
% This program computes the response of an artificial neural network for
% multiclass classification
%
% Inputs:
% dataX                   [N n] matrix, where each row contains an observation
%                         X = (x_1, x_2,...,x_n)
%
% theta                   Cell containing the weights of each layer of the
%                         network, where theta{1} contains the weights of
%                         the connections between the input and the first
%                         hidden layer, theta{2} contains the weights of
%                         the connections between the first and the second
%                         hidden layer, and so on
%
% transferFunctions       Cell containg the name of the transfer functions
%                         of each layer of the neural network. Options of transfer
%                         functions are:
%                         - none: input layer has no transfer functions
%                         - tanh: hyperbolic tangent
%                         - elu: exponential linear unit
%                         - softplus: log(exp(x) + 1)
%                         - relu: rectified linear unit
%                         - logsig: logistic function
%                         - softmax: transfer function of the last layer
%
% options                structure containing additional settings for the
%                        neural network (e.g., rectified linear unit
%                        threshold)
%
% Outputs:
% Z                     Cell containing the multiplicaction between the inputs
%                       and weights of each layer (this is used for the
%                       back-propagation algorithm)
%
% A                     Cell containing the response of the activation functions
%                       of each layer
%
% Marco E. Benalcázar
% Escuela Politecnica Nacional
% marco.benalcazar@epn.edu.ec
% (C) Copyright Marco E. Benalcázar
%

numLayers = length(theta) + 1;
% Threshold for the RELU transfer function
if ~isfield(options, 'reluThresh')
    reluThresh = 1e-4;
else
    reluThresh = options.reluThresh;
end

N = size(dataX, 1);
vectorOnes = ones(N,1);
A = cell(1, numLayers);
Z = cell(1, numLayers);
Z{1} = dataX;
A{1} = [vectorOnes Z{1}];
% Forward propagation through all the layers, except the last one
for i = 2:numLayers
    Z{i} = A{i - 1}*theta{i - 1}';
    % Transfer function values
    switch transferFunctions{i}
        case 'logsig'
            A{i} = [vectorOnes logsig(Z{i})];
        case 'relu'
            A{i} = [vectorOnes bsxfun(@max, Z{i}, reluThresh)];
        case 'tanh'
            A{i} = [vectorOnes tansig(Z{i})];
        case 'softmax'
            Zaux = bsxfun( @minus, Z{i}, max(Z{i}, [], 2) );
            Zaux = exp(Zaux);
            A{i} = bsxfun( @rdivide, Zaux, sum(Zaux, 2) );
        case 'softplus'
            A{i} = [vectorOnes log(exp(Z{i}) + 1)];
        case 'elu'
            A{i} = [vectorOnes elu(Z{i})];            
        otherwise
            error('Invalid transfer function. Valid options are elu, softplus, relu, logsig, tanh, and softmax');            
    end
end
return