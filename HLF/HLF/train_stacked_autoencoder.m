function [stack] = train_stacked_autoencoder(data, netconfig, sparsityParam, beta, maxIter, lambda)
% This code is for performing unsupervised learning of features with 
% greedy layerwise training


if nargin < 3
    sparsityParam = 0.2; % desired average activation of the hidden units.
    beta = 3; % weight of sparsity penalty term 
    maxIter = 400; % maximum number of iterations
    lambda = 3e-3; % weight decay parameter 
    
elseif nargin < 4
    beta = 3; 
    lambda = 3e-3; 
    maxIter = 400;
    
elseif nargin < 5
    maxIter = 400;


    
    lambda = 3e-3; 
    
elseif nargin < 6
    lambda = 3e-3; 
    
end

depth = numel(netconfig.layersizes);
stack = cell(depth,1);
Theta = cell(depth,1);


if depth >= 1
%%======================================================================
%% STEP 1: Train First layer of sparse autoencoders

    Theta_init = initializeParameters(netconfig.layersizes{1}, netconfig.inputsize);

    addpath minFunc/
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
    options.maxIter = maxIter;% Maximum number of iterations of L-BFGS to run 
    %options.display = 'on';

    [Theta{1}, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
    netconfig.inputsize, netconfig.layersizes{1}, ...
    lambda, sparsityParam, ...
    beta, data), ...
    Theta_init, options);

    stack{1}.w = reshape(Theta{1}(1:netconfig.layersizes{1}*netconfig.inputsize), ...
    netconfig.layersizes{1}, netconfig.inputsize);

    stack{1}.b = Theta{1}(2*netconfig.layersizes{1}*netconfig.inputsize+1:2*netconfig.layersizes{1}*netconfig.inputsize+netconfig.layersizes{1});

end

%%======================================================================
%% STEP 2: Train further layers of sparse autoencoders

if depth >= 2

    [higher_features] = feedForwardAutoencoder(Theta{1}, netconfig.layersizes{1}, ...
    netconfig.inputsize, data);


    for i = 2:depth
        % Randomly initialize the parameters
        Theta_init = initializeParameters(netconfig.layersizes{i}, netconfig.layersizes{i-1});

        [Theta{i}, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
        netconfig.layersizes{i-1}, netconfig.layersizes{i}, ...
        lambda, sparsityParam, ...
        beta, higher_features), ...
        Theta_init, options);

        [higher_features] = feedForwardAutoencoder(Theta{i}, netconfig.layersizes{i}, ...
        netconfig.layersizes{i-1}, higher_features);

        stack{i}.w = reshape(Theta{i}(1:netconfig.layersizes{i}*netconfig.layersizes{i-1}), ...
        netconfig.layersizes{i}, netconfig.layersizes{i-1});

        stack{i}.b = Theta{i}(2*netconfig.layersizes{i}*netconfig.layersizes{i-1}+1:2*netconfig.layersizes{i}*netconfig.layersizes{i-1}+netconfig.layersizes{i});
    end


end


% [stackparams, ~] = stack2params(stack);  


