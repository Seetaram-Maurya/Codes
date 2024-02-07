function [finetuned_stack, clust_param] = finetune_SA_fuzclsr(stack, data, group, classifier_cost_type, max_iter, lambda)

if nargin < 5
    options.maxIter = 300;% Maximum number of iterations of L-BFGS to run
    lambda = 1e-4;
elseif nargin < 6
    options.maxIter = max_iter;
    lambda = 1e-4;    
else
    options.maxIter = max_iter;    
end

hlevel_data = stackedAE_Out(data, stack);
[op_nodes num_samples] = size(hlevel_data);
unique_group = unique(group);
num_classes = length(unique_group);
class_data = cell(1,num_classes);
pred_op = zeros(op_nodes,num_samples); 
class_size = zeros(1,num_classes);
addpath minFunc/
options.Method = 'lbfgs'; 
clust_param = {};
%options.display = 'on';



if classifier_cost_type == 1
    for i = 1:num_classes
        ind = find(group == unique_group(i));
        class_data{i} = hlevel_data(:,ind);
        class_size(i) = size(class_data{i},2);
        %class_median{i} = median(class_data{i},2);
        pred_op(:,ind) = repmat(median(class_data{i},2),1,class_size(i));
    end
    % [stackparams, netconfig] = stack2params(stack);
    % [cost, grad] = feedforwardnet_Cost(stackparams, netconfig, lambda, data, pred_op);
    % numgrad = computeNumericalGradient( @(theta) feedforwardnet_Cost(theta, netconfig, lambda, data, pred_op), stackparams);
    % diff = norm(numgrad-grad)/norm(numgrad+grad);

    % disp('lolololol');
    % disp(diff);


    % exit;
    [stackparams, netconfig] = stack2params(stack);
    %[cost, grad] = feedforwardnet_Cost(stack, lambda, data, desired_out);
    [updated_params, cost] = minFunc( @(p) ...
        feedforwardnet_Cost(p, netconfig, lambda, data, pred_op) , ...
        stackparams, options);
    
    
elseif classifier_cost_type == 2
    clust_param = cell(op_nodes,1);
    for i = 1:op_nodes
        node_data = hlevel_data(i,:);
        [clust_indices, clust_centre, k, silcoeff_mat] = find_kmeans_1D(node_data',2);
        for j = 1:k
            ind = (clust_indices == j);
            pred_op(i,ind) = repmat(clust_centre(j),1,sum(ind));
        end
        clust_param{i} = clust_centre;
    end
    
    [stackparams, netconfig] = stack2params(stack);
    %[cost, grad] = feedforwardnet_Cost(stack, lambda, data, desired_out);
    [updated_params, cost] = minFunc( @(p) ...
        feedforwardnet_Cost(p, netconfig, lambda, data, pred_op) , ...
        stackparams, options);
    
    
elseif classifier_cost_type == 3
    [theta_init, netconfig] = stack2params(stack) ; 
    ga_options = gaoptimset(@ga);
    ga_options.InitialPopulation = theta_init';
    ga_options.Generations = length(theta_init) * 10;
    h = @(theta) fun_entropy_fitness(theta, netconfig, data, group, 1, 1);
    [updated_params,fval,exitflag,output,population,scores] = ga(h,length(theta_init),ga_options); 
end
    
    
finetuned_stack = params2stack(updated_params, netconfig);


% hlevel_data_new = stackedAE_Out(data, finetuned_stack);
% sqrt(sum(sum((hlevel_data_new - pred_op).^2)))
% sqrt(sum(sum((hlevel_data - pred_op).^2)))    
% tic
% ind = find(z<=0.5);
% sum(ind)
% toc
