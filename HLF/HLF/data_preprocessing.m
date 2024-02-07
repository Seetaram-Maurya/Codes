function [pre_processed_data, preprocess_param] = data_preprocessing(data, pre_process_type, nominal_ind)

[n m] = size(data);

if nargin < 2
    pre_process_type = 2;
    nominal_ind = [];
    real_ind = true(1,n)

elseif nargin < 3
    nominal_ind = [];
    real_ind = true(1,n)
    
else
    real_ind = true(1,n);
    real_ind(nominal_ind) = false;
end

preprocess_param = cell(n,1);

%% =========================================================================================
% Pre-processing of Real Valued Attributes

real_data = data(real_ind,:);
real_att_num = sum(real_ind);    

if pre_process_type == 1  % simple Scaling between 0-1
    
    pp_real_data = zeros(real_att_num,m);
    
    for i = 1:real_att_num
        node_data = real_data(i,:);
        preprocess_param{i}{1}(1) = min(node_data); 
        preprocess_param{i}{1}(2) = max(node_data); 
        preprocess_param{i}{2} = 'real_type_1';
        % Sigmoid MF fails when data values are below 1. Therefore not used here for 0-1 scaling.
       
        pp_real_data(i,:) = zero_one_scaling_1D(node_data, preprocess_param{i}{1}(1), preprocess_param{i}{1}(2));  %pp_real_data(i,:) = (node_data - repmat(preprocess_param{i}{1}(1), 1, m))./repmat((preprocess_param{i}{1}(2) - preprocess_param{i}{1}(1)),1,m);    
    end
    

%==============================================================

elseif pre_process_type == 2  %clustering followed by Gaussian Membership Function

    pp_real_data = [];

    parfor i = 1:real_att_num
        node_data = real_data(i,:);
        [clust_indices, clust_centre, k, silcoeff_mat] = find_kmeans_1D(node_data',2);
        pp_node_data = zeros(k, m);
        preprocess_param{i}{k+1} = 'real_type_2';
        [sort_clust_centre idx] = sort(clust_centre);

        for j = 1:k
            ind = find(clust_indices == idx(j));
            preprocess_param{i}{j} = [std(node_data(ind)), clust_centre(j,:)];
            pp_node_data(j,:) = gaussmf(node_data, preprocess_param{i}{j});
        end

        pp_real_data = [pp_real_data ; pp_node_data];
    end

%==============================================================
    
elseif pre_process_type == 3  % Clustering followed by normal scaling between 0-1.
    
    pp_real_data = [];

    parfor i = 1:real_att_num
        node_data = real_data(i,:);
        [clust_indices, clust_centre, k] = find_kmeans_1D(node_data',2);
        pp_node_data = zeros(k, m);
        preprocess_param{i}{k+1} = 'real_type_3';
        [sort_clust_centre idx] = sort(clust_centre); 

        for j = 1:k
            ind = find(clust_indices == idx(j)); clust_data = [];
            clust_data = node_data(ind); 
            preprocess_param{i}{j}(1) = min(clust_data);  preprocess_param{i}{j}(2) = max(clust_data); 
            %pp_node_data(j,:) = (clust_data - repmat(min_temp,1,length(clust_data)))./repmat((max_temp-min_temp),1,length(clust_data)) ; 
        end
        
        if k == 2
            preprocess_param{i}{1}(2) = (preprocess_param{i}{1}(2) + preprocess_param{i}{2}(1))/2 ;
            preprocess_param{i}{2}(1) = (preprocess_param{i}{1}(2) + preprocess_param{i}{2}(1))/2 ;
            
            pp_node_data(1,:) = zero_one_scaling_1D(node_data, preprocess_param{i}{1}(1), preprocess_param{i}{1}(2), 0, 1) ; 
            pp_node_data(2,:) = zero_one_scaling_1D(node_data, preprocess_param{i}{2}(1), preprocess_param{i}{2}(2), 1, 0) ; 
        
        else
            preprocess_param{i}{1}(2) = (preprocess_param{i}{1}(2) + preprocess_param{i}{2}(1))/2 ;
            pp_node_data(1,:) = zero_one_scaling_1D(node_data, preprocess_param{i}{1}(1), preprocess_param{i}{1}(2), 0, 1) ;
            
            for j = 2:(k-1)
                preprocess_param{i}{j}(1) = (preprocess_param{i}{j-1}(2) + preprocess_param{i}{j}(1))/2 ;
                preprocess_param{i}{j}(2) = (preprocess_param{i}{j}(2) + preprocess_param{i}{j+1}(1))/2 ;
                pp_node_data(j,:) = zero_one_scaling_1D(node_data, preprocess_param{i}{j}(1), preprocess_param{i}{j}(2), 1, 1) ; 
                
            end
            
            preprocess_param{i}{k}(1) = (preprocess_param{i}{k-1}(2) + preprocess_param{i}{k}(1))/2 ;
            pp_node_data(k,:) = zero_one_scaling_1D(node_data, preprocess_param{i}{k}(1), preprocess_param{i}{k}(2), 1, 0) ; 
        end
        
        pp_real_data = [pp_real_data ; pp_node_data] ; 
        
    end

end


%% =========================================================================================
% Pre-processing of Nominal Valued Attributes
 
nominal_data = data(nominal_ind,:); 
nominal_att_num = length(nominal_ind); 
pp_nominal_data = [];

for i = 1:nominal_att_num
    node_data = nominal_data(i,:);
    unique_elements = unique(node_data);
    k = length(unique_elements);
    pp_node_data = zeros(k, m);    
    preprocess_param{real_att_num+i}{k+1} = 'nominal_type';    
    
    for j = 1:k
        ind = find(node_data == unique_elements(j));
        preprocess_param{real_att_num+i}{j} = unique_elements(j);
        pp_node_data(j,ind) = 1;
    end
    
    pp_nominal_data = [pp_nominal_data ; pp_node_data];

end

pre_processed_data = [pp_real_data; pp_nominal_data]; 


% Sample Data
% patches = sampleIMAGES;
% n_data_1 = ceil(4*rand(1,10000));
% n_data_2 = ceil(2*rand(1,10000));
% n_data_3 = ceil(5*rand(1,10000));
% n_data_4 = ceil(3*rand(1,10000));