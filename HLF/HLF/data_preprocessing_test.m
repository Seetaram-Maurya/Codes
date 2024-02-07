function [pre_processed_data] = data_preprocessing_test(data, preprocess_param)

[n m] = size(data);

size_ppdata = 0;

for i = 1:size(preprocess_param,1)
    size_ppdata = size_ppdata + length(preprocess_param{i});
end

size_ppdata = size_ppdata - size(preprocess_param,1);
pre_processed_data = zeros(size_ppdata, m); %Initializing the matrix containing pre-processed data

%% =========================================================================================
% Feature by feature pre-processing of data
size_ppdata = 0;

for i = 1:n
    node_data = data(i,:); 
    
    if strcmp(preprocess_param{i}{end},'real_type_1')
        k = 1;
        pp_node_data = zeros(1,m);
        pp_node_data = zero_one_scaling_1D(node_data, preprocess_param{i}{1}(1), preprocess_param{i}{1}(2));  %pp_node_data = (node_data - repmat(preprocess_param{i}{1}(1), 1, m))./repmat((preprocess_param{i}{1}(2) - preprocess_param{i}{1}(1)),1,m);    
 
        
    
    elseif strcmp(preprocess_param{i}{end},'real_type_2')
        
        k = length(preprocess_param{i}) - 1;
        pp_node_data = zeros(k, m);
        
        for j = 1:k
            pp_node_data(j,:) = gaussmf(node_data, preprocess_param{i}{j});
        end
        
        
        
    elseif strcmp(preprocess_param{i}{end},'real_type_3')
        
        k = length(preprocess_param{i}) - 1;
        
        if k == 2
            pp_node_data(1,:) = zero_one_scaling_1D(node_data, preprocess_param{i}{1}(1), preprocess_param{i}{1}(2), 0, 1) ; 
            pp_node_data(2,:) = zero_one_scaling_1D(node_data, preprocess_param{i}{2}(1), preprocess_param{i}{2}(2), 1, 0) ; 
        
        else
            pp_node_data(1,:) = zero_one_scaling_1D(node_data, preprocess_param{i}{1}(1), preprocess_param{i}{1}(2), 0, 1) ; 
            
            for j = 2:(k-1)
                pp_node_data(j,:) = zero_one_scaling_1D(node_data, preprocess_param{i}{j}(1), preprocess_param{i}{j}(2), 1, 1) ; 
            end
            
            pp_node_data(k,:) = zero_one_scaling_1D(node_data, preprocess_param{i}{k}(1), preprocess_param{i}{k}(2), 1, 0) ; 
        end
        
        
        
    elseif strcmp(preprocess_param{i}{end},'nominal_type')
        
        k = length(preprocess_param{i})-1;
        pp_node_data = zeros(k, m);
        
        for j = 1:k
            pp_node_data(j,:) = double(node_data == repmat(preprocess_param{i}{j}, 1, m));
        end
    
    end
    
    pre_processed_data(size_ppdata+1:size_ppdata+k,:) = pp_node_data(1:k,:) ;
    size_ppdata = size_ppdata + k;

end


