function [selected_features, mean_train, transform_mat] = find_pca(traindata, num_selected_fea, mean_train, transform_mat)

row = size(traindata, 1);    

if nargin >= 1 && nargin < 3
    mean_train = mean(traindata);
    norm_traindata = traindata - repmat(mean_train,row,1);
    [U S V] = svd(norm_traindata);
    transform_mat = V;  
end

norm_traindata = traindata - repmat(mean_train,row,1);
selected_features = norm_traindata * transform_mat(:,1:num_selected_fea);

end