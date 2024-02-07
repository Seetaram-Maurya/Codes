
%% ------------------EMD---------------------------
subplot(1,3,1);
% Load data
% load 'Air_EMDLE_feature.mat';
train_X=[CWRU_FusedFeatures(1:1210,1:4);CWRU_FusedFeatures(3631:4840,1:4)];
% train_X=train_X(:,20:30);
ind = randperm(size(train_X, 1));
train_X = train_X(ind(1:1210),:);
% train_labels=[ones(225,1);ones(225,1)+1;ones(225,1)+2;ones(225,1)+3];
train_labels=[AA;AAA];
train_labels = train_labels(ind(1:1210));
% Set parameters
no_dims = 2;
initial_dims = 4;
perplexity = 10;
% Run t?SNE
mappedX = tsne(train_X, [], no_dims, initial_dims, perplexity);
% Plot results
gscatter(mappedX(:,1), mappedX(:,2), train_labels,'rkgbmc','*',6,'on','Dimension-1','Dimension-2');
%% ------------------DNN---------------------------
subplot(1,3,2);
% Load data
% load 'Air_EMDLE_feature.mat';
% train_X=[CWRU_data(1:1210,:);CWRU_data(2421:3630,:)];
train_X=[CWRU_FusedFeatures(1:1210,5:14);CWRU_FusedFeatures(3631:4840,5:14)];
% train_X=train_X(:,20:30);
ind = randperm(size(train_X, 1));
train_X = train_X(ind(1:1210),:);
train_labels=[AA;AAA];
train_labels = train_labels(ind(1:1210));
% Set parameters
no_dims = 2;
initial_dims = 8;
perplexity = 10;
% Run t?SNE
mappedX = tsne(train_X, [], no_dims, initial_dims, perplexity);
% Plot results
gscatter(mappedX(:,1), mappedX(:,2), train_labels,'rkgbmc','*',6,'on','Dimension-1','Dimension-2');
%% ------------------for proposed with PCA---------------------------
subplot(1,3,3);
% Load data
% load 'Air_EMDLE_feature.mat';
train_X=sel_fea_pca;
% train_X=[CWRU_FusedFeatures(1:1210,:);CWRU_FusedFeatures(3631:4840,:)];
% train_X=[CWRU_data(1:1210,:);CWRU_data(3631:4840,:)];
ind = randperm(size(train_X, 1));       
train_X = train_X(ind(1:1210),:);
train_labels=[AA;AAA];
train_labels = train_labels(ind(1:1210));
% Set parameters
no_dims = 2;
initial_dims = 8;
perplexity = 20;
% Run t?SNE
mappedX = tsne(train_X, [], no_dims, initial_dims, perplexity);
% Plot results
gscatter(mappedX(:,1), mappedX(:,2), train_labels,'rkgbmc','*',6,'on','Dimension-1','Dimension-2');