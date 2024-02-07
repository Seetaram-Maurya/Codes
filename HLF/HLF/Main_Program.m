 % Modified on Dec 28th 2016
 % Modified on Jan 02 2017 
 % Modified on Nov 12th 2017
addpath ('minFunc');
name='AirCompressorDatasetWPT'
 %% Load your data
 alldata = csvread('AirCompressorDatasetWPT.csv');
 data = alldata(:,1:end-1)';
 group = alldata(:,end)';
 mkdir(name)
 p=9; % loop for varying sparisity constraints 
 jj=1; % loop for hpc job
 k = 5;% variable for setting values of kfold 
 classifier_type = 1; %to decide finetuning type 1 = median based 
 nominal_ind = [];
 c = cvpartition(group,'kfold',k); % for diving data into training and testing data
 %Intialize parameters
 pp_train_data = cell(1,k);  pp_test_data = cell(1,k);
 train_data = cell(1,k); train_gp = cell(1,k); 
 test_data = cell(1,k); test_gp = cell(1,k);
 stack = cell(1,k); finetuned_stack = cell(1,k);netconfig = cell(1,k);
 Train_accuracy = zeros(k,p);  Test_accuracy = zeros(k,p);
 tic_preprocess = tic;
 
% This loop forms 5 sets of preprocessed training and testing data
  for i = 1:k
    train_data{i} = data(:,training(c,i));
    train_gp{i} = group(:,training(c,i));
    test_data{i} = data(:,test(c,i));
    test_gp{i} = group(:,test(c,i));
    % Performing data pre-processing. As learning is involved, the necessary parameters are learnt and stored in preprocess_param, for future use.
     [pp_train_data{i}, preprocess_param{i}] = data_preprocessing(train_data{i},1,nominal_ind) ;  
     [pp_test_data{i}] = data_preprocessing_test(test_data{i}, preprocess_param{i});   % Performing data pre-processing of test data with learnt pre-processing parameters.
 end
% Defining the structure of the neural net
for i = 1:k
    netconfig{i}.inputsize = size(train_data{i},1);
    netconfig{i}.layersizes{1} = 200;
    netconfig{i}.layersizes{2} = 150;
    netconfig{i}.layersizes{3} = 50;
end
clear train_data test_data;

% Main Loop 
for j = 1:p
    sparsityparam = 0.1*j;
    tic_classify = tic;
    clear stack hlevel_data;
% Training Phase
    for i = 1:k
         stack{1}{i} = train_stacked_autoencoder(pp_train_data{i}, netconfig{i},sparsityparam); %, sparsityParam, beta, maxIter, lambda); % Train a stacked sparse autoencoder, as per the above neural net structure to get good initial weights
         [finetuned_stack{1}{i}] = finetune_SA_fuzclsr(stack{1}{i}, pp_train_data{i}, train_gp{i}, classifier_type);  % Once the autoencoder is trained, the weights are finetuned in a manner that facilitates better rule reduction/classification performance (based on heuristics)
         hlevel_data{i} = stackedAE_Out(pp_train_data{i}, finetuned_stack{1}{i}); %Generate the high level features that will be used for learning the fuzzy classifier.
         test_hlevel_data{i} = stackedAE_Out(pp_test_data{i}, finetuned_stack{1}{i}); %Generate the high level features that 
    end
    fprintf(1,'\nCompleted for Sparsity Parameter------------------------------%d-',j);
    save(strcat(name,'/stack_',int2str(j),'.mat'),'stack')   % save variable in the output.mat file
    save(strcat(name,'/base_ft_',int2str(j),'.mat'),'finetuned_stack')   % save variable in the output.mat file 
    save(strcat(name,'/train_hlevel_features',int2str(j),'.mat'),'hlevel_data')   % save variable in the output.mat file
    save(strcat(name,'/test_hlevel_features',int2str(j),'.mat'),'test_hlevel_data')   % save variable in the output.mat file 
   
end
save ('FT1_SAE_WS_AirCompressorDatasetWPT','pp_train_data','pp_test_data','train_gp','test_gp');

