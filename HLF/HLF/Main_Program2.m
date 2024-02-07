addpath ('minFunc');
addpath ('libsvm');
% name='AirCompWPT'
 %% Load your data
 alldata = CWRU_time_feature_input;
 data = alldata(:,1:end-1)';
 group = alldata(:,end)';
 p=1; % loop for varying sparisity constraints 
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
    netconfig{i}.layersizes{1} = 80;
    netconfig{i}.layersizes{2} = 60;
    netconfig{i}.layersizes{3} = 30;
end
clear train_data test_data;

%% Main Loop 
for j = 1
    sparsityparam = 0.5;
    tic_classify = tic;
    clear stack hlevel_data;
%  Training Phase
    for i = 1:k
        stack{1}{i} = train_stacked_autoencoder(pp_train_data{i}, netconfig{i},sparsityparam); %, sparsityParam, beta, maxIter, lambda); % Train a stacked sparse autoencoder, as per the above neural net structure to get good initial weights
        [finetuned_stack{1}{i}] = finetune_SA_fuzclsr(stack{1}{i}, pp_train_data{i}, train_gp{i}, classifier_type);  % Once the autoencoder is trained, the weights are finetuned in a manner that facilitates better rule reduction/classification performance (based on heuristics)
        hlevel_data{i} = stackedAE_Out(pp_train_data{i}, finetuned_stack{1}{i}); %Generate the high level features that will be used for learning the fuzzy classifier.
        % Classifier
        A{i} = div_data_cell(hlevel_data{i},train_gp{i});
       [Bin_clsfr{i}, l_node, r_node] = gen_classifier_OAO(A{i});
        L_Node{i} = l_node; R_Node{i} = r_node;
% Testing Phase
        test_hlevel_data{i} = stackedAE_Out(pp_test_data{i}, finetuned_stack{1}{i}); %Generate the high level features that 
        [res_class{i}] = classify_clsfr_OAO(test_hlevel_data{i}, Bin_clsfr{i}, l_node, r_node);
        err_number(i) = sum(res_class{i} ~= test_gp);
        Accuracy(i) = (length(test_gp) - err_number(i)) / length(test_gp);
    end
     mean_testAccuracy(1,j)= mean(Accuracy);
end

