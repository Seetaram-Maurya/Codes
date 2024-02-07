addpath ../libsvm/


%load('CV_health_piston.mat');
% load('CV_MWT_CWRUFeat_NorBall');

k=5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% with Linear kernel
% Finding the best parameters through grid search
%  prob_est1=cell(10,1);
%  prob_est2=cell(10,1);
j=1;
for j=1:k
   trainData = train_data{j};
    trainLabels = train_gp{j};
    testData = test_data{j};
    testLabels = test_gp{j};
    
    hlevel_data= trainData;
    test_hlevel_data = testData;
   % zero _one_normalization
	%trainData = (trainData - X.min_of_all*(ones(size(trainData))))./(X.max_of_all - X.min_of_all);
    %testData = (testData - X.min_of_all*(ones(size(testData))))./(X.max_of_all - X.min_of_all);
   
       % mean_var_normalization
       % trainData = mn_var_norm(trainData,X.mean_of_all,X.std_of_all);
       % testData = mn_var_norm(testData,X.mean_of_all,X.std_of_all);
    % median Based fine tunning
    % hlevel_data = stackedAE_Out(trainData',X.finetuned_stack{j});
    % test_hlevel_data = stackedAE_Out(testData',X.finetuned_stack{j});
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % softmax based fine tunning
    %theta = X.stackedAEOptTheta{j};
    %hlevel_data = stackedAE_Out1(theta, inputSize, hiddenSizeL3, numClasses, netconfig, trainData');
    %test_hlevel_data = stackedAE_Out1(theta, inputSize, hiddenSizeL3, numClasses, netconfig, testData');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    bestcv = 0;
    %%%%%%%%%tuning of parameters%%%%%%%%%
    for log2c = 1:15
        cmd = ['-t 0 -v 5 -c ', num2str(2^log2c), ' -q'];
        cv = svmtrain(trainLabels, hlevel_data, cmd);
        if (cv >= bestcv)
            bestcv = cv; bestc = log2c;
        end
        fprintf('%g %g (best c = %g, rate=%g)\n', log2c, cv, bestc, bestcv);
    end
    
    cur_bestc = bestc;
    
    % Finer grid search for finding best parameters
    for i = 1:20
        log2c = cur_bestc - 1 + 0.25*i;
        cmd = ['-t 0 -v 5 -c ', num2str(2^log2c), ' -q'];
        cv = svmtrain(trainLabels, hlevel_data, cmd);
        if (cv >= bestcv)
            bestcv = cv; bestc = 2^log2c;
        end
        fprintf('%g %g (best c = %g, rate=%g)\n', log2c, cv, bestc, bestcv);
    end
    
    cmd = ['-t 0 -c ', num2str(bestc),' -b 1'];
    
    % Final model made ready
    clsfr_model = svmtrain(trainLabels,hlevel_data, cmd);
    
    [pred_label, acc1, prob_est1] = svmpredict(testLabels, test_hlevel_data, clsfr_model); %%%'-b 1'
    
    [~,~,T,auc_rf1(i),optrocpt1] = perfcurve(testLabels,prob_est1,2);
    acc_lin(j) = acc1(1);
    
    
    
    
    
    
    
    
    %%
    bestcv = 0;
    % with RBF kernel
    % Finding the best parameters through grid search
    for log2c = -2:12
        for log2g = -10:4
            cmd = ['-t 2 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g),' -q'];
            cv = svmtrain(trainLabels, hlevel_data, cmd);
            if (cv >= bestcv)
                bestcv = cv; bestc = log2c; bestg = log2g;
            end
            fprintf('%g %g %g (best c = %g, best g = %g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
        end
    end
    
    cur_bestc = bestc; cur_bestg = bestg;
    
    % Finer grid search for finding best parameters
    for i = 1:7
        log2c = cur_bestc - 1 + 0.25*i;
        for k = 1:7
            log2g = cur_bestg - 1 + 0.25*k;
            cmd = ['-t 2 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g),' -q'];
            cv = svmtrain(trainLabels, hlevel_data, cmd);
            if (cv >= bestcv)
                bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
            end
            fprintf('%g %g %g (best c = %g, best g = %g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
        end
    end
    
    cmd = ['-t 2 -c ', num2str(bestc), ' -g ', num2str(bestg),' -b 1'];
    
    % Final model made ready
    clsfr_model = svmtrain(trainLabels, hlevel_data, cmd);
    
    [pred_label, acc2, prob_est2] = svmpredict(testLabels, test_hlevel_data, clsfr_model);%%, '-b 1');
    
    [~,~,T,auc_rf1(i),optrocpt2] = perfcurve(testLabels,prob_est2,2);
    acc_rbf(j) = acc2(1);
    
   %    Random forest
   
   
%    B = TreeBagger(1000,hlevel_data,trainLabels);
%  	[Yfit,scores,stdevs] = predict(B,test_hlevel_data);
%     	Yfit=cell2mat(Yfit);
%     	Yfit=str2num(Yfit);
%      	acc_rf(i) = mean(testLabels(:) == Yfit(:));
%      	[~,~,T,acc_rf_simp(i)] = perfcurve(testLabels,scores(:,1),1);
%         
%      B = TreeBagger(1000,trainData,trainLabels);
%  	[Yfit,scores,stdevs] = predict(B,testData);
%     	Yfit=cell2mat(Yfit);
%     	Yfit=str2num(Yfit);
%      	acc_rf_simp(i) = mean(testLabels(:) == Yfit(:));
end

acc_mean_lin=mean(acc_lin);
acc_mean_rbf=mean(acc_rbf);


