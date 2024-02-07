%  load('CV_FD_CWRUFeat_NorInner');
 k=5;
for i=1:k
        hlevel_data = train_data{i};
        trainLabels = train_gp{i};
        test_hlevel_data = test_data{i};
        testLabels = test_gp{i};

 	    B = TreeBagger(1000,hlevel_data,trainLabels);
 	    [Yfit,scores,stdevs] = predict(B,test_hlevel_data);
    	Yfit=cell2mat(Yfit);
    	Yfit=str2num(Yfit);
     	acc_Rf(i) = mean(testLabels(:) == Yfit(:));
end
acc_mean_rf=mean(acc_Rf);