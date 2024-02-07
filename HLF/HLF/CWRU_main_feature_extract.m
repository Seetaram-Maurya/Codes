for i=1:size(CWRU_data,1)
[features(i,:)] = fun_featureExtraction_generalData(CWRU_data(i,:),12000,2);
end 