labeled_data=[DB_FusedNormal;DB_FusedFlank];
labeled_group=[A;B];
k=5;
c=cvpartition(labeled_group,'kfold',k);
for i=1:k
    train_data{i}=labeled_data(training(c,i),:);
    train_gp{i}=labeled_group(training(c,i),:);
    test_data{i}=labeled_data(test(c,i),:);
    test_gp{i}=labeled_group(test(c,i),:);
end
save('CV_Nor_DB_Fused_Flank5','train_data','train_gp','test_data','test_gp', '-v7.3')

% DB_FusedNormal=[DB_HealthyFeatures,DBfff_hlevel_data(1:1214,:)];
% DB_FusedFlank=[DB_flankFeatures,DBfff_hlevel_data(1215:2428,:)];
% DB_FusedChisel=[DB_chiselFeatures,DBfff_hlevel_data(2429:3642,:)];
% DB_FusedOuter=[DB_outerFeatures,DBfff_hlevel_data(3643:4856,:)];
% a=[IMS_FeaturesNormal,hlevel_dataNormal20];
% b=[IMS_FeaturesOuterFailure,hlevel_dataOut20];
% DBfff_hlevel_data(1:1214,:);