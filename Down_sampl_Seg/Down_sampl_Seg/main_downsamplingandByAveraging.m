
for k=1:225   
 % Enter the path
 path1 = ['E:\seetaram\research\cbm\cbm_code\TrainingData_2011\Piston\preprocess_Reading',int2str(k),'.dat'];  
 Piston = load(path1);
 
 % down sampling
 Piston_raw_down_50(k,:)=downsample(Piston,50);
 
 % Feature Extraction
%  [LowLOV_features_DunSmplig_5(k,:), FD_testrun_time(k,:)] = fun_featureExtraction_generalData(down_raw,25000,286);
 
end
%%   moving average
% for j = 3:50:1000
%     smdata(i)=(a(j-2)+a(j-1)+a(j)+a(j+1)+a(j+2))/5;
%     i=i+1;
% end

for jj=1:225
    path1 = ['E:\seetaram\research\cbm\cbm_code\TrainingData_2011\Piston\preprocess_Reading',int2str(jj),'.dat'];
    Piston = load(path1);
    i=1;
    k=0;
for j = 50:50:50000
    Avrgdata_Piston(jj,i)=sum(Piston(1+50*k:j))/50;
    i=i+1;
    k=k+1;
end
end





