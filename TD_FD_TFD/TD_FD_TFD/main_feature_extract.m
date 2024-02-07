% Enter the path
inpath = input('ENTER THE INPUT PATH OF DATABANK: ','s');
features = zeros(225, 629);
testrun_time = zeros(225, 15);

for k=1:225   
 % number of readings
 fprintf('Loading data from Recording No:-%d \n',k);
 
 hpath = [inpath,'\preprocess_Reading',int2str(k),'.dat'];   %Healthy Samples Loading & Filtering................
 raw = load(hpath);
 
 % Feature Extraction
 fprintf('Feature Extraction for Recording No:-%d  is in progress......\n',k);  
 [features(k,:), testrun_time(k,:)] = func_FEATURE_EXTRACTION_MODULE_complete_20130907_v1(raw,50000,629);
 
end

str = [inpath,'\Features.csv']; 
dlmwrite(str,features);
% end