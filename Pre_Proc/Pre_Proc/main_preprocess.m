% Design Filter
Fs = 50000;            %sampling frequency
Ts = 1/Fs;             %sampling time interval
filter_freq = 12000;   %butterworth frequency
[B, A] = butter(18, (filter_freq*2 / (Fs)), 'low');
% for u=1:1

inpath=input(' ENTER THE INPUT PATH OF DATABANK: ','s');
%  inpath=['D:\ZPOOL\Dataset ',int2str(u)];

for k=1:225   
 % number of readings
 fprintf('Loading data from Recording No:-%d \n',k);
 hpath = [inpath,'Reading',int2str(k),'.dat'];   %Healthy Samples Loading & Filtering................
 raw = load(hpath);
 raw = raw';
 raw = raw(:);
 % Preprocessing.........
 fprintf('PreProcessing for Recording No:-%d  is in progress......\n',k);  
 preprocess_data = func_preprocess(raw,B,A);    %Healthy
 str = [inpath,'preprocess_Reading',int2str(k),'.dat']; 
 dlmwrite(str,preprocess_data);
  
end

% end