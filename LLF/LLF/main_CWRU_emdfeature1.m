% clear all;
close all;
clc;
FS=12000;
% number of readings
% inpath='Piston\preprocess_Reading';
for k=1:1210  % fprintf('Loading data from Recording No:-%d \n',k);
%   hpath = [inpath,int2str(k),'.dat'];   %Samples
%   disp(hpath);
%   raw = load(hpath);
  imfNormal{k}=emd(CWRU_FeaturesOuter(k,:));
end
for j=1:1210
    for i=1:4
        powe=nextpow2(FS);
        H1=fft(imfNormal{1,j}{1,i},2^powe);
        energy(i,:)= sum(abs(H1).^2);
        CWRU_FeaturesOuter{j,:}=(energy)/(sum(energy));
    end
end
