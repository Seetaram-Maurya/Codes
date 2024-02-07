clear all;
close all;
clc;
FS=50000;
% number of readings
inpath='Piston\preprocess_Reading';
for k=1:225
  % fprintf('Loading data from Recording No:-%d \n',k);
  hpath = [inpath,int2str(k),'.dat'];   %Samples
  disp(hpath);
  raw = load(hpath);
  imf{k}=emd(raw);
end
for j=1:225
    for i=1:10
        powe=nextpow2(FS);
        H1=fft(imf{1,j}{1,i},2^powe);
        energy(i,:)= sum(abs(H1).^2);
        T{j,:}=(energy)/(sum(energy));
    end
end

% subplot(4,1,1)
% plot(preprocessReadingh);
% xlabel('(a)','FontSize',12,'FontWeight','bold');
% 
% subplot(4,1,2)
% plot(preprocessReadinglov);
% xlabel('(b)','FontSize',12,'FontWeight','bold');
% 
% subplot(4,1,3)
% plot(preprocessReadingliv);
% xlabel('(c)','FontSize',12,'FontWeight','bold');
% subplot(4,1,4)
% 
% plot(preprocessReadingp);
% xlabel({'(d)','Time (sec)'},'FontSize',12,'FontWeight','bold')
