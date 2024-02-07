close all;
clear all;
clc;
%% data segmentation 
load('innerrace_0_1797.mat');
data=X130_DE_time;
Start=1;
End=100;
for i=1:1210
    outerrace_DE_data(i,:)=data(Start:End,1);
    Start=End+1;
    End=(i+1)*100;
end
save('outerrace_DE_data');
%% input data formation

CWRU_data=[normal_DE_data;innerrace_DE_data;ball_DE_data;outerrace_DE_data];
save('CWRU_data');
% for j=0:1:3
append=[ones(1210,1);ones(1210,1)+1;ones(1210,1)+2;ones(1210,1)+3];
% end
CWRU_data_input=[CWRU_data append];
save('CWRU_data_input');

Freqny_features_Input=Freqny_features(1:end,9:136);



