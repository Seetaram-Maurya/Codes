function [ ndata ] = normalise( data )
%FINDMAXMIN Summary of this function goes here
%   Detailed explanation goes here
%   [ ndata ] = findmaxmin( data )
%   ndata: normalised data between -1 to 1
%   data : takes input data as a vector(row/column)

N=length(data);
minPts=N*0.0005;
maximum=max(data);
minimum=min(data);
eps=(maximum-minimum)/1e6;
pn=ceil((maximum-minimum)/eps)+1;
partition=zeros(1,pn);

%put data in partition
for i=1:N
   pi=floor((data(i)-minimum)/eps)+1;
   partition(pi)=partition(pi)+1; 
end

Max=maximum;
Min=minimum;

%find required maximum
index_max=pn;
n=partition(index_max);
for i=1:pn
   if n >= minPts
       Max=index_max*eps+minimum;
       break;
   end 
   index_max=index_max-1;
   n=n+partition(index_max);
   
end

%find required minimum
n=0;
for i=1:pn
   n=n+partition(i); 
   if n >= minPts
       Min=maximum-(pn-i)*eps;
       break;
   end    
end
%normalise between -1 to 1
ndata=(2*data-Max-Min)/(Max-Min);
end
