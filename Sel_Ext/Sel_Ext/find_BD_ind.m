function [feature_ind]=find_BD_ind(A,num_fea)
% Input  : feature matrix
% Output : Ranked features
%
[m n]=size(A);
mu=50;sigma=400;u=20;

for x=1:m
    y(x)=u*exp(-(x-mu).^2/(2*sigma^2));
end
for j=1:n
    [nsigma,nmu,nu]=mygaussfit(A(:,j)',y');
    for i=1:m
        pdf(i,j)=nu*exp(-(A(i,j)-nmu).^2/(2*nsigma^2));
    end
end
idx=zeros(3,n*(n-1)/2);
k=1;

for i=1:n
    for j=(i+1):n
        idx(1,k)=bhattacharyya(pdf(:,i),pdf(:,j));
        idx(2,k)=i;
        idx(3,k)=j;
        k=k+1;
    end
end

id(1,:)=idx(2,:);
id(2,:)=idx(3,:);
for i=1:n
    [r c]=find(id==i);
    for j=1:numel(c)
    temp(j)=idx(1,c(j));
    end
    rating(i)=mean(temp);
end

[a selected_index]=sort(rating,'descend');
feature_ind = selected_index(1:num_fea);

clear rating id idx temp pdf y
end