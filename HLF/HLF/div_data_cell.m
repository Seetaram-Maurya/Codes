function [A] = div_data_cell(data, group)

n = max(group);
A = cell(1,n);
count = zeros(n,1);

for i = 1:size(data,1)
    count(group(i)) = count(group(i)) + 1; %increasing the count.
    A{group(i)}(count(group(i)),:) = data(i,:); %
end

end
