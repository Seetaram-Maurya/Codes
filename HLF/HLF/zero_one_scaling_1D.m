function [output] = zero_one_scaling_1D(data, min_val, max_val, min_flatten, max_flatten)

if nargin < 4
    min_flatten = 0;
    max_flatten = 0;
end

[n, m] = size(data);
output = (data - repmat(min_val,1,m))./repmat((max_val - min_val),1,m) ; 

if min_flatten == 1
    below_min = (data < min_val);
    output(below_min) = 0;
end

if max_flatten == 1
    above_max = (data > max_val);
    output(above_max) = 1;
end


