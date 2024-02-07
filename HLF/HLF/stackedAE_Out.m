function [output] = stackedAE_Out(data, stack)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
[nfeatures, nsamples] = size(data);
depth = numel(stack);
a = cell(depth + 1, 1);
a{1} = data;
 
for layer = 1 : depth
a{layer + 1} = bsxfun(@plus, stack{layer}.w * a{layer}, stack{layer}.b);
a{layer + 1} = sigmoid(a{layer + 1});
end
 
output = a{depth+1};

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
