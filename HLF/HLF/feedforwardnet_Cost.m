function [cost, grad] = feedforwardnet_Cost(theta, netconfig, lambda, data, desired_out)

stack = params2stack(theta, netconfig);
   
% stack: the neural net
% lambda: weight decay parameter
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example 
% desired_out : the output that is desired after passing through the feedforward network 

[m, n] = size(data); 

depth = numel(stack); 
W = cell(1,depth); b = cell(1,depth); 
Wgrad = cell(1,depth); bgrad = cell(1,depth);
Wgrad_temp = cell(1,depth); %bgrad_temp = cell(1,depth);
z = cell(1,depth); a = cell(1,depth+1); a{1} = data; 
cost = 0; delta = cell(1,depth+1); grad = [];


for i = 1:depth  % where i refers to layer of the network here.
    W{i} = stack{i}.w;
    b{i} = stack{i}.b;
    Wgrad{i} = zeros(size(W{i})); 
    bgrad{i} = zeros(size(b{i}));
    z{i} = W{i}*a{i} + repmat(b{i},1,n); 
    a{i+1} = sigmoid(z{i}); 
    cost = cost + 0.5*lambda*sum(sum(W{i}.^2)) ;
end


% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 


%% ----------  FInding MSE for Cost and Gradient values--------------------------------------


MSE = sum(sum((a{depth+1} - desired_out).^2))/(2*n) ;

cost = MSE + cost ; 

delta{depth+1} = -(desired_out - a{depth+1}).* a{depth+1}.*(1 - a{depth+1}) ; 

for i = depth:-1:1
    delta{i} = (W{i}'*delta{i+1}) .* a{i} .*(1 - a{i}) ;
    Wgrad_temp{i} = delta{i+1} * a{i}' ;
    Wgrad{i} = (Wgrad_temp{i}/n) + lambda*W{i} ;
    bgrad{i} = (sum(delta{i+1},2))/n ;
    grad = [Wgrad{i}(:); bgrad{i}(:); grad];
end



%delta3 = -(data - est_op).* a3.*(1-a3);
%delta2 = (W2'*delta3 + beta*((-repmat(rho,1,n)./a2) + ((1-repmat(rho,1,n))./(1-a2)))) .* a2.*(1-a2);

%delta2 = (W2'*delta3 + beta*(repmat(-rho./rho_est + (1-rho)./(1-rho_est), 1, n))) .* a2.*(1-a2);

%delta2 = zeros(hiddenSize,n);
%delta3 = zeros(visibleSize,n);

% % W1grad_temp = delta2 * data';
% % W2grad_temp = delta3 * a2';
%W2grad_temp = zeros(size(W2));
%W1grad_temp = zeros(size(W1));

%{
for i = 1:n
    delta3(:,i) = -(data(:,i) - est_op(:,i)).* a3(:,i).*(1-a3(:,i));
    delta2(:,i) = W2'*delta3(:,i) .* a2(:,i).*(1-a2(:,i));
    W2grad_temp = W2grad_temp + delta3(:,i)*(a2(:,i))';
    W1grad_temp = W1grad_temp + delta2(:,i)*(data(:,i))';
end
%}

% % b1grad_temp = sum(delta2,2);
% % b2grad_temp = sum(delta3,2);
% % 
% % W1grad = ((W1grad_temp/n) + lambda*W1) ;
% % W2grad = ((W2grad_temp/n) + lambda*W2) ;
% % b1grad = b1grad_temp / n ; 
% % b2grad = b2grad_temp / n ;
%% 

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

%grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)]; 

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 


function sigm = sigmoid(x)  
    sigm = 1 ./ (1 + exp(-x));
end

