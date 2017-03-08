function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

%===Step 1
% z is a vector (or scalar) and this function implements the sigmoid
% function on the vector element-wise

g = 1./(1 + exp(-z));




% =============================================================

end
