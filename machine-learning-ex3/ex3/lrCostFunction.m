function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%===Step 1
%what's the guess of what you think will happen (h0) which is calculated
%using the sigmoid function. Remember that X*theta is the sum of all
%features multiplied by their corresponding theta, done for each example,
%i.e. each row individually. h0 is thus a m_examples x 1 vector.

h0 = sigmoid(X*theta);

%===Step 2
%Calculate the two terms inside the brackets of the cost function using
%matrix multiplication and then add them together as per the cost function
T1 = -y'*(log(h0));
T2 = (1-y)'*(log(1-h0));

J = (sum(T1-T2))./m;


%===Step 3
%Calculate the gradient. Using the error between h0 and y, use this error
%vector to multiply with each column of the X matrix to generate a gradient
%vecotr of n_features x 1.
% grad2 = (sum((h0-y).*X'))./m;

error = h0-y;
grad = (X'*error)./m;


% =============================================================

grad = grad(:);

end
