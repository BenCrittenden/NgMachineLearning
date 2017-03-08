function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%===Step 1
%Compute the cost - step by step instruction available in ex1

%Get the predictions
h = X*theta;

%what's the error
error = h-y;

%what's the sum squared error
sum_sq_error = sum(error.^2);

%what's the unregularized cost
J = (1/(2.*m)) .* sum_sq_error;

%what's the regularization term (skipping the first term, which isn't
%regularized
reg = (lambda/(2*m)) * sum(theta(2:end).^2);

%what's the regularized cost
J = J + reg;



%===Step 2
%Compute the gradient step by step instruction available in ex1

%The error has already been calculated above, so just get onto the sums of
%the two terms

%Calculate the term inside the brackets (same in both gradients) and scale
%this with 1/m

nr_grad = (1/m).* (X'*error);

%now calculate the regularization term (to be added to all gradients except
%the first one)

reg_grad_term = (lambda/m).*(theta(2:end));

%add the regularization term to the later gradients
grad = nr_grad;
grad(2:end) = grad(2:end) + reg_grad_term;



% =========================================================================

grad = grad(:);

end
