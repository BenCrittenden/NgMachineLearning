function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

%cost
hx = sigmoid(theta'*X');
T1 = (-y'*log(hx)');
T2 = (1-y)'*log(1-hx)';

J1 = (sum(T1-T2))./m;
J2 = (lambda./(2.*m)).*sum(theta(2:end).^2);

J = J1+J2;

%grad
grad(1) = (sum((hx'-y).*X(:,1))./m);

for a = 2:size(X,2)
    grad(a) = (sum((hx'-y).*X(:,a))./m) + (lambda./m).*theta(a);
end




% =============================================================

end