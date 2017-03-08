function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

%===Step 1
%Take the the product of the X matrix (m_examples x n_features) and the
%theta (n_features x 1) vector. Produces an m_examples x 1 vector.

h = sigmoid(X*theta);


%===Step 2 
%Calculate the first and second terms (either side of the sum) of the cost
%function.
%The first term is the sum of the product of the y (mx1) and h(mx1)
%vectors. This can be done by summing element-wise multiplication of the
%two vectors and taking the sum or a matrix multiplication (but the y
%vector needs to be transposed). The result is a scalar.

%non-vector method:
T1 = sum(y.*log(h));
T2 = (1-y).*log(1-h);

%vector method:
T1 = (-y'*log(h));
T2 = (1-y)'*log(1-h);


%===Step 3
%Calculate the cost function by summing the two T terms just calculated and
%dividing by the number of examples, as per the equation. Answer is a
%scalar

J = (sum(T1-T2))./m;


%===Step 4
%Calculate the gradient. First calculate the error by subtracting y from h.
%This give an mx1 vector

error = h-y;

%===Step 5
%get the new gradient using either a loop or vector multiplication. If a
%loop you have to treat each column of X (i.e. each feature sperately) as a 
%vector to get the gradient change for that feature. So taking each column 
%in turn, you multiply the two vectors element wise and take the sum. Then
%loop over the other features.
%vector it's simple vector multiplication.
%also divide each by the number of examples, m. The gradient is a vector of
%n_features x 1.

%loop method
for a = 1:size(X,2)
    grad(a) = sum(error.*X(:,a))./m;
end

%vector method
grad = (X'*error)./m;



% =============================================================

end
