function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% for z = 1:size(X)
%     
%     a(z) = theta(1) + theta(2).*X(z,2);
%     
% end
% 
% ad = a'-y;
% ads = ad.^2;
% 
% s_ad = sum(ads);
% 
% J1 = s_ad ./ (2*m);
% 
% 
% J = sum(((theta'*X')-y').^2)./ (2*m);


%===Step 1
%X is a m_examples x n_features matrix, theta is a n_features x 1 matrix
%h should be a mx1 matrix.
%

%The loop method: multiply each value of X by the corresponding feature and
%add them together
for i = 1:size(X,1)
    h(i,1) = X(i,1).*theta(1) + X(i,2).*theta(2);
end

%Or as a matrix multiplication:
h = X*theta;


%===Step 2
%Calculater the error - the difference between the actual value, y, and the
%guess, h. Answer is a mx1 vector.

error = y - h;


%===Step 3
%Now get the sum of the squared error term. Each element is squared, this
%is not a matrix multiplication. The answer is a scalar.

sum_sq_err = sum(error.^2);


%===Step 4
%Now caluclate J using the cost function formula. Answer is a scalar.

J = (1/(2.*m)) .* sum_sq_err;


% =========================================================================

end
