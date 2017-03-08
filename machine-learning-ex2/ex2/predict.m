function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

%===Step 1
%multiply each each element in column of X by the corresponding theta
%value. Then sum over the values in each row.  Can be a loop, or just do it
%with vectors. Here's the vector way.

h = sigmoid(X*theta);

%===Step 2
%set values greater than 0.5 to 1 and less to 0.
h(h>=0.5) = 1;
h(h<0.5) = 0;

p = h;



% =========================================================================


end
