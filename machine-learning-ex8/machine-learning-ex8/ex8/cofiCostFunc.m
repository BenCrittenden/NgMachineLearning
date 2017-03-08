function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%%% Step 1, Calculate the un regularized Cost Function
%Do the matrix multiplication giving a movies x users matrix
X_Theta = X*Theta';

%Subtract the actual ratings from these predictions
XThetaY_diff = X_Theta - Y;

%Now make all values where the person didn't see the film equal to 0
error_term = XThetaY_diff .* R;

%Now square these values and take the sum, dividing the total by 2, as per
%the equation.
%Both of these do the same thing, the double sum is slightly faster

% J = (sum(inside_term(:).^2))./2; 
J = (sum(sum(error_term.^2)))./2;



%%% Step 2, Calculate the gradients
%Calculate the x gradient (order doesn't matter), using some of the values
%from above. X_grad dims are (movies x features) because X was originally a
% movies by features matrix, so here you get a gradient for each
% movie/feature pair.
%The for loop equivalent of this is that for the top left vlue you get the first row of
%the error_term (i.e. the error in each film in each individual) and
%multiply that by the first column of the theta (i.e. the value of the
%first feature in each individual) and then sum across the individuals.
%This gives you the gradient of the error for that the first feature in the
%first movie. 

X_grad = error_term * Theta;

%Now calucate the Theta_grad. Dims are users x features because Theta was a
%users x features matrix, so you need a gradient for each pair.

Theta_grad = error_term' * X;

%Now, you need to unroll the matrices and combine them, as that's the input
%that the rest of the script likes.


%%% Step 3, add regularization to the cost function
%This doesn't work properly, I'm not sure why

Theta_reg = sum(sum(Theta.^2));
Theta_reg = Theta_reg.*(lambda./2);

X_reg = sum(sum(X.^2));
X_reg = X_reg.*(X_reg./2);

J = J + X_reg + Theta_reg;


%%% Step 4, Regularize the gradient
%Also, not sure if this is working properly.

X_grad = X_grad + X.*lambda;
Theta_grad = Theta_grad + Theta.*lambda;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
