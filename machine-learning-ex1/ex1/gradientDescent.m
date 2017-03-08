function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    %===Step 1
    %Calculate the hypothesis by summing accros the product of all the
    %features and their respective theta values. To do this with a loop see
    %the computeCost.m function, otherwise just do it with matrix
    %multiplication.
    
    %X is an m_examples x n_features matrix. Theta is an n_features x 1
    %vector. The result h is a m x 1 vector.
    h = X * theta;
    
    
    %===Step 2
    %Get the error by subtracting the y vector from the h vector, giving an
    %m x 1 error vector
    
    error = h - y;
    
    
    %===Step 3
    % Get the bit of the gradient that is inside the brackets. Remember
    % that the gradient function is different for the first term theta0
    % than the other theta terms.
    
    inside_term(1) = sum(error);
    inside_term(2) = sum(error.*X(:,2));
    
    %fortunately, because the first column of X is all 1's, we could also
    %do this via matrix multiplication. The result is the same. However,
    %first we need to transpose X, to a n_features x m_examples matrix to
    %multiply it with the m_examples x 1 error vector. This gives a
    %n_features x 1 vector.
    
    inside_term = X'*error;
    
    
    %===Step 4
    %Scale the inside term by (alpha/m), which is a scalar, as per the 
    %gradient function. The difference term is a n x 1 vector.
    
    dt = (alpha./m).*inside_term;
    
    %===Step 5
    %Subtract these difference values from the old theta values to get the 
    %new theta values. As both are nx1 vectors, the outcome is also an
    %nx1 vector.
    
    theta_new = theta - dt;
    
    theta = theta_new; %asign these new thetas to the theta variable

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
