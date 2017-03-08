function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%===Step 1
%First thing is that you need to convert the y variable which says what the
%actual number was for each example into a matrix that has a 1 in the
%correct location for the correct answer and zeros otherwise. 
%Basically the output needs to be a vector that you can map onto each
%output unit saying if it should be a 1 or 0. And you need a vector for
%each example, so the result is a:
% m_examples x ou_ouput_units matrix.

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

%===Step 2
%Add the vector of 1's to the X matrix so that matrix multiplication
%becomes possible. Set this as the activation of layer 1, i.e. a1.

a1 = [ones(m,1) X];


%===Step 3
%To calculate the activation of the next layer first you must calcuate the
%z value, then pass this trough the sigmoid function to get the activation

%the z for each row is the sum of the product of the activation a1 and the
%respective theta values for that connection in theta. By matrix
%multiplication:
%a1 is (m_examples x n_features(+1))
%Theta1 is (n_hidden_units x n_features(+1)) - needs to be transposed
%The results is a:
%z2 is a matrix of m_examples x n_hidden_units;

z2 = a1*Theta1';

a2 = sigmoid(z2);

%===Step 4
%More or less same as steps 2 and 3:
%add a row of 1's (bias) at the beginning of a2
%Multiply this modified a2 with the second theta matrix
%produce a matrix of m_examples x n_output_units.

a2_b = [ones(m,1) a2];

z3 = a2_b*Theta2';

a3 = sigmoid(z3);


%===Step 5
%Calcuate the non-regularized cost function, using the formula
%First calcualte the two terms inside the sums
%For this there is no matrix multiplication - just element wise
%multiplication of the y_matrix and the log transfored activation of the
%output layer. This makes sense - you want a value fore each output unit
%for each example, so the result should be a m_examples x n_output_units
%matrix.


t1 = -y_matrix.*(log(a3));
t2 = (1-y_matrix).*log(1-a3);
        
err = t1-t2;

%With this error, you need to sum across units then examples, as in the
%formula (although in practice the order can be either way around). The
%answer is a scalar (because the the cost function J is ultimately also a
%scalar.

summed_terms = sum(sum(err));

%Then multipliply this sum by 1/m to get the cost J (a scalar)

J = (1/size(X,1)).*summed_terms;


%===Step 6
%Calculate the regularization term
%Calculate each of the sums seperately then plub them into the term.
%Firstly, ignore the first column of the Theta matrix because as it's not
%included in the regularization term (hence the 2:end)
%So square each element of the theta matrix then do the double sum, in
%theory over the earlier layer then the latter layer.
%each r term is a scalar

r1 = sum(sum(Theta1(:,2:end).^2));
r2 = sum(sum(Theta2(:,2:end).^2));

%now combine them and multiply by (lambda/2m)

lambda = 1;
R = (lambda/(2*m)).*(r1+r2);


%=============================
%Backpropagation

%===Step 1
%First step is to calculate the delta's of the output units. This is simply
%the subtraction of the a3's from the y's
%this error is a 

d3 = a3-y_matrix;


%===Step 2
%Second step is to calculate the delta's of the second layer, the hidden
%units.
%This requires the theta's going from l2 to l3 i.e. Theta2 and d3, which we
%just calculated. Plus z2 i.e. the not normalized activation of the second
%level which is needed for the sigmoid gradient calculation
%Ignore the first column of the theta matrix (i.e. use 2:end)


%First multiply the m_examples x n_output_units error of the output layer
%(d3) with the n_output_units x n_hidden_units Theta values (Theta2 -1).
%Produces a m_examples x n_hidden_units matrix. This is like taking the sum
%of the product of all the d3's and their respective theta values for each
%of the hidden_units, for each example. i.e. for one example and one hidden
%unit it is: 
%sum( d3_1*theta2_1 + d3_2*theta2_2 + d3_3*theta2_3+ ... + d3_10*theta2_10)

dt = d3*Theta2(:,2:end);

%no pass the z2 values through the gradient sigmoid function. This is a
%scalar operation, so the result is the same dimesions as z2.

sg = sigmoidGradient(z2);

%Put these two values together by doing element-wise matrix multiplication.
%The result has the same dimesions as both input matrices (if they were
%different sizes it wouldn't work).

d2 = dt .* sg;



%===Step 3
%calculate the delta accumulator for the second level (there is no delta in
%layer 1).
%The point of the next few steps is to calcuate the theta gradient of all
%the theta values - i.e. how much each theta value should change in the
%next iteration. Therefore the outputs will need to be a matrix with
%dimensions of the number of units of the layers between which the theta
%connections are connecting.
% Here you want to multiply the delta's of each unit with the activation
% values of the units from the previous level for each example and then sum
% over all the examples. So for each example you're getting the accumulated
% D for each hidden_unit-feature pair and then you just add these all
% together.
% This can be done with matrix multiplication:
% n_hidden_units x m_examples  x   m_examples x n_features
% result is a n_hidden_units x n_features matrix

D2 = d2'*a1;


%===Step 4
%calculate the delta accumulator for the third level
%Same principle as before, but here you use the deltas from the output layer
%and the activations from level 2.
%n_output_units x m_examples   x   m_examples x n_hidden_units(+ 1 bias)
%results is a n_output_units x n_hidden_units

D3 = d3'*a2_b;



%===Step 5
%calculate the gradient of the theta values by scaling each of the D delta
%values by (1/m), as per the unregularized theta gradient equation.

Theta1_grad = D2./m;
Theta2_grad = D3./m;



%===Step 6
%Regularization of the theta gradients
%The first column of the theta term (corresponding to the bias) isn't
%regularized,so you either do this column seperately or it needs to be set 
%to 0's so that it has no influence.
%Once you've got the regularization term for each theta_grad then add it to
%it.

%Do this for the fist theta gradient
Theta1_temp = Theta1;
Theta1_temp(:,1) = 0;
r_grad1 = (lambda/m).*Theta1_temp;

Theta1_grad = Theta1_grad + r_grad1; %because first col=0 there is no update of these values


%And do it again for the second theta gradient
Theta2_temp = Theta2;
Theta2_temp(:,1) = 0;
r_grad2 = (lambda/m).*Theta2_temp;

Theta2_grad = Theta2_grad + r_grad2;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
