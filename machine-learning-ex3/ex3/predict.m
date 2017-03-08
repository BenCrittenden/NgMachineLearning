function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%===Step 1
% First of all, the X matrix doesn't have a constant term, so need to add a
% column of 1s to the first column.

X = [ones(m,1) X];


%===Step 2
%Get the sum of the product of each X with it's corresponding theta. Use
%matrix multiplication. 
%X is a m_examples x n_features(+1 for the 1's col)
%Theta 1 is a matrix of 25_hidden_neurons x 401_features (i.e. a theta
%value for every connection. 
%The input_layer is 400 feature nodes + the constant unit and the hidden
%layer is 25 units - thus there are 401x25 connections and hence a 401x25
%matrix of theta values.
%The output should be a m_examples by hu_hidden units matrix

layer_h = X*Theta1'; %calculate hidden layer



%===Step 3
%Add a column of ones to the first matrix so that you can do easy matrix
%multiplication with the additional constant unit in the hidden layer.

layer_h = [ones(m,1) layer_h]; %add column of one's to hidden layer


%===Step 4
%Just like in step 2 you've got an m_examples x hu_hidden_units (+1) matrix
%and a ou_output_units x hu_hidden_units(+1) theta matrix to give theta
%values for all the connections.
%Do a matrix multiplication to get what the activation of the output
%neruons would be. The output shold be a matix of m_examples x
%ou_output_units. i.e. a value of each output neuron for every example.

layer_o = layer_h*Theta2'; %calculate output layer

%===Step 5
%For each row, i.e. each example, find out which neuron has the greatest
%activity. Thus giving a predicition of what each example is (i.e. what is
%the number shown by each picture.

[~, p] = max(layer_o,[],2);


% =========================================================================


end
