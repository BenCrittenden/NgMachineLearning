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

X = [ones(m,1) X];
y2 = zeros(length(y),10);
for t = 1:length(y)
    y2(t,y(t)) = 1;
end

Delta1 = 0;
Delta2 = 0;

%Part 1============================

%non-regularized cost function:
for x = 1:size(X,1)
    
    clear h1 h2 j
    for a = 1:size(Theta1,1)
        
        t1 = Theta1(a,:);
        z1(a) = t1*X(x,:)';
        h1(a) = sigmoid(z1(a));

    end
    
    h1 = [1 h1];
    
    for b = 1:size(Theta2,1)
        
        t2 = Theta2(b,:);
        z2(b) = t2*h1';
        h2(b) = sigmoid(z2(b));
 
    end
    
    y3 = y2(x,:);
    
    %calculate error in output later
    
    delta1 = (h2 - y3)';
    
    %calcualter error in layer 2
    
    delta2  = (Theta2(:,2:end)'*delta1).*sigmoidGradient(z1)';
    
    %calculate delta acumulator for the different layers
    
    Dlta1 = Delta1 + (delta2'*h1(2:end)');
    Dlta2 = Delta2 + (delta1'*h2');
    
    
    for k = 1:size(y3,2)
        
        j1 = -y3(k).*log(h2(k) );
        j2 = (1-y3(k)).*log(1-h2(k));
        
        j(k) = j1-j2;
  
    end
    
    K(x) = sum(j);
    
end

J = (1/size(X,1)).*sum(K);

%get the unregularized gradient

D1 = Dlta1./m;
D2 = Dlta2./m;

grad = [D1 D2];


%now add the regularization bit

for ji1 = 1:size(Theta1,1) 
    
    clear r1k
    for k1 = 2:size(Theta1,2)
        r1k(k1-1) = Theta1(ji1,k1).^2;
    end
    
    rk1(ji1) = sum(r1k);
    
end

r1 = sum(rk1);


for ji2 = 1:size(Theta2,1) 
    
    clear r2k
    for k2 = 2:size(Theta2,2)
        r2k(k2-1) = Theta2(ji2,k2).^2;
    end
    
    rk2(ji2) = sum(r2k);
    
end

r2 = sum(rk2);


lambda = 1;
R = (lambda/(2*m)).*(r1+r2);



%combine regularization term with normal cost function

J = J + R;




%Part 2============================





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
