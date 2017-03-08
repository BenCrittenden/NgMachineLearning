function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%


%===Step1
%looping over the values of lambda in lambda_vec calcualte the cost
%function for each value of lambda in order to choose a good one. Note that
%lambda shoud be used for training, but NOT for testing (i.e. set it to
%lambda = 0).


for  i = 1:length(lambda_vec)
    
    lambda = lambda_vec(i);

    %minimise the cost function using this training data and this 
    %particular value of lambda. Thus get the theta estimates.
    [theta] = trainLinearReg(X,y,lambda);
    
    %Now use these theta estimates to calculate the training cost and the
    %cross validation cost (unlike in the training, during testing lambda 
    %should be set to 0, so that there is no regularization term)
    
    error_train(i) = linearRegCostFunction(X,y,theta,0);
    
    error_val(i) = linearRegCostFunction(Xval,yval,theta,0);
            
end


% =========================================================================

end
