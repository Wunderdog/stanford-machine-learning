function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

temp = theta;
h_theta = sigmoid(X * theta);
bias_term = (lambda/(2*m))*sum((theta.^2)(2:end));

% J = (1/m)*sum(((-y).*log(htheta)) - ((ones(size(y))-y).*log(1-htheta)));
J=(1/m)*(-y'*log(h_theta)-(1-y)'*log(1-h_theta))+bias_term;

% theta
% theta(2:end)


bias_grad = (lambda/m)*theta;

% bg_size = size(bias_grad)

% X
% X(2:end, :)
% y(2:end)

% y(2:end).*X(2:end, :) + bias_grad



% grad = (1/m)*sum((h_theta-y).*X)'
% same as
grad = (1/m)*((h_theta-y)'*X)';


grad(2:end) = grad(2:end) + bias_grad(2:end);

% grad_2 = size

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

grad = grad(:);

end
