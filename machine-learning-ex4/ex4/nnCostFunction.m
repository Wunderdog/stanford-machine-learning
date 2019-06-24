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

%{
Variables in the current scope:

   Attr Name                   Size                     Bytes  Class
   ==== ====                   ====                     =====  ===== 
        Theta1                25x401                    80200  double
        Theta2                10x26                      2080  double
        X                   5000x400                 16000000  double
        ans                    1x1                          8  double
        hidden_layer_size      1x1                          8  double
        input_layer_size       1x1                          8  double
        m                      1x1                          8  double
        nn_params          10285x1                      82280  double
        num_labels             1x1                          8  double
        y                   5000x1                      40000  double
%}

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); % number of rows (images)

alpha_one = [ones(m,1), X]; % 5000x401

alpha_two = [ones(m,1), sigmoid(alpha_one * Theta1')]; % 5000x26

alpha_three = sigmoid(alpha_two * Theta2'); % 5000x10
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%{
L  = total number of layers in the network
sl = number of units (not counting bias unit) in layer l
K = number of output units/classes
%}

K = num_labels;

% h_Theta = sigmoid(X)
h_Theta = alpha_three;

y_matrix = (1:num_labels)==y;

size(y_matrix);
size(h_Theta);

% regularization_term = (lambda/(2*m))*(sum(sum(Theta1(1:end).^2))+sum(sum(Theta2(1:end).^2)));
regularization_term = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))



% J=(1/(m))*sum(-y_matrix'*log(h_Theta)-(1-y_matrix)'*log(1-h_Theta));

% J = (-1/m)*sum((y_matrix'*log(h_Theta)) + (1-y_matrix)'*log(1-h_Theta));
% J = (-1/m) * sum(sum((y_matrix.*log(h_Theta))+((1-y_matrix).*log(1-h_Theta))));  %scalar

J = (-1/m) * sum(sum((y_matrix.*log(h_Theta))+((1-y_matrix).*log(1-h_Theta)))) + regularization_term;

% J


delta_3 = alpha_three - y_matrix; % 5000x10 -> difference in hypothesis and actual output
% delta_2 = (Theta2' * delta_three) .* (1 - alpha_2)
delta_2 = (delta_3 * Theta2) .* [ones(size(ones(size(alpha_two(:,2:end)), 1))) sigmoidGradient(alpha_one * Theta1')]; % 5000x26
delta_2 = delta_2(:,2:end); % 5000x25
% delta_1 = (Theta1' * delta_two) .* (1 - )

Theta1_grad = (1/m) * (delta_2' * alpha_one); % 25x401
Theta2_grad = (1/m) * (delta_3' * alpha_two); % 10x26

Theta1_grad_reg_term = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; % 25x401
Theta2_grad_reg_term = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; % 10x26

% Add regularization term to Theta gradients
Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
Theta2_grad = Theta2_grad + Theta2_grad_reg_term;



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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
