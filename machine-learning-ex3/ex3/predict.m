function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

alpha_one = [ones(m, 1) X]; % 5000 x 401
% Theta1 has size 25 x 401
z_two = alpha_one * Theta1'; % 5000 x 25

alpha_two = sigmoid(z_two); % 5000 x 25
alpha_two = [ones(m, 1) alpha_two]; % Add ones column for size 5000 x 26

% Theta2 has size 10 * 26
z_three = alpha_two * Theta2'; % 5000 x 10

alpha_three = sigmoid(z_three);

[prob p] = max(alpha_three, [], 2);

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









% =========================================================================


end
