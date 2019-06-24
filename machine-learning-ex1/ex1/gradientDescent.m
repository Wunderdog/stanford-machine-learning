function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_1 = theta(1);
theta_2 = theta(2);

equation = " \
theta_1 = theta_1-(alpha/m)*(sum(((X*theta)-y).*X(:,1)))\n \
theta_2 = theta_2-(alpha/m)*(sum(((X*theta)-y).*X(:,2)))\n\n"


fprintf("<<< Gradient Descent >>>\n")

whos -blank

fprintf('Equations: \n%s', equation);
figure(1);
    clf(1);
    plot(X(:,2), y, 'rx', 'MarkerSize', 10);
    xlabel('population');
    ylabel('revenue');
    hold on
    plot(X(:,2), theta_2*X(:,2)+theta_1, 'b');
    legend('Training data', 'Linear regression');
    hold off
    pause(0.05);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    

    % theta1-(alpha/m)*(sum(((X*theta)-y).*X(:,1)));
    % theta(1) = theta(1) - sum((theta*X) - y)*(alpha/m)
    % theta(2) = theta(2) - sum((theta*X) - y)*(alpha/m)
 

    theta_1 = theta_1-(alpha/m)*(sum(((X*theta)-y).*X(:,1)));
    theta_2 = theta_2-(alpha/m)*(sum(((X*theta)-y).*X(:,2)));
    
    if((theta_1 - theta(1))^2 >= 0.00005 || (theta_2 - theta(2))^2 >= 0.00005)
    figure(1);
    clf(1);
    plot(X(:,2), y, 'rx', 'MarkerSize', 10);
    xlabel('population');
    ylabel('revenue');
    hold on
    plot(X(:,2), theta_2*X(:,2)+theta_1, 'b');
    legend('Training data', 'Linear regression');
    hold off
    pause(0.05);
    % fprintf('\niter: %f | diff_1: %f\n diff_2: %f\n theta2: %f\n', iter, (theta_1 - theta(1))^2, (theta_2 - theta(2))^2)
    endif;
    theta(1) = theta_1;
    theta(2) = theta_2;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
