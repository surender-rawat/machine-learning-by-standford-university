function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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

	%temp1 = sum((theta(1) + theta(2) .* X(:,2)) - y);
	%temp2 = sum(((theta(1) + theta(2) .* X(:,2)) - y) .* X(:,2));

	%theta(1) = theta(1) - (alpha/m)* temp1;
	%theta(2) = theta(2) - (alpha/m)* temp2;
	
	error = (X * theta) - y;
	temp1 = theta(1) - (alpha/m)* sum(error .* X(:,1));
	temp2 = theta(2) - (alpha/m)* sum(error .* X(:,2));

	theta = [temp1; temp2];

    % ============================================================

    % Save the cost J in every iteration   
    cost =  computeCost(X, y, theta);
    fprintf('\n Iteration = %f |  Cost computed = %f | theta[ %f ; %f]',iter, cost,theta(1),theta(2));
    J_history(iter) = cost;

	
end

end
