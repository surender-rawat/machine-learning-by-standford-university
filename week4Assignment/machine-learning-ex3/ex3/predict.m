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




 % p = sigmoid(Z3) = sigmoid(t2*a2) = sigmoid(t2*(t1*X))

X = [ones(size(X,1),1) X];
a2 = sigmoid( X * Theta1') ;
A2 = [ones(size(a2,1),1) a2];
a3 = sigmoid(A2 * Theta2');
predict  = a3;
[maxele, index] = max(predict, [], 2);
 p = index;







% =========================================================================


end
