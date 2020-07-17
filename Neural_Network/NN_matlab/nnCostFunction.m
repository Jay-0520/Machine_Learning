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

K = num_labels;
Y = eye(K)(y, :);
% A lousy way for making Y
% labels=[1;2;3;4;5;6;7;8;9;10];%y_ = repmat(y, 1, 10);                                                % labels_=repmat(labels', m, 1);%y_vec = (y_ == labels_);


% Part 1 - Feedforward the neural network and return the cost 
a1 = [ones(size(X, 1), 1), X];

a2 = sigmoid(a1*Theta1');
a2 = [ones(size(a2, 1), 1), a2];
a3 = sigmoid(a2*Theta2'); % a3 is a 5000 * 10 matrix


cost = sum((-Y .* log(a3)) - ((1 - Y) .* log(1 - a3)), 2);
J = (1 / m) * sum(cost);

% add regularization term into the cost function
Theta1NB=Theta1(:,2:end);
Theta2NB=Theta2(:,2:end);

TheSum1 = sum(sum(Theta1NB.^2));
TheSum2 = sum(sum(Theta2NB.^2));

J = J + (lambda/(2*m))*(TheSum1 + TheSum2);

% -------------------------------------------------------------
% =========================================================================
% Part 2 - Implement the backpropagation algorithm to compute the gradients

%Theta1_grad = Theta1_grad(:, 2:end)
%Theta2_grad = Theta2_grad(:, 2:end)

DEL1 = 0; 
DEL2 = 0;
for t = 1:m
    
    % step 0 - forward propagation
    a1 = X(t,:); a1 = [ones(size(a1, 1), 1) a1]; 
    z2 = a1*Theta1'; a2 = sigmoid(z2); a2 = [ones(size(a2, 1), 1) a2]; 
    z3 = a2*Theta2'; a3 = sigmoid(z3);
   
    % step 1 - compute delta at the output layer
    Y3 = Y(t,:); % 1 by 10 matrix
    d3 = (a3 - Y3);
   
    % step 2 - compute delta at the hidden layer
    d2 = d3*Theta2NB .* sigmoidGradient(z2); % Theta2'*d3' is a 26 by 1 matrix
    %d2 = d2(2:end); % we need to remove the bias
   
    % step 3 - accumulate
    DEL2 += d3' * a2;
    DEL1 += d2' * a1;

endfor

Theta1_grad = (1/m) * DEL1;
Theta2_grad = (1/m) * DEL2; 

% add regularization into the gradient

Theta1_grad(:, 2:end) += ((lambda / m) * Theta1NB);
Theta2_grad(:, 2:end) += ((lambda / m) * Theta2NB);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


