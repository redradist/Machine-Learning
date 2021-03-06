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
M = size(X, 1);
output_layer_size = size(y, 2);

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
temp_output = zeros(M, num_labels);
for i = 1:M,
    temp_output(i, y(i)) = 1;
end;
y = temp_output;

% -------------------------------------------------------------
J0 = 0;
for m = 1:M,
    a1 = [1 X(m,:)];
    z2 = Theta1 * a1';
    a2 = [1 sigmoid(z2')];
    z3 = Theta2 * a2';
    a3 = sigmoid(z3');
    for k = 1:num_labels,
        J0 -= y(m, k)*log(a3(k)) + (1-y(m, k))*log(1-a3(k));
    end;
end;
J0 = J0 / M;
% -------------------------------------------------------------

% -------------------------------------------------------------
Sl = [input_layer_size;hidden_layer_size;num_labels];
J1 = 0;
for l = 1:size(Sl)-1,
    for i = 1:Sl(l+1),
        for j = 1:Sl(l),
            % For ignore bias weight I have started counting from j+1 (ignoring j=1 as bias weight)
            if l == 1,
                J1 += Theta1(i, j+1) ^ 2;
            else
                J1 += Theta2(i, j+1) ^ 2;
            end;
        end;
    end;
end;
J1 = (lambda / (2 * M)) * J1;
% -------------------------------------------------------------

J = J0 + J1;

% -------------------------------------------------------------
% Part 2 Backpropagation algorithm
Theta1_grad;
Theta2_grad;
for m = 1:M,
     % ---------------------------------------------------------
     a1 = [1 X(m,:)];
     z2 = Theta1 * a1';
     a2 = [1 sigmoid(z2')];
     z3 = Theta2 * a2';
     a3 = sigmoid(z3');
     % ---------------------------------------------------------
     q3 = a3 - y(m, :);
     Theta2_grad .+= (1/M) .* (q3' * a2);

     theta2 = Theta2;
     theta2(:, 1) = [];
     q2 = (theta2' * q3') .* sigmoidGradient(z2);
     Theta1_grad .+= (1/M) .* (q2 * a1);
 end;
% -------------------------------------------------------------

theta2_reg = Theta2;
theta2_reg(:, 1) = 0;
Theta2_grad .+= ((lambda/M) * theta2_reg);
theta1_reg = Theta1;
theta1_reg(:, 1) = 0;
Theta1_grad .+= ((lambda/M) * theta1_reg);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
