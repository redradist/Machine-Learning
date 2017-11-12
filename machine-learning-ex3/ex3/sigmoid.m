function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

sizes = size(g);
for i = 1:sizes(1,1),
	for k = 1:sizes(1,2),
		g(i,k) = 1 / (1 + exp(-z(i,k)));
	end;
end;

% =============================================================

end
