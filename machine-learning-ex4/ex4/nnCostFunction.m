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





% Generate Y in format [0 1 0 0 0 0 0 0 0 0; ....]
Y = zeros(m,num_labels);
for i = 1:m,
	Y(i,y(i)) = 1;
end


%  output_layer_costs(example,label)
output_layer_costs = zeros(m,num_labels); % cost function for every example in output layer
delta_3 = zeros(m,num_labels); % delta "error" for every example in output layer
delta_2 = zeros(m,hidden_layer_size); % delta "error" for every example in hidden layer
DELTA_1 = zeros(size(Theta1)); % DELTA acumulative error for layer 1
DELTA_2 = zeros(size(Theta2)); % DELTA acumulative error for layer 2
for i = 1:m,
	% FORWARD PROPAGATION for example i
	% Layer 1
	% size 401x1
	a1 = [1 X(i,:)];

	% Layer 2
	% size 1x26
	a2 = sigmoid(a1 * Theta1');

	% hidden_layer_costs(i) = sum(-(1/m)*sum(Y(i,:).*log(a2) + (1-Y(i,:)).*log(1-a2)));

	% Layer 3
	a2 = [1 a2];
	a3 = sigmoid(a2 * Theta2');	% a3 prediction of example i
	
	% size(Y(i,:)) = 1x10
	% size(a3) = 1x10

	% l = log(a3);
	% p = Y(i,:);


	% a = (p.*l);
	% % size(l)

	% b = (1-Y(i,:)).*log(1-a3);

	output_layer_costs(i,:) = -(1/m)*(Y(i,:).*log(a3) + (1-Y(i,:)).*log(1-a3));
	
	% BACK PROPAGATION for example i

	% size(delta_3(i,:)) 10x1
	delta_3(i,:) = (a3 - Y(i,:))';
	
	% size(((delta_3(i,:)*Theta2).*(a2.*(1-a2)))') 26x1
	delta_2_temp = ((delta_3(i,:)*Theta2).*(a2.*(1-a2)))';

	% remove bias element
	delta_2_temp(1,:) = [];

	% size(delta_2(i,:)) 25x1
	delta_2(i,:) = delta_2_temp;

	
	DELTA_1 = DELTA_1 + delta_2(i,:)'*a1;
	DELTA_2 = DELTA_2 + delta_3(i,:)'*a2;

end



J = sum(sum(output_layer_costs));


% Regularization
theta1_temp = Theta1;
theta1_temp(:,1) = zeros(hidden_layer_size,1);
theta2_temp = Theta2;
theta2_temp(:,1) = zeros(num_labels,1);;



J = J + sum((lambda/(2*m))*sum(theta1_temp.^2)) + sum((lambda/(2*m))*sum(theta2_temp.^2));



Theta1_grad = (1/m)*DELTA_1 + (lambda/m)*theta1_temp;
Theta2_grad = (1/m)*DELTA_2 + (lambda/m)*theta2_temp;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
