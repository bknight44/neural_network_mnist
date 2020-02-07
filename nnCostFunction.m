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
K = num_labels;
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
saveY = y; %save the original y vector befor it gets changed

for i = 1:size(y,1)
temp(i,:) = [1:K];
endfor
y = y==temp; %changed the numbers into values like [0,0,1,0,0,0,0,0,0,0]
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%



    X = [ones(m, 1) X];
    layer_one = X * Theta1';
    % Add ones to the layer_one data matrix

    layer_one = sigmoid(layer_one);
    layer_one = [ones(m, 1) layer_one];
    output_layer = sigmoid(layer_one * Theta2');

 
    sig = output_layer;
    logs1 = log(sig);
    logs2 = log(1-(sig));
    temp2 = 0;
    
for i = 1:m
  %{
   for j = 1:K
         temp2 = temp2 + (-y(i,j) * logs1(i,j) - (1-y(i,j)) * logs2(i,j));
     
   endfor
   
   %}
   temp2 = temp2 + (-y(i,:) * logs1(i,:)' - (1-y(i,:)) * logs2(i,:)');
endfor
%}

     %temp2 = -y * logs1' - (1-y) * logs2';
J = (1/m) * temp2;

regularization = 0; %this will hold the sum of the theta^2


regularization = regularization + sum(sum(Theta1(:,2:end) .^2)) + sum(sum(Theta2(:,2:end) .^2));
regularization = (lambda /(2*m))*regularization;

J = J + regularization;

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


%X = [ones(m, 1) X]; %we already added this extra column before
%{
for i = 1:m
    %first feed forward one set
    
    layer_one = sigmoid(X(i,:) * Theta1'); %this will result in a 1X25 Matrix
    % Add ones to the layer_one data matrix
    layer_one = [ones(1, 1) layer_one]; %use 1 instead of m since we are stepping through
    output(i,:) = sigmoid(layer_one * Theta2');%should reult in a 1X10
    
    a2 = X(i,:) * Theta1';
    a3 = output(i,:);
    
    d3 = output(i,:) - y(i,:);
    d2 = Theta2' * d3' .* [ones(1,1) sigmoidGradient(X(i,:) * Theta1')]'; % z = (X(i,:) * Theta1')
    %{
    size(d2)
    size(d3)
    size(Theta2')
    size(sigmoidGradient(X(i,:) * Theta1'))
    pause;
    %}
    Theta1_grad(i,2:end) = Theta1_grad(i,2:end) + d3 * a2';
    Theta2_grad(i,2:end) = Theta2_grad(i,2:end) + d2 * a3';
    
endfor
%}
%vecotorized version
    %X = [ones(m, 1) X];
    layer_one = X * Theta1';
    % Add ones to the layer_one data matrix

    layer_one = sigmoid(layer_one);
    
    layer_one = [ones(m, 1) layer_one];
    a2 = layer_one;
    output_layer = sigmoid(layer_one * Theta2');

    a1 = X;

    z2 = X * Theta1';
    z3 = output_layer;

    d3 = output_layer - y;
    %size(d3)
   % size(a1)
   % size(a2)
   % size(X)
   % size(Theta1)
    %size(Theta2)
   %size(sigmoidGradient(X * Theta1')')
   %size(Theta2(:,2:end)' * d3')

    d2 = (Theta2(:,2:end)' * d3')' .* sigmoidGradient(z2); % z = (X(i,:) * Theta1')
 
    Theta1_grad = Theta1_grad + d2' * a1; %(:,2:end)

    Theta2_grad = Theta2_grad + d3' * a2;

    Theta1_grad = Theta1_grad .* (1/m);
    Theta2_grad = Theta2_grad .* (1/m);
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,2:end) = Theta1_grad(:,2:end)+ (lambda/m) .* Theta1(:,2:end);

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) .* Theta2(:,2:end);









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
