function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit

%You should use epsilon = 0.12 This range of values ensures that the parameters are kept small and makes the learning more e?cient. 


%{
One e?ective strategy for choosing epsilon is to base it on the number of units in the network. 
A good choice of epsilon is epsilon = v6) / vLin+Lout) , where Lin = sl and Lout = sl+1 are the number of 
units in the layers adjacent to T(l).
%}

epsilon = 0.12;

W = (rand(L_out, 1 + L_in) * 2 * epsilon) - epsilon;






% =========================================================================

end
