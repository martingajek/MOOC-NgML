function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


theta_redux = theta(3:end);
J = 1/length(y)*(-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)) + 0.5*lambda*theta_redux'*theta_redux) ;


grad0 = 1/length(y)*X(:,1)'*(sigmoid(X*theta)-y);
gradJ = 1/length(y)*( X(:,2:end)'*(sigmoid(X*theta)-y) + 0.5*lambda*theta(2:end) );
grad = [grad0;gradJ];




% =============================================================

end
