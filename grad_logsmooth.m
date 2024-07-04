function [y]=grad_logsmooth(x, eps, delta)
% Derivative of the logsmooth (softplus) function g_{eps} with negative
% offset g(-inf) = -delta pointwise at x
z = x/eps;
y = (1+delta)./(exp(-z)+1) - delta;
end
