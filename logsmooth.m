function [y]=logsmooth(x, eps)
% Logsmooth (softplus) function g_{eps} applied pointwise to x
y = eps*log(1 + exp(x/eps));
y(isnan(y)|isinf(y)) = x(isnan(y)|isinf(y));
end
