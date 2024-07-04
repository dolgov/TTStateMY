function [gradJ] = solve_adj_vi(model,u,xi,tau)
global Nsolves
% Adjoint solution and grad_u of the state loss
[y,Jac] = solve_y_vi(model,u,xi,tau);
y = y';
misfit = model.My*(y - model.yd);
yadj = (Jac') \ misfit(:);
Nsolves = Nsolves + size(y,2);
yadj = reshape(yadj, size(y));
gradJ = (-model.B') * yadj;

gradJ = gradJ';  % For multifun cross I x d format
end
