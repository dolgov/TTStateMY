function [Hvec] = mean_field_hess_vi(model,u,tau,d, vec)
global Nsolves
% Hessian*vec of the state loss with E[grad_y' * My * grad_y] replaced by anchor at xi=0
[~,Jac] = solve_y_vi(model,u,zeros(1,d),tau);
Hvec = (-model.B) * vec;
Hvec = Jac \ Hvec;
Hvec = model.My * Hvec;
Hvec = (Jac') \ Hvec;
Nsolves = Nsolves + 2;
Hvec = (-model.B') * Hvec;
end
