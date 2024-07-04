function [J] = state_cost_vi(model,u,xi,tau)
[y] = solve_y_vi(model,u,xi,tau);
y = y';
misfit = model.My*(y - model.yd);
J = zeros(size(xi,1), 1);
for j=1:size(y,2)
    J(j) = (y(:,j) - model.yd)' * misfit(:,j);
end
J = J*0.5;
end
