function [ju, j_total, y_tt, MY, MY2, ju_tt, ttimes_y, ttimes_cost] = cost_fun(model, Xi_grid, W, u, tol, alpha, gamma, eps, y_tt, MY, MY2, ju_tt, solve_fun, j_grad_fun)
% Generic cost function
ny = size(model.B, 1);

% Assemble the full solution
tic;
y_tt = amen_cross_s(Xi_grid, @(xi)solve_fun(model,u,xi,true), tol, 'exitdir', -1, 'y0', tt_reshape(y_tt, W.n, tol, ny,1));
y_tt = tt_reshape(y_tt, [ny; W.n], tol);
ttimes_y = toc;

tic;
% Model cost function
ju_tt = amen_cross_s(Xi_grid, @(xi)j_grad_fun(model, u, xi, 0), tol, 'y0', ju_tt);
ju = dot(ju_tt,W) + dot(u, model.Mu*u) * alpha/2;

% MY term without the norm
MY = amen_cross_s({y_tt, model.Ymax}, @(U)logsmooth(U(:,1)-U(:,2), eps), tol, 'y0', MY);
% Square of this for cost & Armijo
% First, mult with mass in space
MY1 = MY{1};
MY1 = reshape(MY1, ny, []);
MY1 = model.My*MY1;
MY1 = reshape(MY1, 1, ny, []);
MMY = MY;
MMY{1} = MY1;
MY2 = amen_cross_s({MY, MMY, tkron(tt_ones(ny), W)}, @(x)prod(x,2), tol, 'y0', MY2);
j_total = ju + dot(tt_ones(MY2.n), MY2) * gamma/2;

ttimes_cost = toc;
end

