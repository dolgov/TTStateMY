function [grad_ju, grad_total, grad_MY_y,grad_MY_u,MY_grad_MY,ju_tt,grad_ju_tt, ttimes_grad] = cost_grad(model, Xi_grid, W, u, tol, alpha, gamma, eps, y_tt,grad_y,MY, grad_MY_y,grad_MY_u,MY_grad_MY,ju_tt,grad_ju_tt, j_grad_fun, grad_my_fun)
% Gradient of the cost
[ny,nu] = size(model.B);

tic;
% Gradient of model cost
grad_ju_tt = amen_cross_s(Xi_grid, @(xi)j_grad_fun(model, u, xi, 1), tol, 'exitdir', -1, 'y0', grad_ju_tt);
grad_ju = dot(grad_ju_tt,W) + alpha*model.Mu*u;

% Gradient of the MY term in state
grad_MY_y = amen_cross_s({y_tt, model.Ymax}, @(U)grad_logsmooth(U(:,1)-U(:,2), eps, 0), tol, 'y0', grad_MY_y);

% Gradient of the MY term in control
if isempty(grad_y) || isempty(MY_grad_MY)
    xi_grid = cell(numel(Xi_grid),1);
    for i=1:numel(Xi_grid) 
        xi_grid{i} = Xi_grid{i}{i}(:); 
    end
    grad_MY_u = amen_cross_s(Xi_grid, @(xi)grad_my_fun(model, u, xi, xi_grid, eps), tol, 'y0', grad_MY_u, 'exitdir', -1);
else
    % Pointwise product of (spatial mass * MY) and grad_MY
    MY1 = MY{1};
    MY1 = reshape(MY1, ny, []);
    MY1 = model.My*MY1;
    MY1 = reshape(MY1, 1, ny, []);
    MMY = MY;
    MMY{1} = MY1;
    MY_grad_MY = amen_cross_s({grad_MY_y, MMY}, @(x)x(:,1).*x(:,2), tol, 'y0', MY_grad_MY);
    grad_MY_u = amen_mm(grad_y, MY_grad_MY, tol, 'x0', grad_MY_u);
end

grad_total = grad_ju + dot(tt_reshape(grad_MY_u, W.n, [], nu, 1), W) * gamma;
ttimes_grad = toc;
end
