function [u, y_tt, ttranks_y, ttranks_grad_MY, ttimes_y, ttimes_cost, ttimes_grad] = moreau_yosida_fixpoint(model, xi_grid, W, alpha, eps0, gamma_final, tol, maxiter, umin, umax, solve_fun, j_grad_fun, grad_y_fun, grad_my_fun, plot_fun)
% Moreau-Yosida solver
% J = j(u) + gamma/2 * E || G_eps(y-Ymax) ||^2
% Using anchored point hessian
% Inputs:
%   model: any structure to be passed to forward solution functions 
%   xi_grid: vector of parameter grid
%   W: tt_tensor of quadrature weights for all parameters
%   alpha: control regularisation parameter
%   eps0: initial smoothing eps
%   gamma_final: final Moreau-Yosida gamma
%   tol: TT and stopping tolerance
%   maxiter: maximum number of iterations
%   umin: lower bound for control
%   umax: upper bound for control
%   solve_fun: function handle to compute the state y
%   grad_y_fun: function handle to compute dy/du or []
%   j_grad_fun: function handle to compute the state cost or its gradient
%   grad_my_fun: function handle to compute grad (MY term) or []
%   plot_fun: function handle to plot something each iteration or []
% Outputs:
%   u: final control
%   y_tt: final state
%   ttranks_y: vector of max TT ranks of y in all iterations
%   ttranks_grad_MY: vector of max TT ranks of grad of MY term in all iterations
%   ttimes_y: vector of CPU times of solving for y in all iterations
%   ttimes_cost: vector of CPU times of computing cost in all iterations
%   ttimes_grad: vector of CPU times of computing gradient in all iterations

d = W.d;
[ny,nu] = size(model.B);

Xi_grid = tt_meshgrid_vert(cellfun(@(xi)tt_tensor(xi), xi_grid, 'uni', 0));

u = 0*ones(nu,1); % Initial guess

xi_mean = zeros(1,d); % Mean parameter point for the Hessian
for j=1:d
    xi_mean(j) = dot(W, Xi_grid{j});
end

y_tt = tt_zeros([ny; W.n]); % Initial guesses - just initial TT ranks for random init
MY=4; MY2=4; grad_MY_y=4; grad_MY_u=4; MY_grad_MY=4; ju_tt=4; grad_ju_tt=4;

grad_y = [];
if (ny<1000) && (~isempty(grad_y_fun))
    % It will be faster to precompute grad y and use it to assemble the
    % gradient of the MY term
    grad_y_vec = amen_cross_s(Xi_grid, @(xi)grad_y_fun(model, xi), tol, 'exitdir', -1);
    grad_y = tt_reshape(grad_y_vec, [nu*ny; W.n]);
    grad_y = tt_matrix(grad_y, [nu; W.n], [ny; ones(d,1)]);
    grad_y = core2cell(grad_y);
    for k=2:d+1
        C = zeros(size(grad_y{k},1), W.n(k-1), W.n(k-1), size(grad_y{k},4));
        for j=1:W.n(k-1)
            C(:,j,j,:) = grad_y{k}(:,j,1,:);
        end
        grad_y{k} = C;
    end
    grad_y = cell2core(tt_matrix, grad_y);
    grad_MY_u=[];
end

gamma = 1; % Initial gamma
for i=1:maxiter
    eps = eps0/sqrt(gamma);
    
    [ju, j_total_old, y_tt, MY, MY2, ju_tt, ttimes_y(i), ttimes_cost(i)] = cost_fun(model, Xi_grid, W, u, tol, alpha, gamma, eps, y_tt, MY, MY2, ju_tt, solve_fun, j_grad_fun);
    [grad_ju, grad_total, grad_MY_y,grad_MY_u,MY_grad_MY,ju_tt,grad_ju_tt, ttimes_grad(j)] = cost_grad(model, Xi_grid, W, u, tol, alpha, gamma, eps, y_tt,grad_y,MY, grad_MY_y,grad_MY_u,MY_grad_MY,ju_tt,grad_ju_tt, j_grad_fun, grad_my_fun);

    ttranks_y(i) = max(y_tt.r);
    ttranks_grad_MY(i) = max([grad_MY_y.r; grad_MY_u.r]);

    % Anchor point
    grad_MY_y = tt_reshape(grad_MY_y, W.n, tol, ny, 1);
    xi_star = zeros(ny,d);    
    for j=1:d
        xi_star(:,j) = dot(grad_MY_y, W.*Xi_grid{j});
    end
    xi_star = xi_star ./ dot(grad_MY_y, W);
    xi_star = sum(model.My*xi_star) ./ sum(sum(model.My))
    grad_MY_y = tt_reshape(grad_MY_y, [ny; W.n], tol);
    
    % total Gauss-Newton Hessian * Vec function
    hfun = @(v)j_grad_fun(model, u, xi_mean, 2, v).' ...
         + gamma * j_grad_fun(model, u, xi_star, 2, v).' ...
         + alpha*(model.Mu*v);
    
    % Gauss-Newton step
    du = pcg(hfun, grad_total, tol*1e-3, 200);

    uold = u;
    step = 1;
    j_total_new = inf;
    ju = inf;
    % Line search
    while (step>5e-4) && (j_total_new > j_total_old + 1e-4*step*dot(du, grad_total))
        u = uold - step*du;
        % Control constraint
        u = max(u, umin);
        u = min(u, umax);
        
        [ju, j_total_new, y_tt, MY, MY2, ju_tt, ~, ~] = cost_fun(model, Xi_grid, W, u, tol, alpha, gamma, eps, y_tt, MY, MY2, ju_tt, solve_fun, j_grad_fun);
        step = step*0.5;
    end   
    step = step*2;
    
    fprintf('iter = %d, step = %g, |grad J_total| = %3.3e, j(u) = %g, eps=%g, gamma=%g, E[g(y-ymax)]=%g\n', i, step, norm(grad_total), ju, eps, gamma, (j_total_new - ju)*2/gamma);
    
    gamma = min(gamma*2, gamma_final); % Increase gamma

    if ~isempty(plot_fun)
        plot_fun(model,u,y_tt,W);
        drawnow;
    end

    if (step<1e-3) && (abs(gamma/gamma_final - 1)<1e-10)
        fprintf('Step size became <1e-3, stopping\n');
        break
    end
end

end


