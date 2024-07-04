% Variational Inequality example

check_tt;
ny = parse_parameter('(Odd) Number of spatial grid points ny', 31);
nxi = parse_parameter('Number of grid points for each random variable nxi', 5);
eps = parse_parameter('Smoothing parameter eps', 1e-6);
tol = parse_parameter('Approximation and stopping tolerance', 1e-3);
alpha = parse_parameter('Control regularization parameter alpha', 1);
d = parse_parameter('Number of random variables d', 5);

% "Mean-zero" (affine) field
[xi,w] = lgwt(nxi, -1, 1);   w = w/2;

Xi_grid = tt_meshgrid_vert(tt_tensor(xi), d);
W = tt_tensor(repmat({w},d,1));

% Set up model from [https://arxiv.org/pdf/2210.03425.pdf, Section 5.1]
model = struct();
model.A = spdiags(ones(ny,1)*[-1,2,-1], -1:1, ny, ny) * (ny+1)^2;
model.A = kron(model.A, speye(ny)) + kron(speye(ny), model.A);
model.B = speye(ny^2);
model.My = speye(ny^2)/((ny+1)^2);
model.Mu = speye(ny^2)/((ny+1)^2);

[X1,X2] = meshgrid((1/(ny+1)):(1/(ny+1)):(1-1/(ny+1)));
model.yhat = 160.*(X1.^3 - X1.^2 + 0.25*X1).*(X2.^3 - X2.^2 + 0.25*X2) .* double((X1<0.5) & (X2<0.5));
model.zeta = max(-2*abs(X1-0.8) - 2*abs(X1.*X2-0.3) + 0.5, 0);
model.f0 = reshape(model.A*model.yhat(:), ny, ny) - model.yhat - model.zeta;
model.f0 = model.f0(:);
model.yd = model.yhat + model.zeta + reshape(model.A*model.yhat(:), ny, ny);
model.yd = -model.yd(:);  % for constraint y<=0

% Assemble the "Mean-zero" (affine) field
jind = repmat((1:d)', 1, d);
kind = repmat(1:d, d, 1);
lambda = 0.01*exp(-(pi/4)*(jind.^2 + kind.^2));
lambda = lambda(:);
[~, prm] = sort(lambda, 'descend');
lambda = reshape(lambda(prm(1:d)), 1, 1, d);
jind = reshape(jind(prm(1:d)), 1, 1, d);
kind = reshape(kind(prm(1:d)), 1, 1, d);
model.phil = 2.*sqrt(lambda) .* cos(jind.*pi.*X2).*cos(kind.*pi.*X1) .* double(X1<0.5);
model.phil = reshape(model.phil, ny^2, d);



% Total number of PDE solves
global Nsolves Newton_inner
Nsolves = 0;
Newton_inner = [0 0];  % #newton_calls, #total_iterations

% Gauss-Newton for the total cost minimization
Jgrad_tt = 4;
u = zeros(ny^2,1);
for iter=1:100
    Jgrad_tt = amen_cross_s(Xi_grid, @(xi)solve_adj_vi(model,u,xi,eps), tol, 'exitdir', -1, 'kickrank', 0, 'y0', Jgrad_tt, 'nswp', 2);
    Jgrad = dot(Jgrad_tt, W);
    Jgrad = Jgrad + alpha*(model.Mu*u);
    
    du = pcg(@(v)mean_field_hess_vi(model,u,eps,d, v) +  alpha*(model.Mu*v), Jgrad, tol*0.1, 200);
    u = u - du;
    fprintf('iter=%d, |du|=%g\n', iter, norm(du,'fro'));
    if (norm(du,'fro') < tol * norm(u,'fro'))
        break;
    end
end

Nsolves
% Average number of linear PDE solves
NLinPDE = Nsolves / (Newton_inner(2)/Newton_inner(1))

Jtt = amen_cross_s(Xi_grid, @(xi)state_cost_vi(model,u,xi,eps), 1e-8);
J = dot(Jtt, W) + (alpha/2) * u'*(model.Mu*u);
fprintf('J = %1.10g\n', J);

Ytt = amen_cross_s(Xi_grid, @(xi)solve_y_vi(model,u,xi,eps), tol, 'exitdir', -1, 'kickrank', 0);
ymean = dot(Ytt,W);
Ysample = tt_sample_lagr(Ytt, repmat({xi},d,1), rand(1000,d)*2-1);
yvar = mean((Ysample - reshape(ymean,1,[])).^2)';

x = (1:ny)'/(ny+1);
figure(1);
surf(x,x,reshape(-ymean,ny,ny), 'EdgeColor', 'None'); view(40,30); colorbar('southoutside');
title('mean Y')

figure(2);
surf(x,x,reshape(yvar,ny,ny), 'EdgeColor', 'None'); view(40,30); colorbar('southoutside');
title('variance Y')

figure(3);
surf(x,x,reshape(u,ny,ny), 'EdgeColor', 'None'); view(40,30); colorbar('southoutside');
title('U')

% Cost from Ytt
Japp = tt_reshape(Ytt, [ny^2; W.n]);
Japp = Japp - tkron(tt_tensor(model.yd), tt_ones(W.n));
sqrtMass = repmat({spdiags(sqrt(w), 0, nxi, nxi)},d,1);
sqrtMass = [{chol(model.My)}; sqrtMass];
Japp = amen_mm(sqrtMass, Japp, 1e-8, 'x0', Japp);
Japp = 0.5*norm(Japp)^2 + (alpha/2) * u'*(model.Mu*u);
fprintf('J(approx y) = %1.10g\n', Japp);

% Dual solution
% Ptt = amen_cross_s(Xi_grid, @(xi)logsmooth(solve_y(model,u,xi,tau), tau)/tau, tol, 'exitdir', -1, 'kickrank', 0);
Ptt = amen_cross_s({Ytt}, @(y)logsmooth(y, eps)/eps, tol, 'exitdir', -1, 'kickrank', 0);
pmean = dot(Ptt,W);

figure(4);
surf(x,x,reshape(pmean,ny,ny), 'EdgeColor', 'None'); view(40,30); colorbar('southoutside');
title('mean P (dual state)')
