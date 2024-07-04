% 2D Elliptic PDE with state constraint y<=0

check_tt;
ny = parse_parameter('(Odd) Number of spatial grid points ny', 63);
nxi = parse_parameter('Number of grid points for each random variable nxi', 17);
gamma = parse_parameter('Final Moreau-Yosida parameter gamma', 100);
eps0 = parse_parameter('Initial smoothing parameter eps0', 0.5);
tol = parse_parameter('Approximation and stopping tolerance', 1e-5);
alpha = parse_parameter('Control regularization parameter alpha', 1e-2);
umin = parse_parameter('Lower control constraint umin', -0.75);
umax = parse_parameter('Upper control constraint umax', 0.75);

[xi,w] = lgwt(nxi, -1, 1);   w = w/2;
d = 6; % #random variables, built-in for this model
xi_grid = repmat({xi}, d, 1);

model = struct();
model.A0 = spdiags(ones(ny+2,1)*[-1,2,-1]*(ny+1)^2, -1:1, ny+2, ny+2);
model.A0 = kron(model.A0, speye(ny+2)) + kron(speye(ny+2), model.A0);
model.My = speye(ny^2)/((ny+1)^2);
model.Mu = model.My;
model.B = speye(ny^2);
model.yd = -sin(50*(1:ny)'/(ny+1)/pi).*cos(50*(1:ny)/(ny+1)/pi);
model.yd = reshape(model.yd, ny^2, 1);
model.Ymax = 0*tt_ones([ny^2; nxi*ones(d,1)]);

W = mtkron(repmat({tt_tensor(w)}, 1, d));

model.Ymaxvec = tt_reshape(model.Ymax, W.n, tol, ny^2, 1);

[u, y_tt, ttranks_y, ttranks_grad_MY, ttimes_y, ttimes_cost, ttimes_grad] = moreau_yosida_fixpoint(model, xi_grid, W, alpha, eps0, gamma, tol, 100, umin, umax, @solve_fun_elliptic_2d, @j_grad_fun_elliptic_2d, [], @grad_my_elliptic_2d, @plot_elliptic_2d);

