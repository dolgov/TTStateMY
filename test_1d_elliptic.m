% 1D Elliptic PDE with state constraint y<=0

check_tt;
ny = parse_parameter('(Odd) Number of spatial grid points ny', 63);
nxi = parse_parameter('Number of grid points for each random variable nxi', 33);
gamma = parse_parameter('Final Moreau-Yosida parameter gamma', 300);
eps0 = parse_parameter('Initial smoothing parameter eps0', 0.5);
tol = parse_parameter('Approximation and stopping tolerance', 1e-7);
alpha = parse_parameter('Control regularization parameter alpha', 1e-2);
umin = parse_parameter('Lower control constraint umin', -0.75);
umax = parse_parameter('Upper control constraint umax', 0.75);

[xi,w] = lgwt(nxi, -1, 1);   w = w/2;
d = 4; % #random variables, built-in for this model
xi_grid = repmat({xi}, d, 1);

model = struct();
model.A0 = spdiags(ones(ny+2,1)*[-1,2,-1]*(ny+1)^2, -1:1, ny+2, ny+2);
model.My = speye(ny)/(ny+1);
model.Mu = model.My;
model.B = speye(ny);
model.yd = -sin(50*(1:ny)'/(ny+1)/pi); % Desired state
model.Ymax = 0*tt_ones([ny; nxi*ones(d,1)]); % -\psi in the paper since here we impose the upper bound

W = mtkron(repmat({tt_tensor(w)}, 1, d)); % Quadrature weights in all parameters

model.Ymaxvec = tt_reshape(model.Ymax, W.n, tol, ny, 1); % Needed for explicit computation of MY gradient

[u, y_tt, ttranks_y, ttranks_grad_MY, ttimes_y, ttimes_cost, ttimes_grad] = moreau_yosida_fixpoint(model, xi_grid, W, alpha, eps0, gamma, tol, 100, umin, umax, @solve_fun_elliptic_1d, @j_grad_fun_elliptic_1d, @grad_y_fun_elliptic_1d, @grad_my_elliptic_1d, @plot_elliptic_1d);

