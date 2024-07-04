% SEIR example with constrained R number
check_tt;

% Contact data
Cfixed(:,:,1) = load('SEIRData/Contact_work.txt');
Cfixed(:,:,2) = load('SEIRData/Contact_school.txt');
Cfixed(:,:,3) = load('SEIRData/Contact_other.txt');
Cfixed(:,:,4) = load('SEIRData/Contact_home.txt');

% initialize Susceptible with the total pop
S = load('SEIRData/pop.txt');

sigma = parse_parameter('Standard deviation of random variables sigma', 3e-2);
nxi = parse_parameter('Number of grid points in random variables', 3);
% random variables
[beta,wbeta] = lgwt(nxi, 0.13-sigma*0.03, 0.13+sigma*0.03);   wbeta = wbeta/sum(wbeta);
[dL,wdL] = lgwt(nxi, 1.57-sigma*0.42, 1.57+sigma*0.42);       wdL = wdL/sum(wdL); 
[dC,wdC] = lgwt(nxi, 2.12-sigma*0.80, 2.12+sigma*0.80);       wdC = wdC/sum(wdC);
[dR,wdR] = lgwt(nxi, 1.54-sigma*0.40, 1.54+sigma*0.40);       wdR = wdR/sum(wdR);
[dRC,wdRC] = lgwt(nxi, 12.08-sigma*1.51, 12.08+sigma*1.51);   wdRC = wdRC/sum(wdRC);
[dD,wdD] = lgwt(nxi, 5.54-sigma*2.19, 5.54+sigma*2.19);       wdD = wdD/sum(wdD);
[rho1,wrho1] = lgwt(nxi, 0.06-sigma*0.03, 0.06+sigma*0.03);   wrho1 = wrho1/sum(wrho1);
[rho2,wrho2] = lgwt(nxi, 0.05-sigma*0.03, 0.05+sigma*0.03);   wrho2 = wrho2/sum(wrho2);
[rho3,wrho3] = lgwt(nxi, 0.08-sigma*0.04, 0.08+sigma*0.04);   wrho3 = wrho3/sum(wrho3);
[rho4,wrho4] = lgwt(nxi, 0.54-sigma*0.22, 0.54+sigma*0.22);   wrho4 = wrho4/sum(wrho4);
[rho5,wrho5] = lgwt(nxi, 0.79-sigma*0.14, 0.79+sigma*0.14);   wrho5 = wrho5/sum(wrho5);
[rhop1,wrhop1] = lgwt(nxi, 0.26-sigma*0.23, 0.26+sigma*0.23); wrhop1 = wrhop1/sum(wrhop1);
[rhop2,wrhop2] = lgwt(nxi, 0.28-sigma*0.25, 0.28+sigma*0.25); wrhop2 = wrhop2/sum(wrhop2);
[rhop3,wrhop3] = lgwt(nxi, 0.33-sigma*0.27, 0.33+sigma*0.27); wrhop3 = wrhop3/sum(wrhop3);
[rhop4,wrhop4] = lgwt(nxi, 0.26-sigma*0.11, 0.26+sigma*0.11); wrhop4 = wrhop4/sum(wrhop4);
[rhop5,wrhop5] = lgwt(nxi, 0.80-sigma*0.13, 0.80+sigma*0.13); wrhop5 = wrhop5/sum(wrhop5);
[Nin, wNin] = lgwt(nxi, 276-sigma*133, 276+sigma*133);        wNin = wNin/sum(wNin);
[alpha123, walpha123] = lgwt(nxi, 0.63-sigma*0.21, 0.63+sigma*0.21); walpha123 = walpha123/sum(walpha123);
[alpha4, walpha4] = lgwt(nxi, 0.57-sigma*0.23, 0.57+sigma*0.23); walpha4 = walpha4/sum(walpha4);
[alpha5, walpha5] = lgwt(nxi, 0.71-sigma*0.23, 0.71+sigma*0.23); walpha5 = walpha5/sum(walpha5);

% Time grid
Nt = 7; % parse_parameter('Number of time steps for control', 7);
[Tnodes,Wt] = lgwt(Nt,17,90);
[Tnodes,prm] = sort(Tnodes);
Wt = Wt(prm);

u = ones(3,Nt); % Initial control

lowerbnd = zeros(size(u));
upperbnd = [0.69*ones(1,size(u,2));
            0.90*ones(1,size(u,2));
            0.59*ones(1,size(u,2))];
        
u = min(u, upperbnd);
u = max(u, lowerbnd);

eps0 = parse_parameter('Initial smoothing parameter eps0', 50);
gamma = parse_parameter('Moreau-Yosida parameter gamma', 5e5);
eps = eps0/sqrt(gamma);

tol = parse_parameter('TT approximation tolerance', 1e-2);

Xi = tt_meshgrid_vert(tt_tensor(Nin), tt_tensor(beta),tt_tensor(dL),tt_tensor(dC),tt_tensor(dR),tt_tensor(dRC),tt_tensor(dD),tt_tensor(rho1),tt_tensor(rho2),tt_tensor(rho3),tt_tensor(rho4),tt_tensor(rho5),tt_tensor(rhop1),tt_tensor(rhop2),tt_tensor(rhop3),tt_tensor(rhop4),tt_tensor(rhop5),tt_tensor(alpha123),tt_tensor(alpha4),tt_tensor(alpha5));
W = mtkron({tt_tensor(wNin), tt_tensor(wbeta),tt_tensor(wdL),tt_tensor(wdC),tt_tensor(wdR),tt_tensor(wdRC),tt_tensor(wdD),tt_tensor(wrho1),tt_tensor(wrho2),tt_tensor(wrho3),tt_tensor(wrho4),tt_tensor(wrho5),tt_tensor(wrhop1),tt_tensor(wrhop2),tt_tensor(wrhop3),tt_tensor(wrhop4),tt_tensor(wrhop5),tt_tensor(walpha123),tt_tensor(walpha4),tt_tensor(walpha5)});

Grad = 4; % Just initial TT rank
cost_increase = 0; % Will count cost increases for stopping

[~, t, Ichist, xhist] = plot_prior_Ic(u, Tnodes, S, Cfixed, gamma, eps0, sigma);
figure(2); plot(17:0.1:100, lagrange_interpolant(Tnodes, 17:0.1:100) * u'); title('u'); legend('work', 'school', 'other'); drawnow; 

Cost_state = amen_cross_s(Xi, @(x)SEIRcost(u, x(:,1)', Tnodes, x(:,2)',x(:,3)',x(:,4)',x(:,5)',x(:,6)',x(:,7)',[x(:,8)';x(:,9)';x(:,10)';x(:,11)';x(:,12)'],[x(:,13)';x(:,14)';x(:,15)';x(:,16)';x(:,17)'],[x(:,18)';x(:,19)';x(:,20)'],S,Cfixed, gamma, eps0), tol, 'normalize', true, 'exitdir', 1, 'dir', -1);
Cost = dot(W, Cost_state);
Cost(1) = Cost(1) + 0.5*eps*norm((u.^2)*Wt);
fprintf('Iter=0, Cost=%g+%g\n', Cost(1), Cost(2));

for iter=1:20    
    Grad = amen_cross_s(Xi, @(x)reshape(SEIRFDGrad(u, x(:,1)', Tnodes, x(:,2)',x(:,3)',x(:,4)',x(:,5)',x(:,6)',x(:,7)',[x(:,8)';x(:,9)';x(:,10)';x(:,11)';x(:,12)'],[x(:,13)';x(:,14)';x(:,15)';x(:,16)';x(:,17)'],[x(:,18)';x(:,19)';x(:,20)'],S,Cfixed, gamma, eps0), [], numel(u)), tol, 'kickrank', 0, 'y0', Grad, 'normalize', true, 'exitdir', 1, 'dir', -1, 'nswp', 2);
    
    % Expectation and the full gradient
    grad_J_u = reshape(dot(Grad, W), size(u)) + eps*u.*(Wt');
    
    % Control constraint
    grad_J_u = u - grad_J_u;
    grad_J_u = max(grad_J_u, lowerbnd);
    grad_J_u = min(grad_J_u, upperbnd);  
    grad_J_u = grad_J_u - u;    
        
    % Line search
    step = 1;
    Cost_new = inf;
    while (sum(Cost_new)>sum(Cost)) && (step>1e-2)
        unew = u + step*grad_J_u;
        
        Cost_state_new = amen_cross_s(Xi, @(x)SEIRcost(unew, x(:,1)', Tnodes, x(:,2)',x(:,3)',x(:,4)',x(:,5)',x(:,6)',x(:,7)',[x(:,8)';x(:,9)';x(:,10)';x(:,11)';x(:,12)'],[x(:,13)';x(:,14)';x(:,15)';x(:,16)';x(:,17)'],[x(:,18)';x(:,19)';x(:,20)'],S,Cfixed, gamma, eps0), tol, 'y0', Cost_state, 'kickrank', 2, 'normalize', true, 'exitdir', 1, 'dir', -1);
        Cost_new = dot(W, Cost_state_new);
        Cost_new(1) = Cost_new(1) + 0.5*eps*norm((unew.^2)*Wt);
        step = step/2;
    end
    if (sum(Cost_new)>sum(Cost))
        cost_increase = cost_increase+1;
    end
    u = unew;
    Cost_state = Cost_state_new;
    Cost = Cost_new;
    fprintf('Iter=%d, Cost=%g+%g, |G|=%g, step=%g, gamma=%g, eps=%g\n', iter, Cost_new(1), Cost_new(2), norm(grad_J_u,'fro'), step*2, gamma, eps0/sqrt(gamma));

    [~, t, Ichist, xhist, Rhist] = plot_prior_Ic(u, Tnodes, S, Cfixed, gamma, eps0, sigma);
    figure(2); plot(17:0.1:100, lagrange_interpolant(Tnodes, min(17:0.1:100, Tnodes(end))) * u'); title('u'); legend('work', 'school', 'other'); drawnow; 
        
    % print(1, sprintf('Ic-gamma%g-iter%d.pdf', gamma, iter), '-dpdf');
    % print(2, sprintf('u-gamma%g-iter%d.pdf', gamma, iter), '-dpdf');    
    % print(3, sprintf('R-gamma%g-iter%d.pdf', gamma, iter), '-dpdf');  
    % 
    % save(sprintf('all-gamma%g-iter%d.mat', gamma, iter));

    if (cost_increase>2)
        break;
    end
end
