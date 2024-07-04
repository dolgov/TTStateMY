function [G] = grad_my_elliptic_1d(model, u, xi, xi_grid, eps)
% Gradient of the Moreau-Yosida term
% xi_grid should be a cell of vector of grid points
I = size(xi,1);
[ny,nu] = size(model.B);
ymax = tt_sample_lagr(model.Ymaxvec, xi_grid, xi);

for i=I:-1:1
    % Forward solution
    [y,A] = solve_fun_elliptic_1d(model,u,xi(i,:),true);
    y = y(:);
    
    ymaxi = ymax(i,:).';
    
    b = grad_logsmooth(y-ymaxi, eps,0) .* (model.My * logsmooth(y-ymaxi, eps));
    
    G(i,1:nu) = model.B'*((A')\(-b));
end
end

