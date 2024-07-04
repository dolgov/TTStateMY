function [G] = grad_y_fun_elliptic_1d(model, xi)
% \nabla_u y  realised by solving for unit control vectors
% xi_grid should be a cell of vector of grid points
I = size(xi,1);
[ny,nu] = size(model.B);
for i=I:-1:1
    % Forward solution
    [gi,~] = solve_fun_elliptic_1d(model,eye(nu),xi(i,:),false);
    gi = reshape(gi, ny, nu);
    G(i,:) = reshape(gi', 1, nu*ny);
end
end
