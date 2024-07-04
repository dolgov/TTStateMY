function [y,A] = solve_fun_elliptic_1d(model,u,xi, bc)
%%% Solve Ex 5.2,
% -nu(xi1) d^2/dx^2 y = f(xi2) + u, u(0) = d0(xi3), u(1) = d1(xi4)
% Compute the solution in block TT format
I = size(xi,1);
[ny,nu] = size(model.B);
for i=I:-1:1
    A = 10^(xi(i,1)-2) * model.A0;
    b = model.B*u;
    if (bc)
        b = b + xi(i,2)/100;
        % Boundary conditions
        b(1,:) = b(1,:) - A(2,1) * (1+xi(i,3)/1000);
        b(ny,:) = b(ny,:) - A(ny+1,ny+2) * (2+xi(i,4))/1000;
    end
    A = A(2:ny+1, 2:ny+1);
    y(i,:) = -reshape(A\b, 1, []);  % Forward solution
end
end
