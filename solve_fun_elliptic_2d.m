function [y,A] = solve_fun_elliptic_2d(model,u,xi, bc)
% Compute the solution in block TT format
I = size(xi,1);
[ny,nu] = size(model.B);
ny = round(sqrt(ny));

b1 = 1 + xi(:,3)/1000;
b2 = (2 + xi(:,4))/1000;
b3 = 1 + xi(:,5)/1000;
b4 = (2 + xi(:,6))/1000;

h = 1/(ny+1);

bound_b = (1:ny+2-1)'; xb = h*((1:ny+2-1)'-1);
bound_l = 1 + ((2:ny+2)'-1)*(ny+2); yl = h*((2:ny+2)'-1); % make boundary indices non-overlapping
bound_t = (2:ny+2)' + (ny+2-1)*(ny+2); xt = h*((2:ny+2)'-1);
bound_r = ny+2 + ((1:ny+2-1)'-1)*(ny+2); yr = h*((1:ny+2-1)'-1);
inner = (1:(ny+2)^2)';
inner([bound_l; bound_t; bound_r; bound_b]) = [];

for i=I:-1:1
    A = 10^(xi(i,1)-2) * model.A0;
    b = model.B*u;
    if (bc)
        b = b + xi(i,2)/100;
        % Boundary conditions    
        b = b - A(inner,bound_l) * (b1(i)*(1-yl) + b2(i)*yl);
        b = b - A(inner,bound_t) * (b2(i)*(1-xt) + b3(i)*xt);
        b = b - A(inner,bound_r) * (b4(i)*(1-yr) + b3(i)*yr);
        b = b - A(inner,bound_b) * (b1(i)*(1-xb) + b4(i)*xb);
    end    
    A = A(inner, inner);
    y(i,:) = -reshape(A\b, 1, []);  % Forward solution
end
end
