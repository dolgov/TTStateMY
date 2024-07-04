function [G] = j_grad_fun_elliptic_1d(model,u,xi, deriv, vec)
%%% Gradients and Hessian(-vecs) for Ex 5.2,
% -nu(xi1) d^2/dx^2 y = f(xi2) + u, u(0) = d0(xi3), u(1) = d1(xi4)
% Compute cost and its derivatives in u
I = size(xi,1);
[ny,nu] = size(model.B);
for i=I:-1:1
    % Forward solution
    [y,A] = solve_fun_elliptic_1d(model,u,xi(i,:),true);
    y = y(:);
    b = model.My*(y-model.yd);
    bi = b;  % interior nodes only
    phi = -((A')\bi); % Adjoint (sensitivity) soln
    if (nargin<4)
        G(i,1:1+nu) = [0.5*(y-model.yd)'*b   phi'*model.B];
    else
        if (deriv==0)
            G(i,1) = 0.5*(y-model.yd)'*b;
        end
        if (deriv==1)
            G(i,1:nu) = phi'*model.B;
        end
        if (deriv==2)
            if (nargin>4)
                % Matvec only
                w = model.B*vec;
                w = -(A\w);
                w = model.My*w;
                w = -((A')\w);
                w = model.B'*w;
                G(i,1:nu) = w;
            else
                % Full hessian
                G(i,1:nu^2) = full(reshape(model.B'*inv(A')*model.My*inv(A)*model.B, 1, nu^2)); % B'*A^{-T}*My*A^{-1}*B
            end
        end
    end
end
end
