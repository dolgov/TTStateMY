function [y,Jac] = solve_y_vi(model,u,xi,tau_min)
global Nsolves Newton_inner
% Forward solution
b = model.phil * xi';
b = model.f0 - b + model.B*u;
% Newton iteration
tau = tau_min;
y = zeros(size(b));
for iter=1:100   
    res = model.A*y + b + logsmooth(y,tau)/tau;
    Jac = kron(speye(size(y,2)), model.A) + spdiags(grad_logsmooth(y(:),tau,0)/tau, 0, numel(y), numel(y));
    dy = Jac \ res(:);
    Nsolves = Nsolves + size(y,2);
    y = y - reshape(dy, size(y));
    if (norm(dy,'fro') < 1e-12 * norm(y,'fro')) && (tau <= tau_min*2)
        break;
    end
    tau = max(tau*sqrt(0.1), tau_min);
end
fprintf('Fwd Newton iter=%d, |dy|=%g\n', iter, norm(dy,'fro')/norm(y,'fro'));
Newton_inner(1) = Newton_inner(1) + 1;
Newton_inner(2) = Newton_inner(2) + iter;
if (nargout>1)
    Jac = kron(speye(size(y,2)), model.A) + spdiags(grad_logsmooth(y(:),tau,0)/tau, 0, numel(y), numel(y));
end
y = y'; % For multifun cross I x d format
end
