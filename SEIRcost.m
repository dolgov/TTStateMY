function [Cost,t,x,R] = SEIRcost(u, Nin, Tnodes, beta,dL,dC,dR,dRC,dD,rho,rhop,alpha,S,Cfixed, gamma, eps0)
% Cost function of the SEIR model

% initialize infected
Nin = Nin .* [0.1; 0.4; 0.35; 0.1; 0.05];

% Severity of infection
Ein = Nin/3;
I1in = rho.*Nin.*2/3;
I2in = (1-rho).*Nin.*2/3;
IC1in = zeros(5,size(Nin,2));
IC2in = zeros(5,size(Nin,2));

% Initial state
x0 = [Ein; I1in; I2in; IC1in; IC2in];
x0 = x0(:);

dt = 0.1;

t = (0:dt:100)';
nend = 90/dt+1;
[x,R] = SEIRIEuler(dt,100/dt,x0, u, Tnodes, beta,dL,dC,dR,dRC,dD,rho,rhop,alpha,S,Cfixed);

x = reshape(x, numel(t), 5, 5, []);
D = sum(x(1:nend,:,4,:), 2);
D = D.*[t(2)-t(1); t(2:nend)-t(1:nend-1)]; D(1,:,:,:)=D(1,:,:,:)/2; D(nend,:,:,:)=D(nend,:,:,:)/2; D = sum(D);
D = D./reshape(dD, 1,1,1,[]);

% Moreau-Yosida term
eps = eps0/sqrt(gamma);

G = logsmooth(R-1, eps);
G = G.^2;

Cost = [D(:)/2, gamma*G(:)];

end

