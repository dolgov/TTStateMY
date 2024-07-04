function [Cost_state, t, Ichist, xhist, Rhist] = plot_prior_Ic(u, Tnodes, S, Cfixed, gamma, eps0, sigma)
for irun=1:100
    % Sample random var for a simple uncontrolled stats
    sbeta = 0.13+sigma*0.03*(rand*2-1);
    sdL = 1.57+sigma*0.42*(rand*2-1);
    sdC = 2.12+sigma*0.80*(rand*2-1);
    sdR = 1.54+sigma*0.40*(rand*2-1);
    sdRC = 12.08+sigma*1.51*(rand*2-1);
    sdD = 5.54+sigma*2.19*(rand*2-1);
    srho1 = 0.06+sigma*0.03*(rand*2-1);
    srho2 = 0.05+sigma*0.03*(rand*2-1);
    srho3 = 0.08+sigma*0.04*(rand*2-1);
    srho4 = 0.54+sigma*0.22*(rand*2-1);
    srho5 = 0.79+sigma*0.14*(rand*2-1);
    srhop1 = 0.26+sigma*0.23*(rand*2-1);
    srhop2 = 0.28+sigma*0.25*(rand*2-1);
    srhop3 = 0.33+sigma*0.27*(rand*2-1);
    srhop4 = 0.26+sigma*0.11*(rand*2-1);
    srhop5 = 0.80+sigma*0.13*(rand*2-1);
    sNin = 276+sigma*133*(rand*2-1);
    salpha123 = 0.63+sigma*0.21*(rand*2-1);
    salpha4 = 0.57+sigma*0.23*(rand*2-1);
    salpha5 = 0.71+sigma*0.23*(rand*2-1);
    [Cost_state(irun,:),t,xhist(:,:,:,irun),Rhist(irun)] = SEIRcost(u, sNin, Tnodes, sbeta,sdL,sdC,sdR,sdRC,sdD,[srho1;srho2;srho3;srho4;srho5],[srhop1;srhop2;srhop3;srhop4;srhop5],[salpha123;salpha4;salpha5],S,Cfixed, gamma, eps0);
end
Ichist = sum(xhist(:,:,4,:)+xhist(:,:,5,:), 2);
Iclow = quantile(Ichist,0.05,4);
Ichigh = quantile(Ichist,0.95,4);
figure(1);
plot(t, Iclow,   t, Ichigh,   90*[1,1], [0,2e4], 'k--');
legend('5%', '95%');
figure(3);
histogram(Rhist);
drawnow;
end
