function plot_elliptic_1d(model,u,y_tt,W,xi_grid)
ny = size(model.B,1);
x = (1:ny)/(ny+1);
subplot(1,2,1); plot(x,u); legend('u'); xlabel('x');
subplot(1,2,2);
hold off;
Ysamples = tt_sample_lagr(tt_reshape(y_tt, W.n, [], ny, 1), xi_grid, rand(1000, W.d)*2-1);
Ylow = quantile(Ysamples, 0.05);
Yhigh = quantile(Ysamples, 0.95);
fill([x fliplr(x)], [Ylow fliplr(Yhigh)], 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none')
hold on;
plot(x, full(dot(W,y_tt,2,y_tt.d)), 'b-');
plot(x, repmat(tt_stat(model.Ymax,'sr'),1,ny), 'k-');
plot(x, model.yd, 'r-');
legend('CI[y]', 'E[y]', '\psi', 'y_d'); xlabel('x');
end