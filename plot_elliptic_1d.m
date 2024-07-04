function plot_elliptic_1d(model,u,y_tt,W)
ny = size(model.B,1);
subplot(1,2,1); plot(u); legend('u'); xlabel('grid point');
subplot(1,2,2); plot(1:ny, full(dot(W,y_tt,2,y_tt.d)), 'b-', ...
    1:ny, repmat(tt_stat(model.Ymax,'sr'),1,ny), 'k-', ...
    1:ny, model.yd, 'r-');
legend('E[y]', 'ymax', 'yd'); xlabel('grid point');
end