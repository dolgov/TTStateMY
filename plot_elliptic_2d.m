function plot_elliptic_2d(model,u,y_tt,W)
ny = size(model.B,1);
subplot(1,2,1); mesh(reshape(u,sqrt(ny),sqrt(ny))'); legend('u'); xlabel('grid point'); ylabel('grid point');
subplot(1,2,2); mesh(reshape(full(dot(W,y_tt,2,y_tt.d)), sqrt(ny), sqrt(ny))'); legend('E[y]'); xlabel('grid point'); ylabel('grid point');
end
