function plot_elliptic_2d(model,u,y_tt,W,xi_grid)
ny = size(model.B,1);
x = (1:sqrt(ny))/(sqrt(ny)+1);
subplot(1,2,1); mesh(x,x,reshape(u,sqrt(ny),sqrt(ny))'); legend('u'); xlabel('x_1'); ylabel('x_2');
subplot(1,2,2); mesh(x,x,reshape(full(dot(W,y_tt,2,y_tt.d)), sqrt(ny), sqrt(ny))'); legend('E[y]'); xlabel('x_1'); ylabel('x_2');
end
