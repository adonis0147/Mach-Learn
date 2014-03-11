function plotLinearBoundary(model, X)
    x = linspace(min(X(:,1)), max(X(:,1)), 100);

    y = -(model.w(1) * x + model.b) / model.w(2);
    y_pos = -(model.w(1) * x + model.b - 1) / model.w(2);
    y_neg  = -(model.w(1) * x + model.b + 1) / model.w(2);

    hold on;

    plot(x, y);
    plot(x(15:end), y_pos(15:end), '--');
    plot(x(1:end), y_neg(1:end), '--');

    hold off;
end
