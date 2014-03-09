function plotLinearBoundary(w, b, x_min, x_max)
    x = linspace(x_min, x_max, 100);

    y = -(w(1) * x + b) / w(2);
    y_pos = -(w(1) * x + b - 1) / w(2);
    y_neg  = -(w(1) * x + b + 1) / w(2);

    hold on;

    plot(x, y);
    plot(x(15:end), y_pos(15:end), '--');
    plot(x(1:end), y_neg(1:end), '--');

    hold off;
end
