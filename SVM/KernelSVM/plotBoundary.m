function plotBoundary(model, X)
    x1 = linspace(min(X(:,1)), max(X(:,1)), 100);
    x2 = linspace(min(X(:,2)), max(X(:,2)), 100);
    [X1, X2] = meshgrid(x1, x2);

    vals = zeros(size(X1));
    for i = 1:size(X1, 2)
        this_X = [X1(:,i), X2(:,i)];
        vals(:,i) = predictSVM(model, this_X);
    end

    hold on;
    contour(X1, X2, vals, [0 0], 'Color', 'b');
    hold off;
end
