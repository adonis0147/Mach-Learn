function [pre] = predictSVM(model, X)
    m = length(model.y);
    n = size(X, 1);
    p = zeros(n, 1);
    pre = zeros(n, 1);

    for i = 1:n
        s = 0;
        for j = 1:m
            s = s + model.alpha(j) * model.y(j) * ...
                model.kernelFunction(model.X(j,:)', X(i,:)');
        end
        p(i) = s + model.b;
    end

    pre(p >= 0) = 1;
    pre(p < 0) = 0;
end
