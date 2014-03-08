function Q = computeQ(X, y, kernel)
    m = length(y);
    Q = zeros(m, m);

    for i = 1:m
        for j = 1:m
            Q(i,j) = y(i) * kernel(X(i,:)', X(j,:)') * y(j);
        end
    end
end
