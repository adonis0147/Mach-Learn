function [model] = trainSVM(X, y, kernel, C, epsilon)
%
% This code implements simple kernel svm with soft margin.
% The objective function is:
%
%    min w' * w / 2 + C * sum xi(i)
%   w,b,xi                 i
%
%    subject to:    y(i) * (w' * f(x(i)) + b) >= 1 - xi    for i = 1,2,...,m
%                   xi(i) >= 0      for i = 1,2,...,m
%                   C > 0
%
% where f(x(i)) is the vector in feature space
%
% The dual problem is:
%
%   max 1' * alpha - 1 / 2 * alpha' * Q * alpha
%  alpha
%
%   subject to:     y' * alpha = 0  0 <= alpha <= C
%
% where Q(i,j) = y(i) * kernel(x(i), x(j)) * y(j)
%
%
% INPUT:
%   X:  Samples
%   y:  Labels - {1, -1}
%   kernel: Kernel function
%   C:  Free parameter which controls the relative importance of the term of w
%   epsilon:  Accuracy
%
% OUTPUT:
%   model:  The SVM model
%       model.X:    Support vectors
%       model.y:    The labels of support vectors
%       model.kernelFunction:   The kernel function used to train
%       model.alpha:    The dual variables correspoding to support vectors
%       model.w:    Linear margin when using linear kernel
%       model.b:    bias

if nargin < 3
    error('Arguments error!');
elseif nargin == 3
    C = 1;
    epsilon = 1e-6;
elseif nargin == 4
    epsilon = 1e-6;
end

m = length(y);

% Kernel matrix
K = zeros(m, m);
for i = 1:m
    for j = i:m
        K(i,j) = kernel(X(i,:)', X(j,:)');
        K(j,i) = K(i,j);
    end
end

Q = diag(y) * K * diag(y);

options = optimset('Algorithm', 'interior-point-convex');
[alpha, fVal] = quadprog(Q, -ones(m, 1), ...
                    [], [], y', 0, zeros(m, 1), C * ones(m, 1), [], options);

% Find the support vectors
sv = find(alpha > epsilon);

% w = sum alpha(i) * y(i) * x(i)    (linear margin when using linear kernel)
w = X(sv,:)' * diag(y(sv)) * alpha(sv);

% Find some support vectors which are on the margin (0 < alpha < C)
msv = find(alpha > epsilon & abs(alpha - C) > epsilon);

% b = 1 / m * sum y(i) - w' * f(x(i))
b = mean(y(msv)' - alpha(sv)' * diag(y(sv)) * K(sv,msv));

model = struct;
model.X = X(sv,:);
model.y = y(sv);
model.kernelFunction = kernel;
model.alpha = alpha(sv);
model.w = w;
model.b = b;

end
