% This code implements simple linear svm with hard margin.
% The objective function is:
%
%    min w' * w / 2
%    w,b
%
%    subject to:    y(i) * (w' * x(i) + b) >= 1     for i = 1,2,...,m
%
%
% The dual problem is:
%
%   max 1' * alpha - 1 / 2 * alpha' * Q * alpha
%  alpha
%
%   subject to:     y' * alpha = 0  alpha >= 0
%
% where Q(i,j) = y(i) * kernel(x(i), x(j)) * y(j)


clear; clc; close all;

% Read data
data = load('../../Data/ex6data1.mat');

% Remove the outiler (the hard margin model is sensitive to outilers)
X = data.X(1:end-1,:);
y = 2 * data.y(1:end-1) - 1;    % y(i) = {1, -1}

plotData(X, y);

Q = computeQ(X, y, @linearKernel);

% Solver for the dual problem:  quadratic programming solver in Matlab
options = optimset('Algorithm', 'interior-point-convex');
[alpha, fVal] = quadprog(Q, -ones(length(y), 1), ...
                        [], [], y', 0, zeros(length(y), 1), [], [], options);


fprintf('Program paused. Press enter to continue.\n\n');
pause;


% Find the support vectors
epsilon = 1e-6;
sv = find(alpha > epsilon);

% plot the support vectors
hold on;
plot(X(sv,1), X(sv,2), 'ro', 'MarkerSize', 10);
hold off;


fprintf('Mark the support vectors. \n\n');
fprintf('Program paused. Press enter to continue.\n\n');
pause;


% w = sum alpha(i) * y(i) * x(i)
w = sum(X(sv,:)' * (alpha(sv) .* y(sv)), 2);

% b = 1 / m * sum y(i) - w' * x(i)
b = mean(y(sv) - X(sv,:) * w);

fprintf('The value of w: \n');
disp(w);

fprintf('The value of b: \n');
disp(b);

plotLinearBoundary(w, b, min(X(:,1)), max(X(:,1)));
