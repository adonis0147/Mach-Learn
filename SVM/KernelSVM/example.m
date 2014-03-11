clc; close all;

%%  Linear Kernel (Dataset 1)

clear;

load('../Data/ex6data1.mat');
y = 2 * y - 1;

plotData(X, y);

fprintf('Program paused. Press enter to continue.\n\n');
pause;

model = trainSVM(X, y, @linearKernel);

plotLinearBoundary(model, X);

fprintf('Program paused. Press enter to continue.\n\n');
pause;

%%  Gaussian Kernel (Dataset 2)

clear; close all;

load('../Data/ex6data2.mat');
y = 2 * y - 1;

plotData(X, y);

fprintf('Program paused. Press enter to continue.\n\n');
pause;

fprintf('Training ...\n');

sigma = 0.1;

model = trainSVM(X, y, @(x1, x2)gaussianKernel(x1, x2, sigma));

fprintf('Done!\n');

plotBoundary(model, X);
