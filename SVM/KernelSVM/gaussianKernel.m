function [val] = gaussianKernel(x1, x2, sigma)
    val = exp(- norm(x1 - x2) ^ 2 / (2 * sigma ^ 2));
end
