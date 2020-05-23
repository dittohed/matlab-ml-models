function [theta, J_vals] = grad_descent(X, y, theta, alpha, n_iters)
% This function performs gradient descent and returns subsequent J_vals
% to then plot how J (hopefully) has been decreasing.

    m = length(y);
    J_vals = zeros(n_iters, 1);
    
    for i = 1 : n_iters
        delta = (1 / m) * ((X * theta - y)' * X)';
        theta = theta - alpha * delta;

        J_vals(i) = compute_cost(X, y, theta);
    end

end