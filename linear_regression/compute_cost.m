function J = compute_cost(X, y, theta)
% This function computes an average cost (value of the square-error function).

    m = length(y); 

    J = 1 / (2 * m) * (X * theta - y)' * (X * theta - y);

end