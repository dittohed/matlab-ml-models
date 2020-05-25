function [J, grad] = cost(theta, X, y)

    m = length(y); 

    h = sigmoid(X * theta);
    J = (-1 / m) * (y' * log(h) + (1 - y)' * log(1 - h)); 
    grad = (1 / m) * X' * (h - y);

end