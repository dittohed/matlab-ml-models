function [X_norm, mu, sigma] = feature_norm(X)
% normalize features using standarization
    mu = mean(X);
    sigma = std(X);
    X_norm = (X - mu) ./ sigma;
end
