% Task: implement linear regression with multiple variables to
% predict the prices of houses. 
% Suppose you are selling your house and you
% want to know what a good market price would be. One way to do this is to
% first collect information on recent houses sold and make a model of housing
% prices.
% The file ex1data2.txt contains a training set of housing prices in Port-
% land, Oregon. The first column is the size of the house (in square feet), the
% second column is the number of bedrooms, and the third column is the price
% of the house.

%% Load and normalize the data 
data = load("ex1data2.txt"); % read csv into a matrix
X = data(:, 1 : 2);
y = data(:, 3);

% Normalization step is vital to make the model work efficient
% and allows the model to 'treat' each feature the same way.
[X, mu, sigma] = feature_norm(X);

% add bias term
m = length(y);
X = [ones(m, 1), X];

%% Training
% prepare for training
n_iters = 500;
alpha = 0.3;
init_theta = zeros(3, 1);

[theta, J_vals] = grad_descent(X, y, init_theta, alpha, n_iters);

fprintf("Computed parameters:\n");
fprintf("%f\n", theta);

%% Plot how J was decreasing - change alpha/n_iters if necessary 
plot(1 : n_iters, J_vals);

%% Now let's use a different method called normal equation derived from stats
% (solving the problem analytically, not numerically)
house = [1650 3];
price = [1, ((house - mu) ./ sigma)] * theta;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

% no need to normalize feautures 
X = data(:, 1 : 2);
X = [ones(m, 1), X];

theta = inv(X' * X) * X' * y; 

house = [1 1650 3];
price = house * theta;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);
