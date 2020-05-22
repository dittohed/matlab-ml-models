% Task: implement linear regression with one
% variable to predict profits for a food truck. Suppose you are the CEO of a
% restaurant franchise and are considering different cities for opening a new
% outlet. The chain already has trucks in various cities and you have data for
% profits and populations from the cities.

%% Load and plot the data
data = load("ex1data1.txt"); % read csv into a matrix
X = data(:, 1);
y = data(:, 2);

plot(X, y, "rx");
xlabel("Population in 10 000s");
ylabel("Profit in $10 000s");

%% Training
% prepare for training
m = length(y);

n_iters = 1500;
alpha = 0.1;
init_theta = zeros(2, 1);

X = [ones(m, 1), X]; % add 0th feauture to fit the hypothesis better

[theta, J_vals] = grad_descent(X, y, init_theta, alpha, n_iters);

fprintf('Computed parameters:\n');
fprintf('%f\n', theta);