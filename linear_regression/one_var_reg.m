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
alpha = 0.01;
init_theta = zeros(2, 1);

X = [ones(m, 1), data(:, 1)]; % add 0th feauture to fit the hypothesis better

[theta, J_vals] = grad_descent(X, y, init_theta, alpha, n_iters);

fprintf("Computed parameters:\n");
fprintf("%f\n", theta);

%% Plot the learned hypothesis
plot(X(:, 2), y, "rx");
hold on;
plot(X(:, 2), X * theta, 'b');
hold off;

%% Plot how J was decreasing
plot(1 : n_iters, J_vals);

%% Plot 3D cost function
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-2, 5, 100);

J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1 : length(theta0_vals)
    for j = 1 : length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i, j) = compute_cost(X, y, t);
    end
end

J_vals = J_vals'; % always do it before calling surf() command

surf(theta0_vals, theta1_vals, J_vals);
xlabel("\theta_0"); % \theta returns actual greek letter 
ylabel("\theta_1");