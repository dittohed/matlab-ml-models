% Task: build a logistic regression model to
% predict whether a student gets admitted into a university.
% Suppose that you are the administrator of a university department and
% you want to determine each applicant’s chance of admission based on their
% results on two exams. You have historical data from previous applicants
% that you can use as a training set for logistic regression. For each training
% example, you have the applicant’s scores on two exams and the admissions
% decision.

%% Load the data
data = load("ex2data1.txt");
X = data(:, 1 : 2);
y = data(:, 3);
m = length(y);

%% Plot the data
% first let's seperate positive and negative samples
pos_indices = find(y);
neg_indices = find(y == 0);

plot(X(pos_indices, 1), X(pos_indices, 2), "bx", "MarkerSize", 10);
hold on;
plot(X(neg_indices, 1), X(neg_indices, 2), "ro", "MarkerSize", 10);
legend("Admitted", "Not admitted");
hold off;

%% Training
% we will use the fminunc (find minimum of unconstrained multivariable
% function) for the efficiency and safety

X = [ones(m, 1), X]; % add intercept term

% options for fminunc
options = optimset("GradObj", "on", "MaxIter", 400);
% we tell the fminunc that we provide gradient and set the no. of iterations

initial_theta = zeros(size(X, 2), 1);

[theta, cost] = ...
    fminunc(@(t)(cost(t, X, y)), initial_theta, options);
% we provide fminunc with the anonymous function made of our cost function
% so it can use it (minimize it by changing theta)

%% Plot the decision boundary along with the data
plot(X(pos_indices, 2), X(pos_indices, 3), "bx", "MarkerSize", 10);

hold on;

plot(X(neg_indices, 2), X(neg_indices, 3), "ro", "MarkerSize", 10);

hold on;

x = linspace(30, 100, 100);
plot(x, (-theta(2) / theta(3)) * x - (theta(1) / theta(3)), 'k');

legend("Admitted", "Not admitted");
hold off;