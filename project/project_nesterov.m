%% Set up Problem
data = importdata('Users/wtq/Desktop/sp18/ie510/project/normed_data.csv');
n = 569;
d = 26;
X = data(:,1:d);
Y = data(:,27);
L = max(eig(X'*X));
w = normrnd(0,1,[d,1]);
max_it = 4*10^2;
alpha = 1/L;
%%
beta = 0.9;
grad_norm = zeros(max_it + 1, 1);
times = zeros(max_it +1, 1);
t = cputime;
w_prev = zeros(d,1);
grad_prev = zeros(d,1);
ite = 1;
%% run the algorithm
while (ite <= max_it)
    grad_w = zeros(d,1);
    yr = w +  beta * (w - w_prev);
    for i=1:n
        z = exp(-Y(i)*X(i,:)*w);
        grad_w = grad_w+ (-Y(i)*X(i,:)*z)'/(1+z);
    end
    w_prev = w;
    w = yr - alpha*grad_w;
    grad_norm(ite+1) = log10(norm(grad_w/n));
    ite = ite+1;
    times(ite+1) = cputime - t;
end

%% Plot the function
figure;
plot_length = max_it;
plot_vec = 1:1:plot_length;
plot(plot_vec, grad_norm(1:plot_length), 'b-');
plot(times(1:plot_length), grad_norm(1:plot_length), 'b-');
xlabel('iteration');
xlabel('cputime');
ylabel('Log Gradient Norm');
title('Nesterov’s');

