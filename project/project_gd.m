data = importdata('Users/wtq/Desktop/sp18/ie510/project/normed_data.csv');
n = 569;
d = 26;
X = data(:,1:d);
Y = data(:,27);
w = normrnd(0,1,[d,1]);
L = max(eig(X'*X));

max_it = 4*10^2;
grad_norm = zeros(max_it + 1, 1);
ite = 1;
alpha = 1/L;
t = cputime;
times = zeros(max_it + 1, 1);
while(ite<=max_it)
    grad_w = zeros(d,1);
    for i=1:n
        z = exp(-Y(i)*X(i,:)*w);
        grad_w = grad_w+ (-Y(i)*X(i,:)*z)'/(1+z);
    end
    w = w - alpha*grad_w;
    grad_norm(ite) = log10(norm(grad_w/n));
    times(ite) = cputime-t;
    ite = ite +1;
    
end
figure;
plot_length = max_it;
plot_vec = 1:1:plot_length;
plot(plot_vec, grad_norm(1:plot_length), 'b-');
plot(times(1:plot_length), grad_norm(1:plot_length), 'b-');
xlabel('iteration');
xlabel('cputime');
ylabel('Log Gradient Norm');
title('GD');