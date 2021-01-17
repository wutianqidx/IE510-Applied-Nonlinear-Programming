clc;clear;
%%
data = importdata('Users/wtq/Desktop/sp18/ie510/project/normed_data.csv');
n = 569;
d = 26;
X = data(:,1:d);
Y = data(:,27);
L = max(eig(X'*X));

max_it = 4*10^2;
alpha = 1/L;
plot_length = max_it;
plot_vec = 1:1:plot_length;
%%
beta_HB = 0.9;

HB_gn = zeros(length(beta_HB), max_it + 1);
for b = 1:length(beta_HB)
    w = normrnd(0,1,[d,1]);
    w_prev = w;
    ite =1;
    while(ite<=max_it)
        grad_w = zeros(d,1);
        for i=1:n
            z = exp(-Y(i)*X(i,:)*w);
            grad_w = grad_w+ (-Y(i)*X(i,:)*z)'/(1+z);
        end
        HB_gn(b,ite) = log10(norm(grad_w/n));
        w_new = w - alpha*grad_w + beta_HB(b)*(w-w_prev);
        w_prev = w;
        w = w_new;
        ite = ite +1;
    end
end
figure(1);
for i = 1:length(beta_HB)
    fig = plot(plot_vec, HB_gn(i,1:plot_length));
    hold on;
%     legend(fig,num2str(beta_HB(i)));
end


 
legend('beta=0.1','beta=0.2','beta=0.3','beta=0.4','beta=0.5','beta=0.6','beta=0.7','beta=0.8','beta=0.9','beta=1.0');
 
xlabel('iteration');
ylabel('grad norm');
title('Log Gradient Norm v.s. iteration');