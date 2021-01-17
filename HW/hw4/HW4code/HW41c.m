clc;clear;
max_ite=1000;
n=100;
d=100;
A = rand(d,n);
%normalize A
for i=1:n
    A(:,i)=A(:,i)/norm(A(:,i));
end
x0 = normrnd(0,1,[n,1]);
%% GD
L = max(eig(A'*A));
f_GD=zeros(max_ite,1);
x_GD = x0;
ite=1;
while(ite <= max_ite)
    x_GD=x_GD-1/L*A'*A*x_GD;
    f_GD(ite)=log((norm(A*x_GD))^2);
    ite=ite+1;
end


%% SGD
f_SGD=zeros(max_ite,1);
x_SGD = x0;
ite=1;
alpha = 1/L;
while(ite <= max_ite)
    for j=1:n
        i = randi(n);
        x_SGD = x_SGD - alpha * A(i,:)'* A(i,:) * x_SGD;
    end 
    f_SGD(ite) = log10(norm(A * x_SGD)^2);
    ite = ite +1;
end
%% plot the gradient normvalue
plot_length=max_ite;
plot_vec=1:1:plot_length;
figure(1);
plot(plot_vec,f_GD(1:plot_length),'b-');
% ylim([0, 0.5]);
xlabel('iteration');
ylabel('log error');
legend('GD with constant stepsize');
title('GD Function Error v.s. iteration');



figure(2);
plot(plot_vec,f_SGD(1:plot_length),'r-');
xlabel('iteration');
ylabel('log error');
legend('SGD');
title('SGD Function Error v.s. iteration');

%% 1.(b)
ei=eig(A'*A);
ratio=max(ei)/mean(ei);