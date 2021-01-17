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
    f_GD(ite)=log10((norm(A*x_GD))^2);
    ite=ite+1;
end


%% CD
f_CD=zeros(max_ite,1);
x_CD = x0;
ite=1;
while(ite <=max_ite)
    r=-A*x_CD;
    for j=1:n
        i=round(rand()*100+0.5);
        x_new=A(:,i)'*r + x_CD(i);
        r=r+A(:,i)*(x_CD(i)-x_new);
        x_CD(i)=x_new;
    end
    f_CD(ite)=log10((norm(A*x_CD))^2);
    ite=ite+1;
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
plot(plot_vec,f_CD(1:plot_length),'r-');
xlabel('iteration');
ylabel('log error');
legend('R-CD');
title('R-CD Function Error v.s. iteration');

%% 1.(b)
ei=eig(A'*A);
ratio=max(ei)/mean(ei);

