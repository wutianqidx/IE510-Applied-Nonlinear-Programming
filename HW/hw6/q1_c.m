rng('default')
n = 10^4;
A = randn(n,n);
max1 = max(eig(A'*A));
min1 = min(eig(A'*A));
k = max1/min1