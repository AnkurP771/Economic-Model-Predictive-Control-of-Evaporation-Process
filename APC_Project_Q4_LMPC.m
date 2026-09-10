clc
clear
close all

%% PARAMETERS 

M  = 20;
C  = 4;
Cp = 0.07;
lambda = 38.5;

UA2 = 6.84;

F1   = 10;
X1   = 5;
F3   = 50;
T1   = 40;
T200 = 25;

d_ss = [F1 X1 F3 T1 T200];

u_ss = [193.45; 207.33];

Ts = 1;

%% NONLINEAR MODEL

f = @(x,u,d) [
(1/M)*( d(1)*d(2) - (d(1)-F4_fun(x,u,d))*x(1) );
(1/C)*( F4_fun(x,u,d) - F5_fun(x,u,d) )
];

function F4 = F4_fun(x,u,d)
Cp = 0.07; lambda = 38.5;
T2 = 0.5616*x(2) + 0.3126*x(1) + 48.43;
T100 = 0.1538*u(1) + 90;
UA1 = 0.16*(d(1) + d(3));
Q100 = UA1*(T100 - T2);
F4 = (Q100 - d(1)*Cp*(T2 - d(4)))/lambda;
end

function F5 = F5_fun(x,u,d)
Cp = 0.07; lambda = 38.5; UA2 = 6.84;
T3 = 0.507*x(2) + 55;
Q200 = UA2*(T3 - d(5)) / (1 + UA2/(2*Cp*u(2)));
F5 = Q200/lambda;
end

%% LINEAR MODEL

x_ss = [25; 50.57];

syms X2 P2 P100 F200 F1 X1 F3 T1 T200 real

T2 = 0.5616*P2 + 0.3126*X2 + 48.43;
T3 = 0.507*P2 + 55;
T100 = 0.1538*P100 + 90;
UA1 = 0.16*(F1 + F3);

Q100 = UA1*(T100 - T2);
F4 = (Q100 - F1*Cp*(T2 - T1))/lambda;

Q200 = UA2*(T3 - T200) / (1 + UA2/(2*Cp*F200));
F5 = Q200/lambda;

f1 = (1/M)*(F1*X1 - (F1 - F4)*X2);
f2 = (1/C)*(F4 - F5);

F = [f1; f2];

A = double(subs(jacobian(F,[X2 P2]), ...
    [X2 P2 P100 F200 F1 X1 F3 T1 T200], ...
    [x_ss' u_ss' d_ss]));

Bu = double(subs(jacobian(F,[P100 F200]), ...
    [X2 P2 P100 F200 F1 X1 F3 T1 T200], ...
    [x_ss' u_ss' d_ss]));

sys = ss(A,Bu,eye(2),0);
sys_d = c2d(sys,Ts);

Phi = sys_d.A;
Gamma = sys_d.B;

%% TEST DIFFERENT Np, Nc

Np_list = [10 20 30];
Nc_list = [3 5 10];

Q = diag([0.92 0.3]);
R = diag([1 1]);

N_sim = 300;

results = zeros(length(Np_list), length(Nc_list));

for a = 1:length(Np_list)
for b = 1:length(Nc_list)

Np = Np_list(a);
Nc = Nc_list(b);

X = zeros(2,N_sim);
U = zeros(2,N_sim);

X(:,1) = x_ss;
u_prev = zeros(2,1);

J_perf = 0;

for k = 1:N_sim-1

    d = d_ss;

    % choosing disturbance rejection
    if k>=100
        d(1) = 1.1*d_ss(1);
    end

    x_ref = x_ss;

    x_dev = X(:,k) - x_ss;

    obj = @(Uvec) cost_function(Uvec, x_dev, zeros(2,1), ...
        Phi, Gamma, Q, R, Np, Nc, u_prev);

    U0 = zeros(2*Nc,1);

    U_opt = fmincon(obj, U0);

    u_dev = U_opt(1:2);
    u = u_dev + u_ss;

    dx = f(X(:,k), u, d);
    X(:,k+1) = X(:,k) + dx*Ts;

    % performance accumulation
    J_perf = J_perf + norm(X(:,k) - x_ref)^2;

    u_prev = u_dev;

end

results(a,b) = J_perf;

end
end

%% DISPLAY RESULTS

fprintf('\n LMPC Performance Table \n\n');

Np_list = [10 20 30];
Nc_list = [3 5 10];

fprintf('        Nc=3        Nc=5        Nc=10\n');

for i = 1:length(Np_list)
    fprintf('Np=%2d   %8.2f   %8.2f   %8.2f\n', ...
        Np_list(i), results(i,1), results(i,2), results(i,3));
end

function J = cost_function(Uvec, x0, x_ref, Phi, Gamma, Q, R, Np, Nc, u_prev)

nx = length(x0);
nu = length(u_prev);

x = x0;
J = 0;

for i = 1:Np

    if i <= Nc
        u = Uvec((i-1)*nu+1:i*nu);
    else
        u = Uvec((Nc-1)*nu+1:Nc*nu);
    end

    %% Prediction
    x = Phi*x + Gamma*u;

    %% Cost
    J = J + (x - x_ref)'*Q*(x - x_ref) + u'*R*u;

    %% Move suppression
    du = u - u_prev;
    J = J + du'*0.5*du;   

    u_prev = u;

end
end