clc
clear
close all

%% PARAMETERS

M  = 20;
C  = 4;
Cp = 0.07;
lambda = 38.5;

UA2 = 6.84;

% Disturbances
F1   = 10;
X1   = 5;
F3   = 50;
T1   = 40;
T200 = 25;

d_ss = [F1 X1 F3 T1 T200];

% Inputs
u_ss = [193.45; 207.33];

% Sampling
Ts = 1;

%% NONLINEAR MODEL

f = @(x,u,d) [
(1/M)*( d(1)*d(2) - (d(1)-F4_fun(x,u,d))*x(1) );
(1/C)*( F4_fun(x,u,d) - F5_fun(x,u,d) )
];

%% FUNCTIONS

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

%% STEADY STATE

x_ss = [25; 50.57];

%% MPC PARAMETERS

Q = diag([0.92 0.3]);
R = diag([1 1]);

u_min = [0; 0];
u_max = [400; 400];

nu = 2;

%% TESTING HORIZONS

Np_list = [10 20 30];
Nc_list = [3 5 10];

results_nmpc = zeros(length(Np_list), length(Nc_list));

N_sim = 300;

%% MAIN ANALYSIS LOOP

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

    %% DISTURBANCE
    d = d_ss;
    if k >= 100
        d(1) = 1.2*d_ss(1);
    end
    
    %% SETPOINT
    x_ref = x_ss;
    
    %% CURRENT STATE
    xk = X(:,k);
    
    %% OPTIMIZATION
    
    obj = @(Uvec) cost_function_nmpc(Uvec, xk, x_ref, d, ...
                                    Q, R, Np, Nc, u_prev, f, Ts, u_ss);
    
    U0 = zeros(nu*Nc,1);

    lb = repmat(u_min - u_ss, Nc,1);
    ub = repmat(u_max - u_ss, Nc,1);

    U_opt = fmincon(obj, U0, [],[],[],[], lb, ub);

    u_dev_k = U_opt(1:nu);
    u_k = u_dev_k + u_ss;

    u_k = max(min(u_k,u_max),u_min);

    %% APPLY INPUT 
    U(:,k) = u_k;

    %% NONLINEAR PLANT
    dx = f(X(:,k), u_k, d);
    X(:,k+1) = X(:,k) + dx*Ts;

    %% PERFORMANCE COST (THIS WAS MISSING ❗)
    err = X(:,k) - x_ref;
    J_perf = J_perf + err'*Q*err + u_dev_k'*R*u_dev_k;

    %% UPDATE 
    u_prev = u_dev_k;

end

results_nmpc(a,b) = J_perf;

end
end

%% DISPLAY TABLE

fprintf('\n NMPC Performance Table (Lower is Better)\n\n');

fprintf('        Nc=3        Nc=5        Nc=10\n');

for i = 1:length(Np_list)
fprintf('Np=%2d   %8.2f   %8.2f   %8.2f\n', ...
Np_list(i), results_nmpc(i,1), results_nmpc(i,2), results_nmpc(i,3));
end


%% COST FUNCTION

function J = cost_function_nmpc(Uvec, x0, x_ref, d, Q, R, Np, Nc, u_prev, f, Ts, u_ss)

nu = length(u_prev);
x = x0;

J = 0;

for i = 1:Np

    if i <= Nc
        u_dev = Uvec((i-1)*nu+1:i*nu);
    else
        u_dev = Uvec((Nc-1)*nu+1:Nc*nu);
    end

    u = u_dev + u_ss;

    dx = f(x, u, d);
    x = x + dx*Ts;

    J = J + (x - x_ref)'*Q*(x - x_ref);
    J = J + u_dev'*R*u_dev;

    du = u_dev - u_prev;
    J = J + du'*0.1*du;   % small move penalty

    u_prev = u_dev;

end
end