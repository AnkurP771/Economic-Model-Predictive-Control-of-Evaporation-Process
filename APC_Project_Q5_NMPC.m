clc
clear
close all

% parameters 

M  = 20;        
C  = 4;         
Cp = 0.07;      
lambda = 38.5;  
UA2 = 6.84;

% disturbances (steady)
F1   = 10;
X1   = 5;
F3   = 50;
T1   = 40;
T200 = 25;

d_ss = [F1 X1 F3 T1 T200];

% inputs (steady)
u_ss = [193.45; 207.33];  

Ts = 1;

% nonlinear model
f = @(x,u,d) [

(1/M)*( d(1)*d(2) - (d(1)-F4_fun(x,u,d))*x(1) );

(1/C)*( F4_fun(x,u,d) - F5_fun(x,u,d) )

];

function F4 = F4_fun(x,u,d)
Cp = 0.07;
lambda = 38.5;

T2 = 0.5616*x(2) + 0.3126*x(1) + 48.43;

T100 = 0.1538*u(1) + 90;
UA1 = 0.16*(d(1) + d(3));
Q100 = UA1*(T100 - T2);

F4 = (Q100 - d(1)*Cp*(T2 - d(4)))/lambda;
end

function F5 = F5_fun(x,u,d)
Cp = 0.07;
lambda = 38.5;

UA2 = 6.84;
T3 = 0.507*x(2) + 55;

Q200 = UA2*(T3 - d(5)) / (1 + UA2/(2*Cp*u(2)));

F5 = Q200/lambda;
end

% steady state

x_ss = [25; 50.57];

% nmpc q r analysis

N_sim = 400;
nx = 2;
nu = 2;

Np = 20;
Nc = 5;

u_min = [0; 0];
u_max = [400; 400];

% different q and r

Q_set = {
    diag([0.5 0.2])
    diag([1   0.5])
    diag([5   2])
};

R_set = {
    diag([0.5 0.5])
    diag([1   1])
    diag([5   5])
};

Performance = zeros(length(Q_set), length(R_set));

% loop

for qi = 1:length(Q_set)
for ri = 1:length(R_set)

Q = Q_set{qi};
R = R_set{ri};

% storage

X = zeros(nx,N_sim);
U = zeros(nu,N_sim);
Yref = zeros(nx,N_sim);

X(:,1) = x_ss;

u_prev = zeros(nu,1);

J_perf = 0;

% simulation

for k = 1:N_sim-1

    % disturbance
    d = d_ss;
    if k>=200 && k<300
        d(1) = 1.2*d_ss(1);
    elseif k>=300
        d(1) = 0.8*d_ss(1);
    end

    % setpoint
    x_ref = x_ss;
    if k>=200 && k<300
        x_ref = 1.4*x_ss;
    elseif k>=300
        x_ref = 0.8*(1.4*x_ss);
    end

    Yref(:,k) = x_ref;

    xk = X(:,k);

    % optimization
    obj = @(Uvec) cost_function_nmpc(Uvec, xk, x_ref, d, ...
                        Q, R, Np, Nc, u_prev, f, Ts, u_ss);

    U0 = zeros(nu*Nc,1);

    lb = repmat(u_min - u_ss, Nc,1);
    ub = repmat(u_max - u_ss, Nc,1);

    U_opt = fmincon(obj, U0, [],[],[],[], lb, ub);

    u_dev_k = U_opt(1:nu);
    u_k = u_dev_k + u_ss;

    % constraints
    u_k = max(min(u_k,u_max),u_min);

    % apply input 
    U(:,k) = u_k;

    % nonlinear plant 
    dx = f(X(:,k), u_k, d);
    X(:,k+1) = X(:,k) + dx*Ts;

    % performance cost
    err = X(:,k) - x_ref;
    J_perf = J_perf + err'*Q*err + u_dev_k'*R*u_dev_k;

    % update
    u_prev = u_dev_k;

end

Performance(qi,ri) = J_perf;

end
end

% display

disp('nmpc performance matrix (q rows, r columns)')
disp(Performance)

% cost function

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
    J = J + du'*0.1*du;

    u_prev = u_dev;

end
end