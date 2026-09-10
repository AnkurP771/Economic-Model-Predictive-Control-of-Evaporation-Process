clc
clear
close all

  
%% PARAMETERS 
  

M  = 20;        %% kg/m
C  = 4;         %% kg/kPa
Cp = 0.07;      %% kW/kg-min
lambda = 38.5;  %% kW/kg-min
lambda_s = 36.6;

UA2 = 6.84;

%% Disturbances (steady)
F1   = 10;
X1   = 5;
F3   = 50;
T1   = 40;
T200 = 25;

d_ss = [F1 X1 F3 T1 T200];

%% Inputs (steady)
u_ss = [193.45; 207.33];   %% [P100, F200]

%% Sampling time
Ts = 1;

  
%% NONLINEAR MODEL
  

f = @(x,u,d) [

%%  dX2/dt 
(1/M)*( d(1)*d(2) - (d(1)-F4_fun(x,u,d))*x(1) );

%%  dP2/dt 
(1/C)*( F4_fun(x,u,d) - F5_fun(x,u,d) )

];


function F4 = F4_fun(x,u,d)

Cp = 0.07;
lambda = 38.5;

%Temperatures
T2 = 0.5616*x(2) + 0.3126*x(1) + 48.43;

% Heat transfer
T100 = 0.1538*u(1) + 90;
UA1 = 0.16*(d(1) + d(3));
Q100 = UA1*(T100 - T2);

% Evaporation flow
F4 = (Q100 - d(1)*Cp*(T2 - d(4)))/lambda;

end

function F5 = F5_fun(x,u,d)

Cp = 0.07;
lambda = 38.5;
UA2 = 6.84;

% Temperature
T3 = 0.507*x(2) + 55;

% Heat removal
Q200 = UA2*(T3 - d(5)) / (1 + UA2/(2*Cp*u(2)));

% Condensation flow
F5 = Q200/lambda;

end
 

x_ss = [25; 50.57];   %% given

% disp('Steady State x_ss:')

syms X2 P2 P100 F200 F1 X1 F3 T1 T200 real

x = [X2; P2];
u = [P100; F200];
d = [F1 X1 F3 T1 T200];

%% Define symbolic model 

T2 = 0.5616*P2 + 0.3126*X2 + 48.43;
T3 = 0.507*P2 + 55;

T100 = 0.1538*P100 + 90;
UA1 = 0.16*(F1 + F3);

Q100 = UA1*(T100 - T2);
F4 = (Q100 - F1*Cp*(T2 - T1))/lambda;

Q200 = UA2*(T3 - T200) / (1 + UA2/(2*Cp*F200));
F5 = Q200/lambda;

%% State equations
f1 = (1/M)*(F1*X1 - (F1 - F4)*X2);
f2 = (1/C)*(F4 - F5);

F = [f1; f2];

%% Jacobians

A_sym  = jacobian(F,[X2 P2]);
Bu_sym = jacobian(F,[P100 F200]);
Bd_sym = jacobian(F,[F1 X1 F3 T1 T200]);

subs_vars = [X2 P2 P100 F200 F1 X1 F3 T1 T200];
subs_vals = [x_ss' u_ss' d_ss];

A  = double(subs(A_sym,subs_vars,subs_vals));
Bu = double(subs(Bu_sym,subs_vars,subs_vals));
Bd = double(subs(Bd_sym,subs_vars,subs_vals));

B = [Bu Bd];
  
eig_A = eig(A);

Cmat = eye(2);
Dmat = zeros(2,7);

sys = ss(A,B,Cmat,Dmat); 

sys_d = c2d(sys,Ts);

Phi   = sys_d.A;
Gamma = sys_d.B;

Gamma_u = Gamma(:,1:2);
Gamma_d = Gamma(:,3:end);

eig_Phi = eig(Phi);

N_sim = 400;
nx = 2;
nu = 2;

Np = 20;
Nc = 5;

u_min = [0; 0];
u_max = [400; 400];

%% DIFFERENT Q AND R

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

%% LOOP OVER Q AND R

for qi = 1:length(Q_set)
for ri = 1:length(R_set)

Q = Q_set{qi};
R = R_set{ri};

%% STORAGE
X = zeros(nx,N_sim);
U = zeros(nu,N_sim);
Yref = zeros(nx,N_sim);

X(:,1) = x_ss;

u_prev = zeros(nu,1);

%% SIMULATION

for k = 1:N_sim-1

    %% DISTURBANCE
    d = d_ss;

    if k>=200 && k<300
        d(1) = 1.1*d_ss(1);
    elseif k>=300
        d(1) = 0.9*d_ss(1);
    end

    %% SETPOINT
    x_ref = x_ss;

    if k>=200 && k<300
        x_ref = 1.1*x_ss;
    elseif k>=300
        x_ref = 0.85*(1.1*x_ss);
    end

    Yref(:,k) = x_ref;

    %% CURRENT STATE
    x_dev = X(:,k) - x_ss;

    %% MPC OPTIMIZATION
    obj = @(Uvec) cost_function(Uvec, x_dev, x_ref-x_ss, ...
                                Phi, Gamma_u, Q, R, ...
                                Np, Nc, u_prev);

    U0 = repmat(u_prev, Nc,1);

    lb = repmat(u_min - u_ss, Nc,1);
    ub = repmat(u_max - u_ss, Nc,1);

    U_opt = fmincon(obj, U0, [],[],[],[], lb, ub);

    u_dev = U_opt(1:nu);
    u = u_dev + u_ss;

    %% APPLY INPUT
    U(:,k) = u;

    %% PLANT
    dx = f(X(:,k), u, d);
    X(:,k+1) = X(:,k) + dx*Ts;

    u_prev = u_dev;

end

%% PERFORMANCE (TRACKING ERROR)
err = X - Yref;
Performance(qi,ri) = sum(err(:).^2);

end
end

%% DISPLAY RESULT
disp('Performance Matrix (Q rows, R columns)')
disp(Performance)

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