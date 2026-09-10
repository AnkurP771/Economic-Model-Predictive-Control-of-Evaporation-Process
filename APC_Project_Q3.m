clc
clear
close all


% PARAMETERS 


M  = 20;        % kg/m
C  = 4;         % kg/kPa
Cp = 0.07;      % kW/kg-min
lambda = 38.5;  % kW/kg-min
lambda_s = 36.6;

UA2 = 6.84;

% Disturbances (steady)
F1   = 10;
X1   = 5;
F3   = 50;
T1   = 40;
T200 = 25;

d_ss = [F1 X1 F3 T1 T200];

% Inputs (steady)
u_ss = [193.45; 207.33];   % [P100, F200]

% Sampling time
Ts = 1;


% NONLINEAR MODEL
f = @(x,u,d) [

% dX2/dt 
(1/M)*( d(1)*d(2) - (d(1)-F4_fun(x,u,d))*x(1) );

%  dP2/dt 
(1/C)*( F4_fun(x,u,d) - F5_fun(x,u,d) )

];


%% SUPPORTING FUNCTIONS


function F4 = F4_fun(x,u,d)

Cp = 0.07;
lambda = 38.5;

% Temperatures
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

%%  STEADY STATE (VERIFY GIVEN VALUES)

x_ss = [25; 50.57];   % given

% disp('Steady State x_ss:')

%% SYMBOLIC LINEARIZATION

syms X2 P2 P100 F200 F1 X1 F3 T1 T200 real

x = [X2; P2];
u = [P100; F200];
d = [F1 X1 F3 T1 T200];

% Define symbolic model

T2 = 0.5616*P2 + 0.3126*X2 + 48.43;
T3 = 0.507*P2 + 55;

T100 = 0.1538*P100 + 90;
UA1 = 0.16*(F1 + F3);

Q100 = UA1*(T100 - T2);
F4 = (Q100 - F1*Cp*(T2 - T1))/lambda;

Q200 = UA2*(T3 - T200) / (1 + UA2/(2*Cp*F200));
F5 = Q200/lambda;

% State equations
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

%% STEP 6 — DISCRETIZATION

sys_d = c2d(sys,Ts);

Phi   = sys_d.A;
Gamma = sys_d.B;

Gamma_u = Gamma(:,1:2);
Gamma_d = Gamma(:,3:end);


%% EIGENVALUES (DISCRETE)

eig_Phi = eig(Phi);



%% NMPC SETUP

N_sim = 400;
nx = 2;
nu = 2;

Q = diag([0.92 0.3]);
R = diag([1 1]);

Np = 20;
Nc = 5;

u_min = [0; 0];
u_max = [400; 400];

%% STORAGE

X = zeros(nx,N_sim);
U = zeros(nu,N_sim);
Y = zeros(nx,N_sim);
Yref = zeros(nx,N_sim);

X(:,1) = x_ss;
Y(:,1) = X(:,1);

%% CASE SELECTION

case_type = input("1: Disturbance Rejection, 2: Setpoint Tracking = ");


% MAIN NMPC LOOP (SAME STRUCTURE)


u_prev = zeros(nu,1);

for k = 1:N_sim-1

    %% DISTURBANCE PROFILE 
    d = d_ss;

    if case_type == 1
        if k>=200 && k<300
            d(1) = 1.1*d_ss(1);   % +10%
        elseif k>=300 && k<400
            d(1) = 0.9*d_ss(1);   % -10%
        end
    end

    %% SETPOINT PROFILE 
    x_ref = x_ss;

    if case_type == 2
        if k>=200 && k<300
            x_ref = 1.1*x_ss;
        elseif k>=300
            x_ref = 0.85*(1.1*x_ss);
        end
    end

    %% CURRENT STATE 
    xk = X(:,k);


    % NMPC OPTIMIZATION


    obj = @(Uvec) cost_function_nmpc(Uvec, xk, x_ref, d, ...
                                    Q, R, Np, Nc, u_prev, f, Ts, u_ss);

    U0 = zeros(nu*Nc,1);

    lb = repmat(u_min - u_ss, Nc,1);
    ub = repmat(u_max - u_ss, Nc,1);

    U_opt = fmincon(obj, U0, [],[],[],[], lb, ub);

    u_dev_k = U_opt(1:nu);
    u_k = u_dev_k + u_ss;

    % Apply constraints
    u_k = max(min(u_k,u_max),u_min);

    %%  APPLY INPUT 
    U(:,k) = u_k;

    %%  NONLINEAR PLANT 
    dx = f(X(:,k), u_k, d);
    X(:,k+1) = X(:,k) + dx*Ts;

    Y(:,k+1) = X(:,k+1);
    Yref(:,k) = x_ref;

    %% UPDATE 
    u_prev = u_dev_k;

end
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

    % Convert to actual input
    u = u_dev + u_ss;

    % NONLINEAR PREDICTION
    dx = f(x, u, d);
    x = x + dx*Ts;

    % COST (TRACKING)
    J = J + (x - x_ref)'*Q*(x - x_ref);

    % INPUT PENALTY
    J = J + u_dev'*R*u_dev;

    % MOVE PENALTY
    du = u_dev - u_prev;
    J = J + du'*0*du;

    u_prev = u_dev;

end
end

% PLOTTING


t = 1:N_sim;

figure;
subplot(2,1,1)
plot(t,X(1,:), 'b', t,Yref(1,:),'r--')
title('X2 (NMPC)')
legend('Output','Reference')

subplot(2,1,2)
plot(t,X(2,:), 'b', t,Yref(2,:),'r--')
title('P2 (NMPC)')
legend('Output','Reference')

figure;
subplot(2,1,1)
stairs(t,U(1,:))
title('P100 (NMPC)')

subplot(2,1,2)
stairs(t,U(2,:))
title('F200 (NMPC)')

