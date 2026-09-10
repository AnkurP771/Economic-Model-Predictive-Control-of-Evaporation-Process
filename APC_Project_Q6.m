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
Cp = 0.07; lambda = 38.5;
UA2 = 6.84;
T3 = 0.507*x(2) + 55;
Q200 = UA2*(T3 - d(5)) / (1 + UA2/(2*Cp*u(2)));
F5 = Q200/lambda;
end

 
%% STEADY STATE
 

x_ss = [25; 50.57];

 
%% LINEAR MODEL
 

syms X2 P2 P100 F200 F1 X1 F3 T1 T200

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

sys_d = c2d(ss(A,Bu,eye(2),0),Ts);

Phi = sys_d.A;
Gamma = sys_d.B;

 
%% COMMON SETTINGS
 

N_sim = 400;
nx = 2;
nu = 2;

Np = 20;
Nc = 5;

Q = diag([1 0.5]);
R = diag([1 1]);

u_min = [0;0];
u_max = [400;400];

case_type = input("1: Disturbance, 2: Setpoint = ");

 
%% STORAGE
 
X_L = zeros(nx,N_sim);
X_N = zeros(nx,N_sim);

U_L = zeros(nu,N_sim);
U_N = zeros(nu,N_sim);

Yref = zeros(nx,N_sim);

X_L(:,1) = x_ss;
X_N(:,1) = x_ss;

u_prev_L = zeros(nu,1);
u_prev_N = zeros(nu,1);

 
%% MAIN LOOP
 

for k = 1:N_sim-1

    d = d_ss;

    if case_type==1
        if k>=200 && k<300
            d(1)=1.2*d_ss(1);
        elseif k>=300
            d(1)=0.8*d_ss(1);
        end
    end

    x_ref = x_ss;

    if case_type==2
        if k>=200 && k<300
            x_ref=1.3*x_ss;
        elseif k>=300
            x_ref=0.8*(1.3*x_ss);
        end
    end

    Yref(:,k)=x_ref;

    %%  LMPC 
    
    x_dev = X_L(:,k)-x_ss;

    objL = @(Uvec) cost_L(Uvec,x_dev,x_ref-x_ss,Phi,Gamma,Q,R,Np,Nc,u_prev_L);

    U0 = repmat(u_prev_L,Nc,1);

    Uopt = fmincon(objL,U0,[],[],[],[],...
        repmat(u_min-u_ss,Nc,1),...
        repmat(u_max-u_ss,Nc,1));

    uL_dev = Uopt(1:nu);
    uL = uL_dev + u_ss;

    U_L(:,k)=uL;

    dx = f(X_L(:,k),uL,d);
    X_L(:,k+1)=X_L(:,k)+dx*Ts;

    u_prev_L = uL_dev;

    %%  NMPC 
    
    xk = X_N(:,k);

    objN = @(Uvec) cost_N(Uvec,xk,x_ref,d,Q,R,Np,Nc,u_prev_N,f,Ts,u_ss);

    U0 = repmat(u_prev_N,Nc,1);

    Uopt = fmincon(objN,U0,[],[],[],[],...
        repmat(u_min-u_ss,Nc,1),...
        repmat(u_max-u_ss,Nc,1));

    uN_dev = Uopt(1:nu);
    uN = uN_dev + u_ss;

    U_N(:,k)=uN;

    dx = f(X_N(:,k),uN,d);
    X_N(:,k+1)=X_N(:,k)+dx*Ts;

    u_prev_N = uN_dev;

end

 
%% PLOTS
 

t=1:N_sim;

figure
subplot(2,1,1)
plot(t,X_L(1,:),'b',t,X_N(1,:),'g',t,Yref(1,:),'r--')
legend('LMPC','NMPC','Ref')
title('X2 Comparison')

subplot(2,1,2)
plot(t,X_L(2,:),'b',t,X_N(2,:),'g',t,Yref(2,:),'r--')
legend('LMPC','NMPC','Ref')
title('P2 Comparison')

figure
subplot(2,1,1)
stairs(t,U_L(1,:),'b')
hold on
stairs(t,U_N(1,:),'g')
legend('LMPC','NMPC')
title('P100')

subplot(2,1,2)
stairs(t,U_L(2,:),'b')
hold on
stairs(t,U_N(2,:),'g')
legend('LMPC','NMPC')
title('F200')

 
%% COST FUNCTIONS
 

function J = cost_L(Uvec,x0,xr,Phi,Gamma,Q,R,Np,Nc,u_prev)

nu=length(u_prev);
x=x0; J=0;

for i=1:Np
    if i<=Nc
        u=Uvec((i-1)*nu+1:i*nu);
    else
        u=Uvec((Nc-1)*nu+1:Nc*nu);
    end

    x=Phi*x+Gamma*u;

    J=J+(x-xr)'*Q*(x-xr)+u'*R*u;

    du=u-u_prev;
    J=J+du'*0.5*du;

    u_prev=u;
end
end


function J = cost_N(Uvec,x0,xr,d,Q,R,Np,Nc,u_prev,f,Ts,u_ss)

nu=length(u_prev);
x=x0; J=0;

for i=1:Np
    if i<=Nc
        u_dev=Uvec((i-1)*nu+1:i*nu);
    else
        u_dev=Uvec((Nc-1)*nu+1:Nc*nu);
    end

    u=u_dev+u_ss;

    dx=f(x,u,d);
    x=x+dx*Ts;

    J=J+(x-xr)'*Q*(x-xr)+u_dev'*R*u_dev;

    du=u_dev-u_prev;
    J=J+du'*2*du;

    u_prev=u_dev;
end
end