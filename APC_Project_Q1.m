clc
clear
close all
 
%% PARAMETERS (FROM GIVEN FILE)

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

%% NONLINEAR MODEL

f = @(x,u,d) [

%  dX2/dt 
(1/M)*( d(1)*d(2) - (d(1)-F4_fun(x,u,d))*x(1) );

%  dP2/dt 
(1/C)*( F4_fun(x,u,d) - F5_fun(x,u,d) )

];
 
% SUPPORTING FUNCTIONS

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
 
%% STEADY STATE (VERIFY GIVEN VALUES) 

x_ss = [25; 50.57];   % given

disp('Steady State x_ss:')
disp(x_ss)

%% LINEARIZATION
 

syms X2 P2 P100 F200 F1 X1 F3 T1 T200 real

x = [X2; P2];
u = [P100; F200];
d = [F1 X1 F3 T1 T200];

%  Define symbolic model 

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
 
%% SUBSTITUTE STEADY STATE


subs_vars = [X2 P2 P100 F200 F1 X1 F3 T1 T200];
subs_vals = [x_ss' u_ss' d_ss];

A  = double(subs(A_sym,subs_vars,subs_vals));
Bu = double(subs(Bu_sym,subs_vars,subs_vals));
Bd = double(subs(Bd_sym,subs_vars,subs_vals));

B = [Bu Bd];

disp('A matrix:')
disp(A)

disp('B matrix:')
disp(B)

%% EIGENVALUES 

eig_A = eig(A);

disp('Eigenvalues of A:')
disp(eig_A)

% Stability comment
if all(real(eig_A) < 0)
    disp('System is asymptotically STABLE (continuous)')
else
    disp('System is UNSTABLE (continuous)')
end
 
%% STEP 5 — STATE SPACE MODEL

Cmat = eye(2);
Dmat = zeros(2,7);

sys = ss(A,B,Cmat,Dmat);

%% DISCRETIZATION 

sys_d = c2d(sys,Ts);

Phi   = sys_d.A;
Gamma = sys_d.B;

Gamma_u = Gamma(:,1:2);
Gamma_d = Gamma(:,3:end);

disp('Phi matrix:')
disp(Phi)

disp('Gamma_U matrix:')
disp(Gamma_u)


disp('Gamma_D matrix:')
disp(Gamma_d)


%% EIGENVALUES 

eig_Phi = eig(Phi);

disp('Eigenvalues of Phi:')
disp(eig_Phi)

% Stability check
if all(abs(eig_Phi) < 1)
    disp('System is asymptotically STABLE (discrete)')
else
    disp('System is UNSTABLE (discrete)')
end