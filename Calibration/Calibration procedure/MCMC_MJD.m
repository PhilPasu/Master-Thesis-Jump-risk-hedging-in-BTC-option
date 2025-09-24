clear all
close all
clc

%data = simMJDdata(4000);
%Y = data(2:end,1);

%%% Load Hourly Data
data = readtable('C:\Thesis\MCMC\Binance_BTCUSDT_1h.csv');  % Use your hourly data file here

% Parse datetime from timestamp column (adjust column name as needed)
data.datetime = datetime(data.Date, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

% Sort rows by time
data = sortrows(data, 'datetime');

% Ensure price is numeric
data.Close = str2double(string(data.Close)); 
data = data(~isnan(data.Close), :);  % Remove rows with NaN prices

% Resample to daily using the last closing price of each day
[uniqueDays, ~, dayIdx] = unique(floor(datenum(data.datetime)));
dailyPrice = accumarray(dayIdx, data.Close, [], @(x) x(end));  % Take last price of each day

% Create corresponding date vectors
dailyDateNum = uniqueDays;
dailyDate = datetime(dailyDateNum, 'ConvertFrom', 'datenum');
dailyDateD = year(dailyDate) * 10000 + month(dailyDate) * 100 + day(dailyDate);

% Filter by date range
start_date = 20170817;
end_date   = 20240114;
idx = (dailyDateD >= start_date) & (dailyDateD <= end_date);

% Construct the final CrixData: [datenum, price, YYYYMMDD]
CrixData = [dailyDateNum(idx), dailyPrice(idx), dailyDateD(idx)];

% Ensure we have enough data points before calculating returns
if size(CrixData, 1) > 1
    % Replace price2ret with manual log-return calculation
    y = diff(log(CrixData(:, 2)));
    Y = y;
else
    error('Not enough data points for return calculation.');
end

%load wti1.mat
% Y = 100 * ( log(WTI(2:end,2))-log(WTI(1:end-1,2)) );
%Y = 100 * ( log(US_index(2:end))-log(US_index(1:end-1)) );
T = length(Y);
one = ones(T,1);

a=0;
%a = [0 0]';
%A = 25*eye(2);
A = 25;
b = [0 0]';
B = 1*eye(length(b));
c = 1;
C = 0.1;
d = 1;
D = 0.1;
e = 0;
E = 10;
f = 100;
F = 0;
g = 0;
G = 0.01;
k = 2;
K = 40;
tmp=[];
% Starting values
m = a; msum = 0; m2sum = 0;                 %% m=mu
kappa = 0; kappasum = 0; kappa2sum = 0;     %% kappa=-alpha/beta, the theta in equation 1
alpha = b(1); alphasum = 0; alpha2sum = 0;   %%alpha in equation 3
beta = b(2); betasum = 0; beta2sum = 0;     %% beta in eq.3
s2V = 1.e-4; s2Vsum = 0; s2V2sum = 0;  %% sigma_v in eq.3
rho = 0; rhosum = 0; rho2sum = 0;           %% the relation between w1 ad w2
mV = 0; mVsum = 0; mV2sum = 0;        %% mu_v, the param in expoential distr. of Z_v 
                                            %%(jump size in variance
mJ = e; mJsum = 0; mJ2sum = 0;              %%mu_y, the mean of jump size in price Z_y
s2J = 2.834390^2; s2Jsum = 0; s2J2sum = 0;   %% sigma_Y, the variance of jump size in price Z_y 
rhoJ = 0; rhoJsum = 0; rhoJ2sum = 0;        %% rho para in the jump size of price
lambda = 0.040412/365; lambdasum = 0; lambda2sum = 0;  %% jump intensity

V = ones(size(Y))*var(Y); %initial values for variance_t;
Vsum = 0;V2sum = 0;
J = abs(Y) - mean(Y) > 4 * std(Y); %J = data(2:end,3);
Jsum = 0;
XV = exprnd(mV,T,1); % the jump size in price, Z_t^y
XVsum = 0;
X = mvnrnd((mJ + XV*rhoJ), s2J); % the jump process in variance, Z_t^y*dN_t
Xsum = 0;
stdevrho = 0.005;
stdevrhoJ = 0.005;
dfrho = 6.5; 
stdevV = 0.9;
dfV = 4.5;
acceptsumV = zeros(size(V));
acceptsumrho = 0;
acceptsumrhoJ = 0;
acceptsums2V = 0;
Z = ones(T,1);
N =5.e3; % Specifies the total number of draws
n =1.e3; % Specifies the burn-in period
test = zeros(N,4);%%matrix for params
V_path = zeros(N,T);
J_path = zeros(N,T);
X_path = zeros(N,T);
%rho = -0.5; alpha = 0.015; beta = -0.03; s2V = 0.01; rhoJ = -1; mJ = -1; s2J = 4; mV = 1;%rho = -0.5; alpha = 0.015; beta = -0.03; s2V = 0.01; rhoJ = -1; mJ = -1; s2J = 4; mV = 1;
for i = 1:N

    i
    Rho = 1 / (1 - rho^2);   
    V0 = V(1);  
        
    % Draw m(i+1)
    Q = ( Y - X.*J - rho/s2V^0.5.*...
        ( V - [V0;V(1:end-1)]*(1+beta)-alpha - J.*XV ) )./ ([V0;V(1:end-1)] ).^0.5; 
    %W = [1./([V0;V(1:end-1)]).^0.5, [0;Y(1:end-1)]./([V0;V(1:end-1)]).^0.5];
    W = [1./([V0;V(1:end-1)]).^0.5];
    As = inv( inv(A) + 1 / (1-rho^2) * W'*W );
    as = As*( inv(A) * a + 1 / ( 1 - rho^2 ) * W'*Q );
    m = mvnrnd(as',As)';

    % mJ
   
    Es = 1/(T/s2J + 1/E);
    es = Es * (sum( (X - XV*rhoJ)/s2J ) + e/E);
    mJ = normrnd(es,Es^0.5);
    %}
    % s2J
    
    fs = f + T;
    Fs = F + sum((X - mJ - rhoJ*XV).^2);
    s2J = iwishrnd(Fs,fs);
    
    % lambda
    
    ks = k + sum(J);
    Ks = K + T - sum(J);
    lambda = betarnd(ks,Ks); 
    %}
    % J
    %{
    eY1 = Y - Z*m - X;
    eY2 = Y - Z*m;
    eV1 = V - [V0;V(1:end-1)] - alpha - beta*[V0;V(1:end-1)] - XV;
    eV2 = V - [V0;V(1:end-1)] - alpha - beta*[V0;V(1:end-1)];        
    p1 = lambda*exp( -0.5 * ( ((eY1 - (rho/sqrt(s2V))*eV1).^2)./((1-rho^2)*[V0;V(1:end-1)]) + (eV1.^2)./(s2V*[V0;V(1:end-1)]) ) );
    p2 = (1 - lambda) * exp( -0.5 * ( ((eY2 - (rho/sqrt(s2V))*eV2).^2)./((1-rho^2)*[V0;V(1:end-1)]) + (eV2.^2)./(s2V*[V0;V(1:end-1)]) ) );
    p = p1./(p1 + p2);
    tmp=[tmp;[p1 p2 p]];
   
    u = rand(T,1);
    J = double(u < p);
    %}
    Jindex = find(J == 1);
    
    % X
    %{
    X(logical(~J)) = normrnd(mJ + rhoJ*XV(logical(~J)),s2J^0.5)';   
    if ~isempty(Jindex)
        if Jindex(1) == 1
            t = 1;
            eV = V(1) - V0 - alpha - beta*V0 - XV(1);
            eY = Y(1) - Z(1,:)*m;
            L = inv(1/((1 - rho^2)*V0) + 1/s2J);
            l = L * ( (eY - (rho/sqrt(s2V))*eV)/((1 - rho^2)*V0) + (mJ + rhoJ*XV(1))/s2J );
            X(1) = normrnd(l,sqrt(L));
        else
            t = Jindex(1);
            eV = V(t) - V(t-1) - alpha - beta*V(t-1) - XV(t);
            eY = Y(t) - Z(t,:)*m;
            L = inv(1/((1 - rho^2)*V(t-1)) + 1/s2J);
            l = L * ( (eY - (rho/sqrt(s2V))*eV)/((1 - rho^2)*V(t-1)) + (mJ + rhoJ*XV(t))/s2J );
            X(t) = normrnd(l,sqrt(L));
        end
        if length(Jindex) > 1
            for t = Jindex(2:end)'
                eV = V(t) - V(t-1) - alpha - beta*V(t-1) - XV(t);
                eY = Y(t) - Z(t,:)*m;
                L = inv(1/((1 - rho^2)*V(t-1)) + 1/s2J);
                l = L * ( (eY - (rho/sqrt(s2V))*eV)/((1 - rho^2)*V(t-1)) + (mJ + rhoJ*XV(t))/s2J );
                X(t) = normrnd(l,sqrt(L));
            end
        end
    end
    %}
        % --- 1. Return Jump X and Volatility Jump XV (Both Updated Always) ---
    for t = 1:T
        % 1.1 Setup
        if t == 1
            Vlag = V0;
        else
            Vlag = V(t-1);
        end
        eY = Y(t) - Z(t,:) * m;
    
        % --- Update X (always) ---
        eV_new = V(t) - alpha - beta * Vlag - XV(t);  % updated XV used here
        L_inv = 1 / ((1 - rho^2) * Vlag) + 1 / s2J;
        L = 1 / L_inv;
        l = L * ((eY - (rho / sqrt(s2V)) * eV_new) / ((1 - rho^2) * Vlag) + (mJ + rhoJ * XV(t)) / s2J);

        X(t) = normrnd(l, sqrt(L));
    end
    %}
    
    % sigma^2 (V)
    cs = c + T;
    Cs = C + sum((Y - m - J .* X).^2);
    v = 1 / gamrnd(cs/2, 2/Cs);  % Inverse Gamma sampling
    V = ones(T,1)*v;
    
    % --- 2. Store latent paths ---
    J_path(i,:) = J;
    X_path(i,:) = X;
    XV_path(i,:) = XV;
    V_path(i,:) = V * 365;
    test(i,:) = [(m * 365) mJ s2J (lambda * 365)];
    [(m * 365) mJ s2J (lambda * 365)]
end

% Params report
param_names = {'m', 'mJ', 's2J', 'lambda'};
num_params = length(param_names);

for k = 1:num_params
    param_vector = test((n + 1):end, k); % Extract samples of the k-th parameter

    mean_val = mean(param_vector);
    std_val = std(param_vector);
    CI = quantile(param_vector, [0.025 0.975]);  % 95% credible interval

    % Display results
    fprintf('%s\n', param_names{k});
    fprintf('Mean: %.6f, Std: %.6f, 99%% CI: [%.6f, %.6f]\n\n', ...
        mean_val, std_val, CI(1), CI(2));
end

% V0 and VT
V0_vec = V_path((n + 1):end, 1);
VT_vec = V_path((n + 1):end, end);

mean_V0 = mean(V0_vec);
std_V0 = std(V0_vec);
CI_V0 = quantile(V0_vec, [0.025 0.975]);

fprintf('V0 (initial)\nMean: %.6f, Std: %.6f, 99%% CI: [%.6f, %.6f]\n\n', ...
    mean_V0, std_V0, CI_V0(1), CI_V0(2));

mean_VT = mean(VT_vec);
std_VT = std(VT_vec);
CI_VT = quantile(VT_vec, [0.025 0.975]);

fprintf('V_T (last)\nMean: %.6f, Std: %.6f, 99%% CI: [%.6f, %.6f]\n', ...
    mean_VT, std_VT, CI_VT(1), CI_VT(2));


% Parameter Names and Posterior Means
figure('Name','Parameter Paths Across MCMC Iterations')

subplot(4,3,1); plot(test(:,1)); title('m'); grid on
subplot(4,3,2); plot(test(:,2)); title('mJ'); grid on
subplot(4,3,3); plot(test(:,3)); title('s2J'); grid on
subplot(4,3,4); plot(test(:,4)); title('lambda'); grid on

sgtitle('MCMC Parameter Traces')
saveas(gcf, 'C:\Thesis\Params trace MJD.png')

TS = CrixData(2:end,1);

% ===========================
% Time Series Decomposition Plots
% ===========================
% Plot Y (Log Returns)
figure
plot(TS, Y, 'LineWidth', 1.2)
datetick('x', 'mmmyy', 'keeplimits', 'keepticks')
title('Y (Log Returns)')
xlabel('Date')
ylabel('Log Return')
grid on
saveas(gcf, 'C:\Thesis\logR plot.png')

% Plot Jump Indicator J
figure
stem(TS, J, 'filled')
datetick('x', 'mmmyy', 'keeplimits', 'keepticks')
ylim([-0.1, 1.1])
title('Realized Jump Indicator J (1 = Jump)')
xlabel('Date')
ylabel('J')
grid on
saveas(gcf, 'C:\Thesis\Jump plot MJD.png')


% Plot E[X^J] (Jump Size in Return)
figure
plot(TS, sum(X_path((n+1):end, :), 1)/(N-n), 'LineWidth', 1.2)
datetick('x', 'mmmyy', 'keeplimits', 'keepticks')
title('E[X^J] (Jump Size in Return)')
xlabel('Date')
ylabel('E[X^J]')
grid on
saveas(gcf, 'C:\Thesis\Jump size plot MJD.png')

% Expectation Calculations
eX = sum(X_path((n+1):end, :), 1)'/(N-n);
eXV = sum(XV_path((n+1):end, :), 1)'/(N-n);
eJ = round(sum(J_path((n+1):end, :), 1)'/(N-n));
eV = sum(V_path((n+1):end, :), 1)'/((N-n) * 365);

% Decomposed Jump Contributions
Jump_P = eX .* eJ;    % Return Jump
Jump_V = eXV .* eJ;   % Volatility Jump

% Residual Calculation for MJD
Sig = sqrt(eV);
ResY = (Y(2:end) - m - Jump_P(2:end)) ./ Sig(1:end-1);

% GARCH estimation
Mdl = garch(1,1);
EstMdl = estimate(Mdl, Y);
vG = infer(EstMdl, Y);
Res_G = Y ./ sqrt(vG);

% MSE Comparison
MSE_MJD = mse(eV * 365, Y.^2)
MSE_G = mse(vG * 365, Y.^2)

% Sort the residuals
residuals = sort(ResY);  % Or Res_G, or Y

% Get theoretical quantiles
n = length(residuals);
theoretical_q = norminv(((1:n) - 0.5) / n, 0, 1);  % Standard normal

% Compute MSE between quantiles
qq_mse = mean((residuals - theoretical_q').^2)

% Create a new figure for MJD Variance Estimation
figure('Name', 'Variance Estimation: MJD', 'Color', 'w');

% Plot the estimated variance
plot(TS, eV * 365, 'LineWidth', 1.5, 'DisplayName', 'E[V] (MJD)');
datetick('x', 'mmmyy', 'keeplimits', 'keepticks');
xlabel('Date', 'FontSize', 11);
ylabel('Variance', 'FontSize', 11);
title('Estimated Variance under MJD Model', 'FontSize', 13);
grid on;

% Save the figure as PNG
saveas(gcf, 'C:\Thesis\Variance plot MJD.png');


% --- QQ Plot: MJD ---
figure('Name','QQ Plot: MJD')
qqplot(ResY)
xlim([-4, 4])
ylim([-4, 4])
title('QQ Plot: MJD')
saveas(gcf, 'C:\Thesis\qqplot MJD.png')

% --- Probability Plot: MJD ---
figure('Name','Probability Plot: MJD')
normplot(ResY)
xlim([-4, 4])
title('Probability Plot: MJD')
saveas(gcf, 'C:\Thesis\probplot MJD.png')

% ===========================
% Save All Variables
% ===========================

save('allVars_mjd_daily')


