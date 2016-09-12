% Same tests as previously performed, just taking the logarithm of the time
% series to test performance 
%
clear;

% 

% set seed for pseudorandom normal numbers
% set twister for pseudorandom uniform numbers
randn('seed', 2);
rand('twister', 4);
% create time length
t = linspace(1, 2000, 2000);
T = length(t);

% set alert threshold for changepoints:
% 0.99	conservative
% 0.95 	robust
% 0.90 	liberal
% <0.9	suspect
alertThold = 0.95;

% Create test cases

%% Test case 1 - Autoregressive model with one changepoint in beginning
% change points: 350, 500, 750, 875, 1430, 1560, 1700
x1 = zeros(T, 1);
% initialise points
x1(1) = randn;
x1(2) = randn;

for i = 2:T
	x1(i) = 0.6*x1(i-1) + 0.5*randn;
	if (i > 350 && i < 500)
		x1(i) = 0.6*x1(i-1) + 1.25*randn;
	end
	if (i > 750 && i < 875)
		x1(i) = 0.6*x1(i-1) + 1 + 0.5*randn;
	end
	if (i > 1430 && i < 1560)
		x1(i) = 0.6*x1(i-1) - 0.5 + 0.5*randn;
	end
	if (i > 1700)
		x1(i) = 0.7*x1(i-1) + randn;
	end
end

% shift data up and take log transform for smoothing
x1 = log(x1 + 5);

% create training set on first part of time series
Ttrain_start1 = 1;	% training set starting point
Ttrain_end1 = 500;	% training set end point
[Ytrain1, Ytest1, Y1] = stdSplit_mod(x1, Ttrain_start1, Ttrain_end1);

% set pseudorandom seeds by name
randnSeed = 2;	% normal seed (Gaussian start)
randSeed = 4;	% twister seed for uniform

% Fix performance with pseudorandom seeds
randn('seed', randnSeed);	% call pseudorandoms
rand('twister', randSeed);	% call preudorandoms

%% Test case 2 - Autoregressive with multiple changepoints in beginning
% Change points: 125, 180, 270, 340, 410, 530, 710, 790, 1530, 1615, 1720, 1810, 1870

% time series
x2 = zeros(T, 1);
x2(1) = randn;

for i = 2:T
	x2(i) = 0.3*x2(i-1) + 0.75*randn;
	if (i >= 125 && i <= 180)
		x2(i) = 0.3*x2(i-1) + 1.75*randn;
	end
	if (i >= 270 && i <= 340)
		x2(i) = 0.6*x2(i-1) + 0.75*randn;
	end
	if (i >= 410 && i <= 530)
		x2(i) = 0.3*x2(i-1) + 1 + 0.75*randn;
	end
	if (i >= 710 && i <= 790)
		x2(i) = 0.3*x2(i-1) + 1.4*randn;
	end
	if (i >= 1530 && i <= 1615)
		x2(i) = 0.3*x2(i-1) - 0.75 + 0.75*randn;
	end
	if (i >= 1720 && i <= 1810)
		x2(i) = 0.8*x2(i-1) + 0.75*randn;
	end
	if (i >= 1870)
		x2(i) = 0.3*x2(i-1) + 0.75*randn;
	end
end

% shift data up and take log transform for smoothing
x2 = log(x2 + 5);

% create training and test set
Ttrain_start2 = 1;	% training set starting point
Ttrain_end2 = 500;	% training set ending point
[Ytrain2, Ytest2, Y2] = stdSplit_mod(x2, Ttrain_start2, Ttrain_end2);

%% Test case 3 - GBM
% Change points: 200, 275, 360, 400, 575, 620, 1250, 1465, 1600, 1840

x3 = zeros(T, 1);

x3(1) = randn;

for i = 1:T
	x3(i) = randn;
	if (i >= 200 && i <= 275)
		x3(i) = 1 + randn;
	end
	if (i >= 360 && i <= 400)
		x3(i) = sqrt(2)*randn;
	end
	if (i >= 475 && i <= 620)
		x3(i) = 2*randn;
	end
	if (i >= 1250 && i <= 1465)
		x3(i) = randn - 1;
	end
	if (i >= 1600 && i <= 1840)
		x3(i) = 0.25*randn;
	end
end

% shift data up and apply log transform for smoothing
x3 = log(x3 + 10);

% create training and test set
Ttrain_start3 = 1;	% training set starting point
Ttrain_end3 = 500;	% training set ending point
[Ytrain3, Ytest3, Y3] = stdSplit_mod(x3, Ttrain_start3, Ttrain_end3);

%% GP models now
% initialise GP features
% load covariance functions
covfunc = {'covSum', {'covRQiso', 'covConst', 'covNoise'}};
% initialise hyperparameters
loghyperUnits = [1; 0; 0; 0; 0];
% GP resamples
GPrestarts = 5;

%% Test case 1 training and testing
% hyperparameters
loghyperGPTS1 = GPlearn(loghyperUnits, covfunc, (Ttrain_start1:Ttrain_end1)', Ytrain1, [1 1]', GPrestarts);
% train/learn model parameters
[theta_h1, theta_m1, theta_s1] = ...
	bocpdGPTlearn(Ytrain1, covfunc, loghyperGPTS1, [logit(1/50) 1 1]', 0, 1);
% find run length distribution
[R1, S1, nlml1, Z1, predMeans1, predMed1] = bocpdGPT(Y1, covfunc, theta_m1, theta_h1, [theta_s1, 0]', 1);
% save results
results1 = find(convertToAlert(S1, alertThold));
% print results
fprintf('The detected changepoints are:\n');
disp(results1);

%% Test case 2 training and testing
% hyperparameters
loghyperGPTS2 = GPlearn(loghyperUnits, covfunc, (Ttrain_start2:Ttrain_end2)', Ytrain2, [1 1]', GPrestarts);
% train/learn model parameters
[theta_h2, theta_m2, theta_s2] = ...
	bocpdGPTlearn(Ytrain2, covfunc, loghyperGPTS2, [logit(1/50) 1 1]', 0, 1);
% find run length distribution
[R2, S2, nlml2, Z2, predMeans2, predMed2] = bocpdGPT(Y2, covfunc, theta_m2, theta_h2, [theta_s2, 0]', 1);
% save results
results2 = find(convertToAlert(S2, alertThold));
% print results
fprintf('The detected changepoints are:\n');
disp(results2);

%% Test case 3 hyperparameters
loghyperGPTS3 = GPlearn(loghyperUnits, covfunc, (Ttrain_start3:Ttrain_end3)', Ytrain3, [1 1]', GPrestarts);
% hyperparameters
% train/learn model parameters
[theta_h3, theta_m3, theta_s3] = ...
	bocpdGPTlearn(Ytrain3, covfunc, loghyperGPTS3, [logit(1/50) 1 1]', 0, 1);
% find run length distribution
[R3, S3, nlml3, Z3, predMeans3, predMed3] = bocpdGPT(Y3, covfunc, theta_m3, theta_h3, [theta_s3, 0]', 1);
% save results
results3 = find(convertToAlert(S3, alertThold));
% print results
fprintf('The detected changepoints are:\n');
disp(results3);

%% Plot results
% Case 1
figure;
[h1, h2] = plotS(S1, x1);
subplot(h1);
vline(Ttrain_start1, 'r--');
vline(Ttrain_end1, 'r--');
subplot(h2);
vline(Ttrain_start1, 'r--');
vline(Ttrain_end1, 'r--');

% Case 2
figure;
[h1, h2] = plotS(S2, x2);
subplot(h1);
vline(Ttrain_start2, 'r--');
vline(Ttrain_end2, 'r--');
subplot(h2);
vline(Ttrain_start2, 'r--');
vline(Ttrain_end2, 'r--');

% Case 3
figure;
[h1, h2] = plotS(S3, x3);
subplot(h1);
vline(Ttrain_start3, 'r--');
vline(Ttrain_end3, 'r--');
subplot(h2);
vline(Ttrain_start3, 'r--');
vline(Ttrain_end3, 'r--');