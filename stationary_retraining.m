% Create a time series with a change point in the training set, run algorithm on
% the test set. Uncover a quiet period in the data, and then retrain the algorithm
% on that part of the data.
% Run the algorithm again with the newly trained features.
%
% change points: 350, 500, 750, 875, 1430, 1560, 1700

clear

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

% time series with change point in training set (~600 points)
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

% scale down slightly
x1 = 0.5 * x1;

% create training set with new splitting method: stdSplit_mod
% train initially with the beginning of the time series
Ttrain_start1 = 1;	% training set start point
Ttrain_end1 = 500;	% training set end point
[Ytrain1, Ytest1, Y1] = stdSplit_mod(x1, Ttrain_start1, Ttrain_end1);

% load covariance functions for GPs
covfunc = {'covSum', {'covRQiso', 'covConst', 'covNoise'}};
loghyperUnits1 = [1 0 0 0 0]';

% set pseudorandom seeds by name
randnSeed = 2;	% seed normal (Gaussian)
randSeed = 4;	% twister uniform

% GP settings
GPrestarts = 5;

% GPTS training
% Fix performance with pseudorandom seeds
randn('seed', randnSeed);	% call pseudorandoms
rand('twister', randSeed);	% call pseudorandoms

% initialise
% train hyperparameters
loghyperGPTS1 = GPlearn(loghyperUnits1, covfunc, (Ttrain_start1:Ttrain_end1)', Ytrain1, [1 1]', GPrestarts);
%
% train/learn model parameters
[theta_h1, theta_m1, theta_s1] = ...
	bocpdGPTlearn(Ytrain1, covfunc, loghyperGPTS1, [logit(1/50) 1 1]', 0, 1);
% find run length distribution now
[R1, S1, nlml1, Z1, predMeans1, predMed1] = bocpdGPT(Y1, covfunc, theta_m1, theta_h1, [theta_s1, 0]', 1);

% plot results
figure;
[h1, h2] = plotS(S1, x1);
subplot(h1);
ylabel('Time series');
vline(Ttrain_start1, 'r--');
vline(Ttrain_end1, 'r--');
subplot(h2);
ylabel('Prob. mass')
vline(Ttrain_start1, 'r--');
vline(Ttrain_end1, 'r--');

results_old = find(convertToAlert(S1, alertThold));

% now we will retrain on the new quiet part and compare results

% new training interval between quiet period
Ttrain_start2 = 901;	% training set starting point
Ttrain_end2 = 1400;		% training set ending point
[Ytrain2, Ytest2, Y2] = stdSplit_mod(x1, Ttrain_start2, Ttrain_end2);

% train new hyperparameters
loghyperGPTS2 = GPlearn(loghyperUnits1, covfunc, (Ttrain_start2:Ttrain_end2)', Ytrain2, [1 1]', GPrestarts);
% train different model parameters
[theta_h2, theta_m2, theta_s2] = ...
	bocpdGPTlearn(Ytrain2, covfunc, loghyperGPTS2, [logit(1 / 50) 1 1]', 0, 1);
% find run length distribution using updated model training
[R2, S2, nlml2, Z2, predMeans2, predMed2] = bocpdGPT(Y2, covfunc, theta_m2, theta_h2, [theta_s2, 0]', 1);

% plot new results
figure;
[h1, h2] = plotS(S2, x1);
subplot(h1);
ylabel('Time series');
vline(Ttrain_start2, 'r--');
vline(Ttrain_end2, 'r--');
subplot(h2);
ylabel('Prob. mass')
vline(Ttrain_start2, 'r--');
vline(Ttrain_end2, 'r--');

results_new = find(convertToAlert(S2, alertThold));