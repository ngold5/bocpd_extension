% Create a time series with a change point in the training set, run algorithm on
% the test set. Uncover a quiet period in the data, and then retrain the algorithm
% on that part of the data.
% Run the algorithm again with the newly trained features.
%
% change points: 350, 500, 750, 925, 1400, 1560, 1700

clear

% set seed for pseudorandom normal numbers
% set twister for pseudorandom uniform numbers
randn('seed', 2);
rand('twister', 4);
% create time length
t = linspace(1, 2000, 2000);
T = length(t);
% time series with change point in training set (~600 points)
x1 = zeros(T, 1);

x1(1) = randn;
x1(2) = randn;

for i = 2:length(T)
	x1(i) = 0.6*x1(i-1) + 0.5*randn;
	if (i > 350 && i < 500)
		x1(i) = 0.6*x1(i-1) + 1.25*randn;
	end
	if (i > 750 && i < 925)
		x1(i) = 0.6*x1(i-1) + 1 + 0.5*randn;
	end
	if (i > 1400 && i < 1560)
		x1(i) = 0.6*x1(i-1) - 0.5 + 0.5*randn;
	end
	if (i > 1700)
		x1(i) = 0.7*x1(i-1) + randn;
	end
end

% scale down slightly
x1 = 0.5 * x1;

% normalise data by applying linear transformation and then log transform
x1 = log(x1 + 5);

% create training set with new splitting method: stdSplit_mod
% train initially with the beginning of the time series
Ttrain_start = 1;	% training set start point
Ttrain_end = 600;	% training set end point
[Ytrain, Ytest, Y] = stdSplit_mod(x1, Ttrain_start, Ttrain_end);

% load covariance functions for GPs
covfunc = {'covSum', {'covRQiso', 'covConst', 'covNoise'}};
loghyperUnits = [1 0 0 0 0]';

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
loghyperGPTS = GPlearn(loghyperUnits, covfunc, (Ttrain_start:Ttrain_end)', Ytrain, [1 1]', GPrestarts);
% train/learn model parameters
[theta_h, theta_m, theta_m] = ...
	bocpdGPTlearn(Ytrain, covfunc, loghyperGPTS, [logit(1/50) 1 1]', 0, 1);
% find run length distribution now
[R, S, nlml, Z, predMeans, predMed] = bocpdGPT(Y, covfunc, theta_m, theta_h, [theta_s, 0]', 1));

% plot results
figure;
[h1, h2] = plotS(S, x1);
subplot(h1);
ylabel('Time series');
vline(Ttrain_start, 'r--');
vline(Ttrain_end, 'r--');
subplot(h2);
ylabel('Prob. mass')
vline(Ttrain_start, 'r--');
vline(Ttrain_end, 'r--');