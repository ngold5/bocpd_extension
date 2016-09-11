% Create a time series with a change point in the training set, run algorithm on
% the test set. Uncover a quiet period in the data, and then retrain the algorithm
% on that part of the data.
% Run the algorithm again with the newly trained features.
%
% change points: 350, 500, 750, 875, 1430, 1560, 1700

clear;

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

%%

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

% save results of change points
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

% save results of change points
results_new = find(convertToAlert(S2, alertThold));

% print results
disp(results_old);
disp(results_new);


%% Second trial: 
% multiple change points in the training set, followed by a quiet period and
% and then a period with multiple change points again. Evaluate performance with
% retraining
%
% Change points: 125, 180, 270, 340, 410, 530, 710, 790, 1530, 1615, 1720, 1810, 1870

% time series
x2 = zeros(2000, 1);
x2(1) = randn;
x2(2) = randn;

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

% create training set with new splitting method: stdSplit_mod
% train initially with the beginning of the time series
Ttrain_start3 = 1;	% training set start point
Ttrain_end3 = 500;	% training set end point
[Ytrain3, Ytest3, Y3] = stdSplit_mod(x2, Ttrain_start3, Ttrain_end3);

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

%% initialise
% train hyperparameters
loghyperGPTS3 = GPlearn(loghyperUnits1, covfunc, (Ttrain_start3:Ttrain_end3)', Ytrain3, [1 1]', GPrestarts);
%
% train/learn model parameters
[theta_h3, theta_m3, theta_s3] = ...
	bocpdGPTlearn(Ytrain3, covfunc, loghyperGPTS3, [logit(1/50) 1 1]', 0, 1);
% find run length distribution now
[R3, S3, nlml3, Z3, predMeans3, predMed3] = bocpdGPT(Y3, covfunc, theta_m3, theta_h3, [theta_s3, 0]', 1);

% plot results
figure;
[h1, h2] = plotS(S3, x2);
subplot(h1);
ylabel('Time series');
vline(Ttrain_start3, 'r--');
vline(Ttrain_end3, 'r--');
subplot(h2);
ylabel('Prob. mass')
vline(Ttrain_start3, 'r--');
vline(Ttrain_end3, 'r--');

% save change points
results_old2 = find(convertToAlert(S3, alertThold));

%%
% now to do retraining on the quiet period detected by the algorithm and then 
% test the results running the algorithm from the beginning

Ttrain_start4  = 825;
Ttrain_end4 = 1325;

[Ytrain4, Ytest4, Y4] = stdSplit_mod(x2, Ttrain_start4, Ttrain_end4);
%%
% initialise training
% train hyperparameters
loghyperGPTS4 = GPlearn(loghyperUnits1, covfunc, (Ttrain_start4:Ttrain_end4)', Ytrain4, [1 1]', GPrestarts);
% train/learn model parameters now
[theta_h4, theta_m4, theta_s4] = ...
	bocpdGPTlearn(Ytrain4, covfunc, loghyperGPTS4, [logit(1/50) 1 1]', 0, 1);
% find runlength distribution
[R4, S4, nlml4, Z4, predMeans4, predMed4] = bocpdGPT(Y4, covfunc, theta_m4, theta_h4, [theta_s4, 0]', 1);

% plot results now
figure;
[h1, h2] = plotS(S4, x2);
subplot(h1);
ylabel('Time series');
vline(Ttrain_start4, 'r--');
vline(Ttrain_end4, 'r--');
subplot(h2);
ylabel('Prob mass');
vline(Ttrain_start4, 'r--');
vline(Ttrain_end4, 'r--');

% save change points
results_new2 = find(convertToAlert(S4, alertThold));

% display change points
disp(results_old2);
disp(results_new2);

%% Test with GBM

% create time series with GBM 
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
%%
Ttrain_start5 = 1;
Ttrain_end5 = 500;
[Ytrain5, Ytest5, Y5] = stdSplit_mod(x3, Ttrain_start5, Ttrain_end5);
%%
% initialise training
% train hyperparameters
loghyperGPTS5 = GPlearn(loghyperUnits1, covfunc, (Ttrain_start5:Ttrain_end5)', Ytrain5, [1 1]', GPrestarts);
% train/learn model parameters now
[theta_h5, theta_m5, theta_s5] = ...
	bocpdGPTlearn(Ytrain5, covfunc, loghyperGPTS5, [logit(1/50) 1 1]', 0, 1);
% find runlength distribution
[R5, S5, nlml5, Z5, predMeans5, predMed5] = bocpdGPT(Y5, covfunc, theta_m5, theta_h5, [theta_s5, 0]', 1);

% plot results now
figure;
[h1, h2] = plotS(S5, x3);
subplot(h1);
ylabel('Time series');
vline(Ttrain_start5, 'r--');
vline(Ttrain_end5, 'r--');
subplot(h2);
ylabel('Prob mass');
vline(Ttrain_start5, 'r--');
vline(Ttrain_end5, 'r--');

% save change points
results_old3 = find(convertToAlert(S5, alertThold));

disp(results_old3);

%% Retrain on identified stationary part

Ttrain_start6 = 650;
Ttrain_end6 = 1150;
[Ytrain6, Ytest6, Y6] = stdSplit_mod(x3, Ttrain_start6, Ttrain_end6);
%%
% initialise training
% train hyperparameters
loghyperGPTS6 = GPlearn(loghyperUnits1, covfunc, (Ttrain_start6:Ttrain_end6)', Ytrain6, [1 1]', GPrestarts);
% train/learn model parameters now
[theta_h6, theta_m6, theta_s6] = ...
	bocpdGPTlearn(Ytrain6, covfunc, loghyperGPTS6, [logit(1/50) 1 1]', 0, 1);
% find runlength distribution
[R6, S6, nlml6, Z6, predMeans6, predMed6] = bocpdGPT(Y6, covfunc, theta_m6, theta_h6, [theta_s6, 0]', 1);

% plot results now
figure;
[h1, h2] = plotS(S6, x3);
subplot(h1);
ylabel('Time series');
vline(Ttrain_start6, 'r--');
vline(Ttrain_end6, 'r--');
subplot(h2);
ylabel('Prob mass');
vline(Ttrain_start6, 'r--');
vline(Ttrain_end6, 'r--');

% save change points
results_new3 = find(convertToAlert(S6, alertThold));

disp(results_new3);

%% Helen's bet with Huaxiong
% Take time series x2 and move the beginning period to the end, and now see 
% the results when you do this

Y4_hbet_begin = Ytrain4;
Y4_hbet_end = Y4(1:Ttrain_start4-1);
Y4_hbet_middle = Y4(Ttrain_end4+1:end);
Y4_hbet = [Y4_hbet_begin; Y4_hbet_middle; Y4_hbet_end];
%%
loghyperGPTS4_hbet = ...
	GPlearn(loghyperUnits1, covfunc, (Ttrain_start3:Ttrain_end3+1)', Ytrain4, [1 1]', GPrestarts);
% train/learn model parameters now
[theta_h4_hbet, theta_m4_hbet, theta_s4_hbet] = ...
	bocpdGPTlearn(Ytrain4, covfunc, loghyperGPTS4_hbet, [logit(1/50) 1 1]', 0, 1);
%
[R4_hbet, S4_hbet, Z4_hbet, predMeans4_hbet, predMed4_hbet] = ...
	bocpdGPT(Y4_hbet, covfunc, theta_m4_hbet, theta_h4_hbet, [theta_s4_hbet, 0]', 1);

% plot results now
figure;
[h1, h2] = plotS(S4_hbet, Y4_hbet);
subplot(h1);
ylabel('Time series');
vline(Ttrain_start3, 'r--');
vline(Ttrain_end3, 'r--');
subplot(h2);
ylabel('Prob mass');
vline(Ttrain_start3, 'r--');
vline(Ttrain_end3, 'r--');

% save change points
results_hbet = find(convertToAlert(S4_hbet, alertThold));

disp(results_hbet);	

%% Nathan second bet with Huaxiong: GBM perform worse (N) than GP (HH)
% test case now with GBM predictive model rather than Guassian Process

% use same training data from first GBM test case, training on nonstationary 
% beginning of the time series

% train model
[hazard_paramsGBM, model_paramsGBM] = learn_IFM(Ytrain5, true);
%% find run length distribution
[R7, S7, nlml7, Z7, predMeans7] = ...
	bocpd_sparse(hazard_paramsGBM', model_paramsGBM', Y5, 'logistic_h', 'gaussian1D', 1e-7);

figure;
[h1, h2] = plotS(S7, x3);
subplot(h1);
ylabel('Time series');
vline(Ttrain_start5, 'r--');
vline(Ttrain_end5, 'r--');
subplot(h2);
ylabel('Prob. Mass');
vline(Ttrain_start5, 'r--');
vline(Ttrain_end5, 'r--');

% save change points in this case
results_gbm_hbet = find(convertToAlert(S7, alertThold));

% print results
disp(results_gbm_hbet);

%% now retrain model on quiet period 

% train model
[hazard_paramsGBM_new, model_paramsGBM_new] = learn_IFM(Ytrain6, true);
% find run length distribution
[R8, S8, nlml8, Z8, predMeans8] = ...
	bocpd_sparse(hazard_paramsGBM_new', model_paramsGBM_new', Y6, 'logistic_h', 'gaussian1D', 1e-7);

figure;
[h1, h2] = plotS(S8, x3);
subplot(h1);
ylabel('Time series');
vline(Ttrain_start5, 'r--');
vline(Ttrain_end5, 'r--');
subplot(h2);
ylabel('Prob. Mass');
vline(Ttrain_start5, 'r--');
vline(Ttrain_end5, 'r--');

% save change points in this case
results_gbm_hbet_new = find(convertToAlert(S7, alertThold));

% print results
disp(results_gbm_hbet_new);