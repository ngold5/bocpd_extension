% stdSplit function modified to split time series at any time sepcified
% Input:
% Y 			time series
% Ttrain_start	start point for training set (int)
% Ttrain_end	end point for training set (int)
% 
% Output:
% Ytrain		training set (time series)
% Ytrest		test set (time series)
% Y 			time series reconstructed

% NOTE
% Works only for single time series
% Next update: make work for vectors of time series (matrix of different elements)

function [Ytrain, Ytest, Y] = stdSplit_mod(Y, Ttrain_start, Ttrain_end)

t = length(Y);

if (Ttrain_start > t)
    error('Ttrain_start is greater than the length of Y, select a shorter time.');
elseif (Ttrain_end > t)
    error('Ttrain_end is greater than the length of Y, select a shorter time.');
end

clear t;

Ytrain = Y(Ttrain_start:Ttrain_end, :);
Ytest1 = Y(1:Ttrain_start - 1, :);
Ytest2 = Y(Ttrain_end + 1:end, :);
Ytest = cat(1, Ytest1, Ytest2);

col_means = mean(Ytrain, 1);
col_stds = std(Ytrain, 0, 1);

Ytrain = cplus(Ytrain, -col_means);
Ytrain = cdiv(Ytrain, col_stds);

Ytest = cplus(Ytest, -col_means);
Ytest = cdiv(Ytest, col_stds);

Y = cplus(Y, -col_means);
Y = cdiv(Y, col_stds);