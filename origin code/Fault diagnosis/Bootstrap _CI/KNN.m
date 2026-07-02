clear; clc; close all;

%% Data
try
    data_train = readtable('data_train.csv');
    data_test  = readtable('data_test.csv');
catch
    error('File not found. Please ensure data1.csv and data2.csv are in the current directory.');
end

XTrain     = data_train{:, 2:10};
rawTrain   = data_train{:, 1};
XTest      = data_test{:, 2:10};
rawTest    = data_test{:, 1};

%% Label Preprocessing
YTrain = categorical(rawTrain);
YTest  = categorical(rawTest);

catList    = categories(YTrain);
numClasses = numel(catList);
if numClasses ~= 3
    warning('Detected %d classes instead of 3. The code will adapt automatically.', numClasses);
end

fprintf('Training samples: %d\n', size(XTrain, 1));
fprintf('Test samples:     %d\n', size(XTest,  1));

%% Train
k_value = 1;
Mdl = fitcknn(XTrain, YTrain, ...
    'NumNeighbors', k_value,    ...
    'Standardize',  true,       ...
    'Distance',     'euclidean');

%% Prediction
YPred    = predict(Mdl, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('\nTest Accuracy: %.2f%%\n', accuracy * 100);

%% Confusion Matrix
[C, ~] = confusionmat(YTest, YPred);
fprintf('Confusion Matrix (rows=true, cols=predicted):\n');
disp(C);

figure('Name', 'KNN Confusion Matrix');
cm_chart       = confusionchart(YTest, YPred);
cm_chart.Title = ['KNN Confusion Matrix (K=' num2str(k_value) ')'];

%% Per-Class Precision / Recall / F1
[precision, recall, f1] = computePRF1(C, numClasses);


macro_p  = mean(precision);
macro_r  = mean(recall);
macro_f1 = mean(f1);


support      = sum(C, 2);
totalSamples = sum(support);
weighted_p   = sum(precision .* support) / totalSamples;
weighted_r   = sum(recall    .* support) / totalSamples;
weighted_f1  = sum(f1        .* support) / totalSamples;

fprintf('\n====== Classification Performance ======\n');
fprintf('%-12s %10s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score', 'Support');
fprintf('%s\n', repmat('-', 1, 57));
for i = 1:numClasses
    fprintf('%-12s %10.4f %10.4f %10.4f %10d\n', ...
        catList{i}, precision(i), recall(i), f1(i), support(i));
end
fprintf('%s\n', repmat('-', 1, 57));
fprintf('%-12s %10.4f %10.4f %10.4f %10d\n', 'Macro Avg',    macro_p,    macro_r,    macro_f1,    totalSamples);
fprintf('%-12s %10.4f %10.4f %10.4f %10d\n', 'Weighted Avg', weighted_p, weighted_r, weighted_f1, totalSamples);


%% Function

function [precision, recall, f1] = computePRF1(cm, numClasses)
% Compute per-class Precision, Recall, F1 from confusion matrix
% (rows = true class, cols = predicted class)
    precision = zeros(numClasses, 1);
    recall    = zeros(numClasses, 1);
    f1        = zeros(numClasses, 1);
    for i = 1:numClasses
        TP = cm(i, i);
        FP = sum(cm(:, i)) - TP;
        FN = sum(cm(i, :)) - TP;

        if (TP + FP) > 0
            precision(i) = TP / (TP + FP);
        end
        if (TP + FN) > 0
            recall(i) = TP / (TP + FN);
        end
        denom = precision(i) + recall(i);
        if denom > 0
            f1(i) = 2 * precision(i) * recall(i) / denom;
        end
    end
end