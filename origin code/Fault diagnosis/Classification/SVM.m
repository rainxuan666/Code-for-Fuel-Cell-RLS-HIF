clear all; close all; clc;

%% Data
data = readmatrix('data.xlsx');
labels = data(:, 1);
X = data(:, 2:10);
X = normalize(X);


cv = cvpartition(labels, 'HoldOut', 0.4);
X_train = X(training(cv), :);
y_train = labels(training(cv));
X_test  = X(test(cv), :);
y_test  = labels(test(cv));

classNames = {'1','2','3'};
numClasses = numel(classNames);

%% One-vs-All
template = templateSVM('KernelFunction', 'gaussian', ...
                       'BoxConstraint', 1, ...
                       'KernelScale',   'auto', ...
                       'Standardize',   true);

tic;
svm_model = fitcecoc(X_train, y_train, ...
                     'Learners', template, ...
                     'Coding',   'onevsall', ...
                     'Verbose',   0);
elapsedTime = toc;
fprintf('Training: %.4f s\n', elapsedTime);

%% Precision
train_pred    = predict(svm_model, X_train);
train_accuracy = sum(train_pred == y_train) / numel(y_train);
fprintf('Training samples: %.2f%%\n', train_accuracy * 100);

test_pred    = predict(svm_model, X_test);
test_accuracy = sum(test_pred == y_test) / numel(y_test);
fprintf('Test samples: %.2f%%\n', test_accuracy * 100);

%% Confusion Matrix
conf_mat = confusionmat(y_test, test_pred);
figure('Name','Confusion Matrix');
confusionchart(conf_mat, classNames);
title('Confusion Matrix');

%%  Precision / Recall / F1 


[precision, recall, f1] = computePRF1(conf_mat, numClasses);


macro_precision = mean(precision);
macro_recall    = mean(recall);
macro_f1        = mean(f1);


classSupport = sum(conf_mat, 2);          
totalSamples = sum(classSupport);
weighted_precision = sum(precision .* classSupport) / totalSamples;
weighted_recall    = sum(recall    .* classSupport) / totalSamples;
weighted_f1        = sum(f1        .* classSupport) / totalSamples;


fprintf('\n====== Classification Performance ======\n');
fprintf('%-10s %10s %10s %10s %10s\n', 'label', 'Precision', 'Recall', 'F1-Score', 'Support');
fprintf('%s\n', repmat('-', 1, 55));
for i = 1:numClasses
    fprintf('%-10s %10.4f %10.4f %10.4f %10d\n', ...
        classNames{i}, precision(i), recall(i), f1(i), classSupport(i));
end
fprintf('%s\n', repmat('-', 1, 55));
fprintf('%-10s %10.4f %10.4f %10.4f %10d\n', ...
    'Macro Avg',   macro_precision,    macro_recall,    macro_f1,    totalSamples);
fprintf('%-10s %10.4f %10.4f %10.4f %10d\n', ...
    'Weighted Avg', weighted_precision, weighted_recall, weighted_f1, totalSamples);



%% Precision / Recall / F1
function [precision, recall, f1] = computePRF1(cm, numClasses)
    precision = zeros(numClasses, 1);
    recall    = zeros(numClasses, 1);
    f1        = zeros(numClasses, 1);
    for i = 1:numClasses
        TP = cm(i, i);
        FP = sum(cm(:, i)) - TP;
        FN = sum(cm(i, :)) - TP;

        if (TP + FP) == 0
            precision(i) = 0;
        else
            precision(i) = TP / (TP + FP);
        end

        if (TP + FN) == 0
            recall(i) = 0;
        else
            recall(i) = TP / (TP + FN);
        end

        if (precision(i) + recall(i)) == 0
            f1(i) = 0;
        else
            f1(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i));
        end
    end
end