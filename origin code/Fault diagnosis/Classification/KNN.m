clear; clc; close all;

%% Data
filename = 'data.xlsx';
try
    data = readtable(filename);
catch
    error('File not found. Please ensure data1.csv and data2.csv are in the current directory.', filename);
end

rawLabels = data{:, 1};
features  = data{:, 2:10};
X = features;

%% label
Y = categorical(rawLabels);
catList    = categories(Y);          
numClasses = numel(catList);
if numClasses ~= 3
    warning('Detected %d classes instead of 3. The code will adapt automatically.', numClasses);
end


holdOutRatio = 0.4;
cv       = cvpartition(rawLabels, 'HoldOut', holdOutRatio);
idxTrain = training(cv);
idxTest  = test(cv);

XTrain = X(idxTrain, :);
YTrain = Y(idxTrain);
XTest  = X(idxTest,  :);
YTest  = Y(idxTest);

fprintf('Training samples: %d\n', sum(idxTrain));
fprintf('Test samples: %d\n', sum(idxTest));

%% Train
k_value = 1;
Mdl = fitcknn(XTrain, YTrain, ...
    'NumNeighbors', k_value,    ...
    'Standardize',  true,       ...
    'Distance',     'euclidean');

%% predict
YPred    = predict(Mdl, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('\n测试集总体准确率: %.2f%%\n', accuracy * 100);

%% Confusion Matrix
[C, order] = confusionmat(YTest, YPred);
fprintf('Confusion Matrix:\n');
disp(C);

figure('Name','KNN');
cm_chart       = confusionchart(YTest, YPred);
cm_chart.Title = ['KNN Confusion Matrix (K=' num2str(k_value) ')'];

%% Precision / Recall / F1 
[precision, recall, f1] = computePRF1(C, numClasses);


macro_p  = mean(precision);
macro_r  = mean(recall);
macro_f1 = mean(f1);


support      = sum(C, 2);
totalSamples = sum(support);
weighted_p   = sum(precision .* support) / totalSamples;
weighted_r   = sum(recall    .* support) / totalSamples;
weighted_f1  = sum(f1        .* support) / totalSamples;


fprintf('\n====== Training CNN Model ======\n');
fprintf('%-10s %10s %10s %10s %10s\n', ...
    'label','Precision','Recall','F1-Score','Support');
fprintf('%s\n', repmat('-',1,55));
for i = 1:numClasses
    fprintf('%-10s %10.4f %10.4f %10.4f %10d\n', ...
        catList{i}, precision(i), recall(i), f1(i), support(i));
end
fprintf('%s\n', repmat('-',1,55));
fprintf('%-10s %10.4f %10.4f %10.4f %10d\n', ...
    'Macro Avg',   macro_p,   macro_r,   macro_f1,   totalSamples);
fprintf('%-10s %10.4f %10.4f %10.4f %10d\n', ...
    'Weighted Avg', weighted_p,weighted_r,weighted_f1,totalSamples);





function [precision, recall, f1] = computePRF1(cm, numClasses)

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