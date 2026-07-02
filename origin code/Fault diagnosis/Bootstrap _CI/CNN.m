clear; clc; close all;

%% Data
try
    data_train = readtable('data_train.csv');
    data_test  = readtable('data_test.csv');
catch
    error('File not found. Please ensure data1.csv and data2.csv are in the current directory.');
end

rawTrain = data_train{:, 1};
featTrain = data_train{:, 2:10};
rawTest  = data_test{:, 1};
featTest  = data_test{:, 2:10};

%% data processing


YTrain     = categorical(rawTrain);
YTest      = categorical(rawTest);
catList    = categories(YTrain);
numClasses = numel(catList);
if numClasses ~= 3
    warning('Detected %d classes instead of 3. The code will adapt automatically.', numClasses);
end

numFeatures = size(featTrain, 2);   


mu_tr     = mean(featTrain);
sig_tr    = std(featTrain);
featTrainN = (featTrain - mu_tr) ./ sig_tr;
featTestN  = (featTest  - mu_tr) ./ sig_tr;


nTr  = size(featTrainN, 1);
nTe  = size(featTestN,  1);
XTrain = reshape(featTrainN', [1, numFeatures, 1, nTr]);
XTest  = reshape(featTestN',  [1, numFeatures, 1, nTe]);

fprintf('Training samples: %d\n', nTr);
fprintf('Test samples:     %d\n', nTe);

%% 4. CNN Architecture
layers = buildCNN(numFeatures, numClasses);

%% Options
options = trainingOptions('adam',               ...
    'MaxEpochs',        150,                    ...
    'MiniBatchSize',    16,                     ...
    'InitialLearnRate', 0.005,                  ...
    'Shuffle',          'every-epoch',          ...
    'Plots',            'training-progress',    ...
    'Verbose',          false,                  ...
    'ValidationData',   {XTest, YTest});

%% Train
disp('========== Training CNN Model ==========');
net = trainNetwork(XTrain, YTrain, layers, options);

%% Prediction
YPred    = classify(net, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('\nTest Accuracy: %.2f%%\n', accuracy * 100);

%% Confusion Matrix
[C, ~] = confusionmat(YTest, YPred, 'Order', catList);
fprintf('Confusion Matrix (rows=true, cols=predicted):\n');
disp(C);

figure('Name', 'CNN Confusion Matrix');
cm_chart       = confusionchart(YTest, YPred);
cm_chart.Title = ['CNN Confusion Matrix (Accuracy: ' ...
                   num2str(accuracy * 100, '%.1f') '%)'];

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

%% Functions

function layers = buildCNN(numFeatures, numClasses)
    layers = [
        imageInputLayer([1 numFeatures 1], 'Name', 'input')
        convolution2dLayer([1 3], 8,  'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        convolution2dLayer([1 3], 16, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(numClasses, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];
end

function [precision, recall, f1] = computePRF1(cm, numClasses)
%  Precision, Recall, F1

    precision = zeros(numClasses, 1);
    recall    = zeros(numClasses, 1);
    f1        = zeros(numClasses, 1);
    for i = 1:numClasses
        TP = cm(i, i);
        FP = sum(cm(:, i)) - TP;
        FN = sum(cm(i, :)) - TP;
        if (TP + FP) > 0, precision(i) = TP / (TP + FP); end
        if (TP + FN) > 0, recall(i)    = TP / (TP + FN); end
        denom = precision(i) + recall(i);
        if denom > 0,     f1(i) = 2 * precision(i) * recall(i) / denom; end
    end
end