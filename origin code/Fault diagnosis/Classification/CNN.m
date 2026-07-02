clear; clc; close all;

%% Data
filename = 'data.xlsx';
try
    data = readtable(filename);
catch
    error('File not found. Please ensure data1.csv and data2.csv are in the current directory.');
end

rawLabels = data{:, 1};
features  = data{:, 2:10};

%% Data processing


Y          = categorical(rawLabels);
catList    = categories(Y);          
numClasses = numel(catList);
if numClasses ~= 3
    warning('Detected %d classes instead of 3. The code will adapt automatically.', numClasses);
end


mu          = mean(features);
sig         = std(features);
featuresNorm = (features - mu) ./ sig;

numSamples  = size(featuresNorm, 1);
numFeatures = size(featuresNorm, 2);  


X4D = reshape(featuresNorm', [1, numFeatures, 1, numSamples]);

%% training/test
cv       = cvpartition(rawLabels, 'HoldOut', 0.4);
idxTrain = training(cv);
idxTest  = test(cv);

XTrain = X4D(:, :, :, idxTrain);
YTrain = Y(idxTrain);
XTest  = X4D(:, :, :, idxTest);
YTest  = Y(idxTest);

fprintf('Training samples: %d\n', sum(idxTrain));
fprintf('Test samples: %d\n', sum(idxTest));

%% CNN 
layers = buildCNN(numFeatures, numClasses);

%% option
options = trainingOptions('adam',              ...
    'MaxEpochs',       150,                    ...
    'MiniBatchSize',   16,                     ...
    'InitialLearnRate',0.005,                  ...
    'Shuffle',         'every-epoch',          ...
    'Plots',           'training-progress',    ...
    'Verbose',         false,                  ...
    'ValidationData',  {XTest, YTest});

%% Training
disp('========== Training CNN Model ==========');
net   = trainNetwork(XTrain, YTrain, layers, options);

%% test
YPred    = classify(net, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('\nprecision: %.2f%%\n', accuracy * 100);

%% confusionmat
[C, ~] = confusionmat(YTest, YPred, 'Order', catList);
fprintf('Confusion Matrix :\n');
disp(C);

figure('Name','CNN');
cm_chart       = confusionchart(YTest, YPred);
cm_chart.Title = ['CNN Confusion Matrix (Accuracy: ' ...
                   num2str(accuracy*100,'%.1f') '%)'];

%%  Precision / Recall / F1 
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








function layers = buildCNN(numFeatures, numClasses)
    layers = [
        imageInputLayer([1 numFeatures 1], 'Name','input')
        convolution2dLayer([1 3], 8,  'Padding','same','Name','conv1')
        batchNormalizationLayer('Name','bn1')
        reluLayer('Name','relu1')
        convolution2dLayer([1 3], 16, 'Padding','same','Name','conv2')
        batchNormalizationLayer('Name','bn2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(numClasses, 'Name','fc')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','output')
    ];
end


function [precision, recall, f1] = computePRF1(cm, numClasses)
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