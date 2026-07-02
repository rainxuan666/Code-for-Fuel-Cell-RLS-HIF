clear; clc; close all;

%% data
data   = readmatrix('data.xlsx');
labels = data(:, 1);
X      = data(:, 2:10);
X      = normalize(X);

classes    = unique(labels);
K          = length(classes);   
classNames = string(classes);


cv      = cvpartition(labels, 'HoldOut', 0.4);
X_train = X(training(cv), :);
y_train = labels(training(cv));
X_test  = X(test(cv), :);
y_test  = labels(test(cv));

%%  RBF 
rbf_kernel = @(x1, x2, sigma) ...
    exp(-pdist2(x1, x2, 'euclidean').^2 / (2 * sigma^2));
sigma = 1.0;

%% One-vs-Rest
N_train       = size(X_train, 1);
w_all         = zeros(N_train, K);
relevance_idx = cell(K, 1);

tic;
for k = 1:K
    t   = double(y_train == classes(k));
    Phi = rbf_kernel(X_train, X_train, sigma);

    alpha = ones(N_train, 1) * 1e-3;
    w     = zeros(N_train, 1);

    max_iter = 300;
    tol      = 1e-4;

    for iter = 1:max_iter
        y_sig = 1 ./ (1 + exp(-Phi * w));
        R     = diag(y_sig .* (1 - y_sig));
        H     = Phi' * R * Phi + diag(alpha);
        g     = Phi' * (t - y_sig) - alpha .* w;
        dw    = H \ g;
        w     = w + dw;
        gamma = 1 - alpha .* diag(inv(H));
        alpha = gamma ./ (w .^ 2 + eps);
        if norm(dw) < tol, break; end
    end

    w_all(:, k)      = w;
    relevance_idx{k} = find(abs(w) > 1e-2);
end
elapsedTime = toc;
fprintf('Training Time: %.4f s\n', elapsedTime);


predictRVM = @(Xq, Xr, w, sig, cls) local_predict(Xq, Xr, w, sig, cls);


y_pred = predictRVM(X_test, X_train, w_all, sigma, classes);


accuracy = mean(y_pred == y_test);
fprintf('precision: %.2f%%\n', accuracy * 100);

%% Confusion Matrix
confmat = confusionmat(y_test, y_pred, 'Order', classes);
fprintf('\nConfusion Matrix:\n');
disp(confmat);

figure('Name','Confusion Matrix');
confusionchart(confmat, classNames);
title('Confusion Matrix');

%% Precision / Recall / F1 
[precision, recall, f1] = computePRF1(confmat, K);


macro_p  = mean(precision);
macro_r  = mean(recall);
macro_f1 = mean(f1);


support      = sum(confmat, 2);          
totalSamples = sum(support);
weighted_p   = sum(precision .* support) / totalSamples;
weighted_r   = sum(recall    .* support) / totalSamples;
weighted_f1  = sum(f1        .* support) / totalSamples;


fprintf('\n====== Classification Performance ======\n');
fprintf('%-10s %10s %10s %10s %10s\n', 'Label','Precision','Recall','F1-Score','Support');
fprintf('%s\n', repmat('-',1,55));
for i = 1:K
    fprintf('%-10s %10.4f %10.4f %10.4f %10d\n', ...
        classNames(i), precision(i), recall(i), f1(i), support(i));
end
fprintf('%s\n', repmat('-',1,55));
fprintf('%-10s %10.4f %10.4f %10.4f %10d\n','Macro Avg',  macro_p,   macro_r,   macro_f1,   totalSamples);
fprintf('%-10s %10.4f %10.4f %10.4f %10d\n','Weighted Avg',weighted_p,weighted_r,weighted_f1,totalSamples);





function y_out = local_predict(Xq, Xr, w_all, sigma, classes)
    K      = size(w_all, 2);
    Phi_q  = exp(-pdist2(Xq,Xr,'euclidean').^2 / (2*sigma^2));
    prob   = zeros(size(Xq,1), K);
    for k  = 1:K
        prob(:,k) = 1./(1+exp(-Phi_q*w_all(:,k)));
    end
    [~,idx] = max(prob,[],2);
    y_out   = classes(idx);
end

% Precision / Recall / F1
function [precision, recall, f1] = computePRF1(cm, numClasses)
    precision = zeros(numClasses,1);
    recall    = zeros(numClasses,1);
    f1        = zeros(numClasses,1);
    for i = 1:numClasses
        TP = cm(i,i);
        FP = sum(cm(:,i)) - TP;
        FN = sum(cm(i,:)) - TP;
        precision(i) = ternary(TP+FP>0, TP/(TP+FP), 0);
        recall(i)    = ternary(TP+FN>0, TP/(TP+FN), 0);
        denom        = precision(i)+recall(i);
        f1(i)        = ternary(denom>0, 2*precision(i)*recall(i)/denom, 0);
    end
end


function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end