
% "Adaptive-Weighting discriminative regression for multi-view classification"
% (https://doi.org/10.1016/j.patcog.2018.11.015) in *Pattern Recognition*

% =========================================================================
%                        A Demo for WeightReg @mlyang
% =========================================================================

% X_train: 1 * v cells with d * n matrix inside, each column is a data point
% Y_train: n * 1, a column vector. pls ensure the class label is starting at 1 as (1, 2, 3, 4, ...)
%clear;
%clc;

% 组合视图
% views = fea; % 根据实际视图数调整
% y=gt;
% 划分训练集和测试集（80%训练，20%测试）

rng(42); % 固定随机种子
% 组合视图
views = X; % 根据实际视图数调整
% 划分训练集和测试集（80%训练，20%测试）
n_samples = size(y, 1);
%rng(1); % 固定随机种子
train_indices = randperm(n_samples, round(0.5*n_samples));
test_indices = setdiff(1:n_samples, train_indices);

% 分割各视图数据
X_train = cell(size(views));
X_test = cell(size(views));
for v = 1:length(views)
    X_train{v} = views{v}(train_indices, :)';
    X_test{v} = views{v}(test_indices, :)';
end
Y_train = y(train_indices);
Y_test = y(test_indices);

view_num = size(X_train, 2);
% gamma = [1e-3 1e-2 1e-1 1.0 10.0 100.0 1000.0];
gamma = 1.0;
iters = 200;
epsilon = 5e-3;

%% ---------- train the model ----------
for j = 1 : size(gamma,2)
    [XX_train, XX_test, W_dv, b, dv, dv_dim,time] = ...
train_weightreg(X_train, X_test, Y_train, view_num, gamma(j), iters, epsilon);
%     W = sqrt(dv_dim) .\ W_dv; % in case you wanna know

    %% --- classifcation with knn ---

    D = L2_distance(W_dv' * XX_train + b*ones(1, length(Y_train)), ...
        W_dv' * XX_test + b*ones(1, length(Y_test))); 

    [acc, acc_predict] = knn_classify(Y_train, Y_test, D', ...
        [1 3 5 7 9 11 13 15 17], 'ascend');
%     fprintf('-------------- gamma == %f --------------\n', gamma(j));

    for i = 1:9
        fprintf('Accuracy (k == %d): %.5f\n', 2*i-1, acc(i));
    end
end

return;
