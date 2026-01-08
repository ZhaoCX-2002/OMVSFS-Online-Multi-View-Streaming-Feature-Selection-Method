%clc; clear;

% Load data
data = load('......\data\3Sources.mat');%Set the path of the dataset

views = data.X;
y=data.y;

n_samples = size(y, 1);
n_folds = 10;
alpha = 0.01;
beta = 0.01;
k_neighbors = 5; % KNN parameters

% Create cross-validation partitions
% By default, the partitions are random. To make the results reproducible, set the random seed:
rng(42); % Set and fix the random seed 
cv = cvpartition(n_samples, 'KFold', n_folds);
% Initialize result storage
knn_accuracy = zeros(n_folds, 1);
feature_selection_times = zeros(n_folds, 1);
selected_features_counts = zeros(n_folds, 1);

% Perform 10-fold cross-validation
for fold = 1:n_folds
    fprintf('Processing fold %d/%d...\n', fold, n_folds);
    
    % Get the training and test indices of the current fold
    train_indices = training(cv, fold);
    test_indices = test(cv, fold);
    
    % Split the data of each view
    train_views = cell(size(views));
    test_views = cell(size(views));
    for v = 1:length(views)
        train_views{v} = views{v}(train_indices, :);
        test_views{v} = views{v}(test_indices, :);
    end
    train_y = y(train_indices);
    test_y = y(test_indices);
        
   % Execute online multi-view feature selection
   [selected_features, time,view_weights] = OMVSFS_MI(train_views, train_y, alpha,beta);
   %[selected_features, time,view_weights] = OMVSFS_fisher_z(train_views, train_y, alpha,beta);
   feature_selection_times(fold) = time;
   selected_features_counts(fold) = length(selected_features);
    
    % Extract selected features
    X_train_selected = [];
    X_test_selected = [];
    for sf = selected_features
        v = sf.view;
        idx = sf.index;
        X_train_selected = [X_train_selected, train_views{v}(:, idx)];
        X_test_selected = [X_test_selected, test_views{v}(:, idx)];
    end
    % Train the KNN classifier
    mdl = fitcknn(X_train_selected, train_y, 'NumNeighbors', k_neighbors);
    % Predict and calculate the accuracy
    pred = predict(mdl, X_test_selected);
    knn_accuracy(fold) = sum(pred == test_y) / numel(test_y);

    % Train SVM multi-class classifier (using RBF kernel，The parameters related to the classifier can be adjusted according to the dataset.)
    mdl_svm = fitcecoc(X_train_selected, train_y, ...
    'Learners', templateSVM('KernelFunction', 'rbf', 'BoxConstraint', 10,'KernelScale', 10), ...
    'Coding', 'onevsall');  % 'onevsone' can also be used
% Predict and calculate accuracy
pred_svm = predict(mdl_svm, X_test_selected);
svm_accuracy(fold) = sum(pred_svm == test_y) / numel(test_y);

% Train decision tree (The parameters related to the classifier can be adjusted according to the dataset.)
mdl_tree = fitctree(X_train_selected, train_y, ...
                'MaxNumSplits', 20, ...      % Maximum number of splits (controls tree depth)
                'MinLeafSize', 5, ...        % Minimum number of samples per leaf node
                'MinParentSize', 10, ...    % Minimum number of samples per parent node (split condition)
                'Prune', 'off');             % Disable pruning first, control complexity via parameters

% Predict and calculate accuracy
pred_tree = predict(mdl_tree, X_test_selected);
tree_accuracy(fold) = sum(pred_tree == test_y) / numel(test_y);
end

% Calculate and display overall results
mean_accuracy = mean(knn_accuracy);
mean_svm_accuracy = mean(svm_accuracy);
mean_tree_accuracy = mean(tree_accuracy);
mean_time = mean(feature_selection_times);
mean_features = mean(selected_features_counts);
%display(sind);
fprintf('\n10-Fold Cross-Validation Results:\n');
fprintf('KNN Average Classification Accuracy: %.2f%% ± %.2f%%\n', mean_accuracy*100, std(knn_accuracy)*100);
fprintf('SVM Average Classification Accuracy: %.2f%% ± %.2f%%\n', mean_svm_accuracy * 100, std(svm_accuracy)*100);
fprintf('Decision Tree Average Classification Accuracy: %.2f%% ± %.2f%%\n', mean_tree_accuracy * 100,std(tree_accuracy)*100);
fprintf('Average Feature Selection Time: %.2f seconds\n', mean_time);
fprintf('Average Number of Selected Features: %.1f\n', mean_features);
