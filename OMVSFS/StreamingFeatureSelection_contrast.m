clc; clear;
% Load data 
%data = load('data/3Sources_integ.mat');
data = load('......\data\3Sources_integ.mat');%Set the path of the dataset
[n_samples, n_features] = size(data.X);
X_data = data.X(:, 1:end-1); % Feature data
y_labels = data.X(:, end);   % Label data
p=n_features;
n_folds = 10;
alpha = 0.01; 
k_neighbors = 5; 
max_features = 200; % OSGFS_FI_inter Parameters

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
    
   % Split the data 
    train_data = X_data(train_indices, :);
    train_labels = y_labels(train_indices);
    test_data = X_data(test_indices, :);
    test_labels = y_labels(test_indices);
    

    % Perform feature selection
    [selected_features,time] =Alpha_Investing(train_data, train_labels);
    %[selected_features, time] = osfs_z([train_data,train_labels],p,alpha);
    %[selected_features, time] = osfs_d([train_data,train_labels],p,alpha,"g2");%"g2""chi2"
    %[ selected_features, time]=fast_osfs_z([train_data,train_labels],p,alpha);
    %[ selected_features, time]=fast_osfs_d([train_data,train_labels],p,alpha,"g2");
    %[selected_features,time] = saola_z_test([train_data,train_labels],alpha);
    %[selected_features,time] = saola_mi([train_data,train_labels],alpha);
    %[ selected_features,time ] = OFS_Density( train_data,train_labels);
    %[ selected_features,time ] = OFSS_FI_mi(train_data,train_labels,alpha);
    %[ selected_features,time ] = OFSS_FI_z(train_data,train_labels,alpha,alpha);
    %[selected_features, time] = OSGFS_FI_inter(train_data, train_labels, max_features, alpha);
    %[ selected_features, time ] = OSGFS_FI_inter_Z(train_data, train_labels, max_features, alpha);
    %[selected_features,time ] = OSFS_DD_z2(train_data,train_labels);
    %[ selected_features,time ] = OSFS_DD_mi2(train_data,train_labels);
    %[ selected_features,time ] = OSSFS(train_data,train_labels,alpha);
    %[selected_features,time,CSP] = OCFSSF_d([train_data,train_labels],p,alpha,"g2");
    %[selected_features,time,CSP]  = OCFSSF_z([train_data,train_labels],p,alpha,"g2");
    
    feature_selection_times(fold) = time;
    selected_features_counts(fold) = length(selected_features);

    
   % Train the KNN classifier
    mdl = fitcknn(train_data(:, selected_features), train_labels, 'NumNeighbors', k_neighbors);
    pred_labels = predict(mdl, test_data(:, selected_features));
    knn_confusion_matrices{fold} = confusionmat(test_labels, pred_labels);
    % Predict and calculate the accuracy
    knn_accuracy(fold) = sum(diag(knn_confusion_matrices{fold})) / sum(knn_confusion_matrices{fold}(:));
    % Train SVM multi-class classifier (using RBF kernel)
    mdl_svm = fitcecoc(train_data(:, selected_features), train_labels, ...
    'Learners', templateSVM('KernelFunction', 'rbf', 'BoxConstraint', 10,'KernelScale', 10), ...
    'Coding', 'onevsall');  % 'onevsone' can also be used
    % Predict and calculate classification accuracy
    pred_svm = predict(mdl_svm, test_data(:, selected_features));
    svm_accuracy(fold) = sum(pred_svm == test_labels) / numel(test_labels);

% Train decision tree (automatically handles multi-class classification)
mdl_tree = fitctree(train_data(:, selected_features), train_labels, ...
                'MaxNumSplits', 20, ...      % Maximum number of splits (controls tree depth)
                'MinLeafSize', 5, ...        % Minimum number of samples per leaf node
                'MinParentSize', 10, ...    % Minimum number of samples per parent node (split condition)
                'Prune', 'off');                    % Disable pruning first, control complexity via parameters

% Predict and calculate classification accuracy
pred_tree = predict(mdl_tree, test_data(:, selected_features));
tree_accuracy(fold) = sum(pred_tree == test_labels) / numel(test_labels);
end

% Calculate and display the overall results
mean_accuracy = mean(knn_accuracy);
mean_svm_accuracy = mean(svm_accuracy);
mean_tree_accuracy = mean(tree_accuracy);
mean_time = mean(feature_selection_times);
mean_features = mean(selected_features_counts);


fprintf('\n10-Fold Cross-Validation Results:\n');
fprintf('KNN Average Classification Accuracy: %.2f%% ± %.2f%%\n', mean_accuracy*100, std(knn_accuracy)*100);
fprintf('SVM Classification Accuracy: %.2f%% ± %.2f%%\n', mean_svm_accuracy * 100, std(svm_accuracy)*100);
fprintf('Decision Tree Average Classification Accuracy: %.2f%% ± %.2f%%\n', mean_tree_accuracy * 100,std(tree_accuracy)*100);
fprintf('Average Time Consumption: %.2f seconds\n', mean_time);
fprintf('Average Number of Features: %.1f\n', mean_features);

