function [selected_features, time, view_weights] = OMVSFS_fisher_z(views, y, alpha, beta)
%OMVSFS_fisher_z algorithm
% Inputs:
%   views: Cell array of feature matrices (one per view)
%   y: Target variable
%   alpha: Feature relevance threshold (Fisher's Z)
%   beta: Feature redundancy threshold (Fisher's Z)
%
% Outputs:
%   selected_features: Selected features structure array
%   time: Execution time
%   view_weights: Final view weights

start_time = tic;
num_views = length(views);
view_feature_counts = cellfun(@(v) size(v,2), views);
max_time = max(view_feature_counts);

% Initialize data structures
selected_features = struct('view', {}, 'index', {}, 'corr', {});
view_scores = zeros(1, num_views);  % View total scores
view_counts = zeros(1, num_views);  % Number of selected features per view
view_weights = ones(1, num_views)/num_views;  % Initial equal weights

for t = 1:max_time
    
    % Process views in order of their weights
    %[~, view_order] = sort(view_weights, 'descend');
    
    for v_idx = 1:num_views
        v = view_order(v_idx);
        if t > view_feature_counts(v), continue; end
        
        % Get current feature data
        X_current = views{v}(:, t);
        data = [X_current, y];
        n = size(data, 1);
        
        % Adaptive threshold (optional)
        alpha_t = alpha / log(t + exp(1)); % Exponential decay 
        % ====================== Online correlation analysis ======================
        % Feature-target relevance test (Fisher's Z)
        [CI, corr] = my_cond_indep_fisher_z(data, 1, 2, [], n, alpha_t);
        current_score = corr;
        
        %current_score_weighted = current_score * (0.8 + 0.2*view_weights(v));
        
        % Skip irrelevant features
        if (CI == 1) || isnan(current_score) , continue; end  %|| (current_score_weighted < alpha_t), continue; end
        
        % ====================== Intra-view online redundancy analysis ======================
        to_keep = true;
        remove_list = [];
        
        % Check relationship with selected features (same view only)
        % References for Redundancy Analysis:You, D.; Wang, Y.; Xiao, J.; Lin, Y.; Pan, M.; Chen, Z.; Shen,L.; and Wu, X. 2023c. Online Multi-Label Streaming Fea-ture Selection With Label Correlation. IEEE Transactionson Knowledge and Data Engineering, 35(3): 2901–2915.
        for s_idx = 1:length(selected_features)
            sf = selected_features(s_idx);
            if sf.view ~= v, continue; end  % Skip cross-view features
            
            X_sel = views{sf.view}(:, sf.index);
            data_pair = [X_current, X_sel, y];
            
            % Calculate pairwise redundancy
            [CI_pair, pair_corr] = my_cond_indep_fisher_z(data_pair, 1, 2, [], n, beta);
           % pair_corr = abs(pair_corr);
            
            if CI_pair == 0  % Significant dependence exists
                selected_score = sf.corr;
                
                if (selected_score >= current_score) && (pair_corr > current_score)
                    to_keep = false;
                    break;
                elseif (current_score > selected_score) && (pair_corr > selected_score)
                    remove_list = [remove_list, s_idx];
                end
            end
        end
        
        % ====================== Cross-view online complementarity analysis ======================
        % Only proceed with cross-view checks if feature passed same-view checks
        if to_keep
            cross_view_interaction_sum = 0;
            cross_view_count = 0;
            
            % Check relationship with selected features from other views
            for s_idx = 1:length(selected_features)
                sf = selected_features(s_idx);
                if sf.view == v, continue; end  % Skip same-view features
                
                X_sel = views{sf.view}(:, sf.index);
                data_triple = [X_sel, y, X_current];%cmi_val = cmi(X_sel, y, X_current);
                
                % Calculate cross-view interaction (conditional correlation)
                [~, corr_cond] = my_cond_indep_fisher_z(data_triple, 1, 2, 3, n, alpha);
                interaction_val = corr_cond - sf.corr;
                
                % Accumulate interaction values
                cross_view_interaction_sum = cross_view_interaction_sum + interaction_val;
                cross_view_count = cross_view_count + 1;
            end
            
            % Calculate average interaction
            if cross_view_count > 0
                avg_interaction = cross_view_interaction_sum / cross_view_count;
                
                % Decision based on interaction level
                if avg_interaction >= 0
                    % Keep feature (high interaction)
                    to_keep = true;
                else%if avg_interaction >= 0
                    % Medium interaction, check redundancy
                    %to_keep = true;
                    
                    % Check redundancy with selected features
                    for s_idx = 1:length(selected_features)
                        sf = selected_features(s_idx);
                        if sf.view == v, continue; end  % Skip same-view features
                        
                        X_sel = views{sf.view}(:, sf.index);
                        data_pair = [X_current, X_sel];
                        [CI_pair, pair_corr] = my_cond_indep_fisher_z(data_pair, 1, 2, [], n, 0);%
                      %  pair_corr = abs(pair_corr);
                        
                        if CI_pair == 0  % Significant dependence exists
                            selected_score = sf.corr;
                            
                            if selected_score >= current_score && pair_corr > current_score
                                to_keep = false;
                                break;
                            elseif current_score > selected_score && pair_corr > selected_score
                                remove_list = [remove_list, s_idx];
                            end
                        end
                    end
                end
            end
        end
        
        % ====================== Online view weights update ======================
        if ~isempty(remove_list)
            selected_features(remove_list) = [];
        end
        
        if to_keep
            % Add new feature
            new_feature = struct('view', v, 'index', t, 'corr', corr);
            selected_features = [selected_features, new_feature];
            
            % Update view statistics
            view_scores(v) = view_scores(v) + current_score;
            view_counts(v) = view_counts(v) + 1;
            
            % Update view weights based on average score
            if view_counts(v) > 0
                view_weights(v) = view_scores(v) / view_counts(v);
            end
        end
    end
    
    % Normalize view weights
    view_weights = view_weights / (sum(view_weights) + eps);
end

time = toc(start_time);
end
