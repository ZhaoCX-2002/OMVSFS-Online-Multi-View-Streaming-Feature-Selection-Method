function [selected_features, time, view_weights] = OMVSFS_MI(views, y, alpha, beta)
% OMVSFS_MI algorithm
% Inputs:
%   views: Cell array of feature matrices (one per view)
%   y: Target variable
%   alpha: Feature relevance threshold (SU)
%   beta: Feature redundancy threshold (SU)
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
selected_features = struct('view', {}, 'index', {}, 'su', {});
view_scores = zeros(1, num_views); % View total scores
view_counts = zeros(1, num_views);  % Number of selected features per view
view_weights = ones(1, num_views)/num_views;  % Initial equal weights

for t = 1:max_time
    
    %Process views in the order of view weights (optional)
    %[~, view_order] = sort(view_weights, 'descend');
    
    for v_idx = 1:num_views
        %v = view_order(v_idx);
        v = v_idx;
        if t > view_feature_counts(v), continue; end
        
        % Get current feature data
        X_current = views{v}(:, t);
        
        % Adaptive threshold (optional)
         alpha_t = alpha * (1 - exp(-0.1 * t));  
         
        % ====================== Online correlation analysis ======================
        % Feature-target relevance test (SU)
        su_val = SU(X_current, y);
        su_val_weight = su_val* (0.8 + 0.2*view_weights(v));
       
        % Skip irrelevant features
        if (su_val_weight < alpha_t) || isnan(su_val_weight), continue; end
 
        % ====================== Intra-view online redundancy analysis ======================
        to_keep = true;
        remove_list = [];
        
        % Check relationship with selected features (same view only)
        % References for Redundancy Analysis:You, D.; Wang, Y.; Xiao, J.; Lin, Y.; Pan, M.; Chen, Z.; Shen,L.; and Wu, X. 2023c. Online Multi-Label Streaming Fea-ture Selection With Label Correlation. IEEE Transactionson Knowledge and Data Engineering, 35(3): 2901–2915.
        
        for s_idx = 1:length(selected_features)
            sf = selected_features(s_idx);
            if sf.view ~= v, continue; end  % Skip cross-view features
            
            X_sel = views{sf.view}(:, sf.index);
            
            % Calculate pairwise redundancy
            SU_xx = SU(X_current, X_sel);
            CI_pair = (SU_xx < beta);
            if ~CI_pair   % Significant dependence exists
                selected_score = sf.su;
                if (selected_score >= su_val) && (SU_xx > su_val)
                    to_keep = false;
                    break;
                end
                if (su_val > selected_score) && (SU_xx > selected_score)
                    remove_list = [remove_list, s_idx];
                end
            end
        end
        
        % ====================== Cross-view online complementarity analysis ======================
        % Only proceed with cross-view checks if feature passed same-view checks
        % References for complementarity analysis :Zhou, P.; Li, P.; Zhao, S.; and Wu, X. 2021. Feature Inter-action for Streaming Feature Selection. IEEE Transactions on Neural Networks and Learning Systems, 32(10): 4691–4702.
        if to_keep
            cross_view_interaction_sum = 0;
            cross_view_count = 0;
            
           % Check relationship with selected features from other views
            for s_idx = 1:length(selected_features)
                sf = selected_features(s_idx);
                if sf.view == v, continue; end  % Skip same-view features
                
                X_sel = views{sf.view}(:, sf.index);
                
                % Calculate cross-view interaction (conditional correlation)
                cmi_val = cmi(X_sel, y, X_current);
                interaction_val = cmi_val - sf.su;
                
                % Accumulate interaction values
                cross_view_interaction_sum = cross_view_interaction_sum + interaction_val;
                cross_view_count = cross_view_count + 1;
            end
            
            % Calculate average interaction
            if cross_view_count > 0
                avg_interaction = cross_view_interaction_sum / cross_view_count;               
                 % Decision based on interaction level
                if avg_interaction >=0  
                    % Keep feature (high interaction)
                    to_keep = true;
                else
                    % low interaction, check redundancy                
                    % Check redundancy with selected features
                    for s_idx = 1:length(selected_features)
                        sf = selected_features(s_idx);
                        if sf.view == v, continue; end  % Skip same-view features
                        
                        X_sel = views{sf.view}(:, sf.index);
                        mi_ij = SU(X_current, X_sel);
                         CI_pair = (mi_ij < beta);
                          if ~CI_pair  % Significant dependence exists
                        % Redundancy check logic
                        if sf.su >= su_val && mi_ij > su_val
                            to_keep = false;
                            break;
                        end
                        if su_val > sf.su && mi_ij > sf.su
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
            new_feature = struct('view', v, 'index', t, 'su', su_val);
            selected_features = [selected_features, new_feature];           
            
           % Update view statistics
            view_scores(v) = view_scores(v) + su_val;
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
