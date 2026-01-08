function [selectedFeatures, time] = OSGFS_FI_intra_Z(G, Y)
% OFS_INTERACTION 
% online streaming feature selection considering feature interaction
% Modified version using conditional independence tests instead of mutual information

start = tic;
[~, p] = size(G); % Get number of features
n = size(G, 1); % Number of samples
alpha = 0.05; % Significance level

% Initialize arrays
depArray = zeros(1, p); % Store dependency measures (now using Fisher's Z)
S = ones(1, p); % Feature state (1=available, -1=excluded, 0=processed)
R = zeros(1, p); % Selected features (1=selected, 0=not selected)

% Calculate initial dependencies
for i = 1:p
    [~, dep, ~] = my_cond_indep_fisher_z([G Y], i, p+1, [], n, alpha);
    depArray(1, i) = dep;
end

I_R = 0; % Current maximum dependency

while ~isempty(find(S == 1, 1)) % While there are available features
    [max_dep, I] = max(depArray); % Select most dependent feature
    
    if max_dep <= 0 % No significant dependencies left
        break;
    end
    
    current_index = I;
    S(1, current_index) = 0; % Mark as processed
    R(1, current_index) = 1; % Select temporarily
    
    % Calculate current dependency of selected set
    selected_indices = find(R == 1);
    [~, current_dep, ~] = my_cond_indep_fisher_z([G Y], selected_indices, p+1, [], n, alpha);
    current_dep = round(current_dep * 10000) / 10000;
    
    unSelected = find(S == 1); % Get unselected features
    sum_unSelected = length(unSelected);
    interactArray = zeros(1, p); % Store interaction information
    
    for j = 1:sum_unSelected
        unSelected_index = unSelected(1, j);
        
        % Test conditional independence: X ⊥ Y | current_feature
        [CI, dep, ~] = my_cond_indep_fisher_z([G Y], unSelected_index, p+1, current_index, n, alpha);
        
        if CI == 1 % Independent given current feature (redundant)
            S(1, unSelected_index) = -1; % Exclude
            depArray(1, unSelected_index) = -1;
        else % Potential interaction
            R(1, unSelected_index) = 1; % Temporarily select
            [~, joint_dep, ~] = my_cond_indep_fisher_z([G Y], [selected_indices unSelected_index], p+1, [], n, alpha);
            joint_dep = round(joint_dep * 10000) / 10000;
            
            if joint_dep > I_R
                interactArray(1, unSelected_index) = joint_dep;
            end
            R(1, unSelected_index) = 0; % Unselect
            S(1, unSelected_index) = -1; % Mark as processed
            depArray(1, unSelected_index) = -1;
        end
    end
    
    % Process interaction features
    if any(interactArray > 0)
        while ~isempty(find(interactArray > 0, 1))
            [~, I] = max(interactArray); % Select most interactive feature
            R(1, I) = 1; % Select
            [~, new_dep, ~] = my_cond_indep_fisher_z([G Y], find(R == 1), p+1, [], n, alpha);
            new_dep = round(new_dep * 10000) / 10000;
            
            if new_dep > I_R
                I_R = new_dep;
            else
                R(1, I) = 0; % Unselect if no improvement
            end
            interactArray(1, I) = 0; % Mark as processed
        end
    else
        if current_dep <= I_R % Current feature doesn't improve dependency
            S(1, current_index) = -1;
            R(1, current_index) = 0;
            depArray(1, current_index) = -1;
        else
            I_R = current_dep;
        end
    end
end

selectedFeatures = find(R == 1); % Final selected features
time = toc(start);
end
