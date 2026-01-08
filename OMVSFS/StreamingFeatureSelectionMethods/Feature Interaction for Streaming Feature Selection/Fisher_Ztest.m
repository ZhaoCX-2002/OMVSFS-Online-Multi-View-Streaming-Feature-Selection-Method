function [CI, dep, p_value, rho] = Fisher_Ztest(X, Y, S, alpha)
    % Fisher Z-test for independence between X and Y (optionally conditioned on S)
    % 输入:
    %   X: 变量1数据向量 (n×1)
    %   Y: 变量2数据向量 (n×1)
    %   S: 条件变量矩阵 (n×m), 可选。如果为空或未提供，则进行简单相关性检验
    %   alpha: 显著性水平 (默认0.05)
    % 输出:
    %   CI: 1表示独立(接受原假设), 0表示相关(拒绝原假设)
    %   dep: 依赖度量 (绝对偏相关系数或简单相关系数)
    %   p_value: 检验的p值
    %   rho: 原始相关系数(或偏相关系数)
    
    % 参数检查与默认值设置
    if nargin < 4, alpha = 0.05; end
    if nargin < 3, S = []; end
    
    n = length(X);
    if n ~= length(Y)
        error('X和Y的样本量必须相同');
    end
    
    % 数据预处理：去除NaN值
    valid_idx = ~(isnan(X) | isnan(Y));
    if ~isempty(S)
        valid_idx = valid_idx & all(~isnan(S), 2);
    end
    X = X(valid_idx);
    Y = Y(valid_idx);
    if ~isempty(S), S = S(valid_idx, :); end
    n_valid = length(X);
    
    % 计算相关系数或偏相关系数
    if isempty(S)
        % 简单相关性检验
        rho = corr(X, Y, 'Type', 'Pearson');
        if isnan(rho)  % 处理全零或其他特殊情况
            CI = 1;
            dep = 0;
            p_value = 1;
            return;
        end
        df = n_valid - 2;  % 自由度
    else
        % 条件独立性检验(偏相关)
        C = cov([X, Y, S]);
        X_idx = 1;
        Y_idx = 2;
        S_idx = 3:(size(S, 2)+2);
        rho = partial_corr_coef(C, X_idx, Y_idx, S_idx);
        df = n_valid - size(S, 2) - 2;  % 自由度
    end
    
    % Fisher Z变换
    if abs(rho) >= 1  % 处理边界情况
        rho = sign(rho) * 0.9999;
    end
    z = 0.5 * log((1 + rho) / (1 - rho));
    
    % 假设检验
    if isempty(S)
        % 简单相关性检验统计量
        z_null = 0;
        z_std = 1 / sqrt(df);
        W = (z - z_null) / z_std;
    else
        % 条件独立性检验统计量
        W = sqrt(df) * z;
    end
    
    % 计算p值(双侧检验)
    p_value = 2 * (1 - normcdf(abs(W)));
    
    % 独立性判断
    CI = (p_value > alpha);
    
    % 依赖度量(使用绝对相关系数)
    dep = abs(rho);
    
    % 可选: 显示警告如果样本量太小
    if n_valid < 20
        warning('样本量较小(%d), Fisher Z检验可能不可靠', n_valid);
    end
end

function r = partial_corr_coef(C, X_idx, Y_idx, S_idx)
    % 计算偏相关系数
    % C: 协方差矩阵
    % X_idx, Y_idx: 要计算相关性的变量索引
    % S_idx: 条件变量索引
    
    X = [X_idx Y_idx];
    i_pos = 1; % X在子矩阵中的位置
    j_pos = 2; % Y在子矩阵中的位置
    
    % 计算偏协方差矩阵
    C_XY = C(X, X);
    C_XS = C(X, S_idx);
    C_SS = C(S_idx, S_idx);
    C_SY = C(S_idx, Y_idx);
    
    % 处理奇异矩阵情况
    if rcond(C_SS) < 1e-12
        inv_C_SS = pinv(C_SS); % 使用伪逆
    else
        inv_C_SS = inv(C_SS);
    end
    
    partial_cov = C_XY - C_XS * inv_C_SS * C_SY';
    
    % 计算偏相关系数
    r = partial_cov(i_pos, j_pos) / sqrt(partial_cov(i_pos, i_pos) * partial_cov(j_pos, j_pos));
    
    % 确保数值稳定性
    if abs(r) > 1
        r = sign(r) * 1;
    end
end
