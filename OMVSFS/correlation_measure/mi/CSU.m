function [csu] = CSU(X, Y, Z)
% Conditional Symmetrical Uncertainty (CSU)
% CSU(X,Y|Z) = 2 * CMI(A;B|C) / (H(X|Z) + H(Y|Z))

% Calculate conditional entropies
hXZ = h([X Z]) - h(Z);
hYZ = h([Y Z]) - h(Z);

% Calculate CSU
numerator = 2 * cmi(X,Y,Z);
denominator = hXZ + hYZ;

% Handle division by zero
if denominator == 0
    csu = 0;
else
    csu = numerator / denominator;
end
end
