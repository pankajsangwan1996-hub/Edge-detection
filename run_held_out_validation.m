function folds = stratified_kfold(labels, K, seed)
% STRATIFIED_KFOLD - Create stratified K-fold indices
%
% Ensures each fold has approximately the same class distribution.
% Alternative implementation if cvpartition is unavailable.
%
% Input:
%   labels - Binary label vector
%   K      - Number of folds
%   seed   - Random seed
%
% Output:
%   folds  - Cell array of K fold index vectors

    rng(seed);
    
    n = length(labels);
    folds = cell(K, 1);
    
    % Separate indices by class
    idx_pos = find(labels == 1);
    idx_neg = find(labels == 0);
    
    % Shuffle within each class
    idx_pos = idx_pos(randperm(length(idx_pos)));
    idx_neg = idx_neg(randperm(length(idx_neg)));
    
    % Distribute to folds
    fold_assignment = zeros(n, 1);
    
    for i = 1:length(idx_pos)
        fold_assignment(idx_pos(i)) = mod(i-1, K) + 1;
    end
    
    for i = 1:length(idx_neg)
        fold_assignment(idx_neg(i)) = mod(i-1, K) + 1;
    end
    
    for k = 1:K
        folds{k} = find(fold_assignment == k);
    end
end
