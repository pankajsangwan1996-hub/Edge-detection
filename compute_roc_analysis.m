function roc_data = compute_roc_analysis(features, labels, cfg)
% COMPUTE_ROC_ANALYSIS - ROC curve and AUC (Section 6.5)

    rng(cfg.cv.random_seed);
    K = cfg.cv.num_folds;
    cv_partition = cvpartition(labels, 'KFold', K, 'Stratify', true);
    
    all_scores = zeros(length(labels), 1);
    
    for k = 1:K
        train_idx = training(cv_partition, k);
        test_idx = test(cv_partition, k);
        
        svm_model = fitcsvm(features(train_idx,:), labels(train_idx), ...
            'KernelFunction', 'rbf', ...
            'BoxConstraint', cfg.svm.C, ...
            'KernelScale', 1/sqrt(cfg.svm.gamma), ...
            'ClassNames', [0, 1]);
        
        svm_scored = fitPosterior(svm_model);
        [~, post_prob] = predict(svm_scored, features(test_idx,:));
        all_scores(test_idx) = post_prob(:, 2);
    end
    
    [roc_data.fpr, roc_data.tpr, roc_data.thresholds] = perfcurve(labels, all_scores, 1);
    roc_data.nearness_auc = trapz(roc_data.fpr, roc_data.tpr);
    
    % Bootstrap CI for AUC
    n_boot = min(cfg.bootstrap.num_samples, 1000);
    n = length(labels);
    auc_boot = zeros(n_boot, 1);
    for b = 1:n_boot
        idx = randsample(n, n, true);
        [fpr_b, tpr_b] = perfcurve(labels(idx), all_scores(idx), 1);
        auc_boot(b) = trapz(fpr_b, tpr_b);
    end
    roc_data.auc_ci = [prctile(auc_boot, 2.5), prctile(auc_boot, 97.5)];
end
