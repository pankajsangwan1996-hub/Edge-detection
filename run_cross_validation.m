function cv_results = run_cross_validation(features, labels, cfg)
% RUN_CROSS_VALIDATION - Stratified 5-fold CV with RBF-SVM

    rng(cfg.cv.random_seed);
    K = cfg.cv.num_folds;
    n = length(labels);
    
    cv_partition = cvpartition(labels, 'KFold', K, 'Stratify', true);
    
    cv_results.accuracy = zeros(K, 1);
    cv_results.precision = zeros(K, 1);
    cv_results.recall = zeros(K, 1);
    cv_results.f1 = zeros(K, 1);
    cv_results.all_predictions = zeros(n, 1);
    cv_results.all_true_labels = labels;
    cv_results.all_scores = zeros(n, 1);
    
    for k = 1:K
        train_idx = training(cv_partition, k);
        test_idx = test(cv_partition, k);
        
        X_train = features(train_idx, :);
        y_train = labels(train_idx);
        X_test = features(test_idx, :);
        y_test = labels(test_idx);
        
        svm_model = fitcsvm(X_train, y_train, ...
            'KernelFunction', 'rbf', ...
            'BoxConstraint', cfg.svm.C, ...
            'KernelScale', 1/sqrt(cfg.svm.gamma), ...
            'Standardize', false, ...
            'ClassNames', [0, 1]);
        
        [y_pred, scores] = predict(svm_model, X_test);
        
        cv_results.all_predictions(test_idx) = y_pred;
        if size(scores, 2) >= 2
            cv_results.all_scores(test_idx) = scores(:, 2);
        else
            cv_results.all_scores(test_idx) = scores(:, 1);
        end
        
        TP = sum(y_pred == 1 & y_test == 1);
        FP = sum(y_pred == 1 & y_test == 0);
        FN = sum(y_pred == 0 & y_test == 1);
        TN = sum(y_pred == 0 & y_test == 0);
        
        cv_results.accuracy(k) = (TP + TN) / length(y_test);
        cv_results.precision(k) = TP / max(TP + FP, 1);
        cv_results.recall(k) = TP / max(TP + FN, 1);
        
        if (cv_results.precision(k) + cv_results.recall(k)) > 0
            cv_results.f1(k) = 2 * cv_results.precision(k) * cv_results.recall(k) / ...
                (cv_results.precision(k) + cv_results.recall(k));
        else
            cv_results.f1(k) = 0;
        end
    end
end
