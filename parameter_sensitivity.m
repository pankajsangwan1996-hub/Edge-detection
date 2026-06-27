function results = parameter_sensitivity(images, labels, annotations, cfg)
% PARAMETER_SENSITIVITY - Test effect of T (Section 4.4, Figure 4)

    T_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30];
    results.T_values = T_values;
    results.f1_scores = zeros(length(T_values), 1);
    results.accuracy = zeros(length(T_values), 1);
    
    num_images = length(images);
    
    for t = 1:length(T_values)
        T_test = T_values(t);
        features_t = zeros(num_images, 5);
        
        for i = 1:num_images
            img_pre = preprocess_image(images{i}, cfg);
            edge_map = fuzzy_nearness_edge_detection(img_pre, T_test, ...
                cfg.params.sigma, cfg.params.GT_normalized);
            features_t(i,:) = extract_features(edge_map, images{i}, annotations(i), cfg);
        end
        
        valid = ~any(isnan(features_t), 2);
        feats_valid = features_t(valid, :);
        labels_valid = labels(valid);
        
        mu_f = mean(feats_valid);
        sd_f = std(feats_valid); sd_f(sd_f==0) = 1;
        feats_std = (feats_valid - mu_f) ./ sd_f;
        
        cv_res = run_cross_validation(feats_std, labels_valid, cfg);
        results.f1_scores(t) = mean(cv_res.f1);
        results.accuracy(t) = mean(cv_res.accuracy);
        
        fprintf('    T=%.2f: F1=%.3f, Acc=%.3f\n', T_test, results.f1_scores(t), results.accuracy(t));
    end
    
    [~, best_idx] = max(results.f1_scores);
    results.optimal_T = T_values(best_idx);
end
