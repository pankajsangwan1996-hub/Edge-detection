function results = compare_edge_methods(images, labels, annotations, cfg)
% COMPARE_EDGE_METHODS - Compare Nearness vs Canny, Sobel, LoG (Table 8)

    num_images = length(images);
    methods = {'Nearness', 'Canny', 'Sobel', 'LoG'};
    
    all_features = struct();
    for m = 1:length(methods)
        all_features.(methods{m}) = zeros(num_images, 5);
    end
    
    for i = 1:num_images
        img_pre = preprocess_image(images{i}, cfg);
        
        % Nearness (proposed)
        edge_near = fuzzy_nearness_edge_detection(img_pre, ...
            cfg.params.T, cfg.params.sigma, cfg.params.GT_normalized);
        all_features.Nearness(i,:) = extract_features(edge_near, images{i}, annotations(i), cfg);
        
        % Canny
        edge_canny = edge(img_pre, 'canny');
        all_features.Canny(i,:) = extract_features(edge_canny, images{i}, annotations(i), cfg);
        
        % Sobel
        edge_sobel = edge(img_pre, 'sobel');
        all_features.Sobel(i,:) = extract_features(edge_sobel, images{i}, annotations(i), cfg);
        
        % LoG
        edge_log = edge(img_pre, 'log');
        all_features.LoG(i,:) = extract_features(edge_log, images{i}, annotations(i), cfg);
        
        if mod(i, 50) == 0
            fprintf('    Compared %d/%d images\n', i, num_images);
        end
    end
    
    results = struct();
    for m = 1:length(methods)
        method = methods{m};
        feats = all_features.(method);
        
        valid = ~any(isnan(feats), 2);
        feats_valid = feats(valid, :);
        labels_valid = labels(valid);
        
        mu_f = mean(feats_valid);
        sd_f = std(feats_valid);
        sd_f(sd_f == 0) = 1;
        feats_std = (feats_valid - mu_f) ./ sd_f;
        
        cv_res = run_cross_validation(feats_std, labels_valid, cfg);
        
        results.(method).accuracy = mean(cv_res.accuracy);
        results.(method).precision = mean(cv_res.precision);
        results.(method).recall = mean(cv_res.recall);
        results.(method).f1 = mean(cv_res.f1);
        results.(method).predictions = cv_res.all_predictions;
        results.(method).true_labels = cv_res.all_true_labels;
    end
    
    fprintf('\n  Method     | Accuracy | Precision | Recall | F1\n');
    fprintf('  -----------|----------|-----------|--------|------\n');
    for m = 1:length(methods)
        r = results.(methods{m});
        fprintf('  %-10s |  %.3f   |   %.3f   | %.3f  | %.3f\n', ...
            methods{m}, r.accuracy, r.precision, r.recall, r.f1);
    end
end
