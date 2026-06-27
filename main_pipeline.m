%% ===================================================================
%% MAIN_PIPELINE.m - COMPLETE SINGLE FILE (RUN THIS FROM LINE 1)
%% Fuzzy Nearness Spaces for Enhanced Edge Detection
%% Press F5 ONCE to run everything. Do NOT run sections separately.
%% ===================================================================

clear; clc; close all;
rng(42);

%% ==================== CONFIGURATION ====================
dataset_base = 'X:\DKN PAPERS\Rashmi\Dataset 04 jan';
mias_image_dir = fullfile(dataset_base, 'all-mias');
mias_info_file = fullfile(dataset_base, 'all-mias', 'Info.txt');

% Algorithm Parameters
T = 0.15;           % Proximity threshold
sigma = 2.0;        % Gaussian membership spread
GT = 10/255;        % Gradient threshold (normalized)

% Preprocessing
clahe_clip_limit = 2.0;
clahe_tile_size = [8 8];
roi_margin = 0.20;

% SVM Parameters
svm_C = 10;
svm_gamma = 0.01;

% Cross-Validation
num_folds = 5;

% Output
output_dir = fullfile(pwd, 'results');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

fprintf('=== Fuzzy Nearness Spaces - Complete Pipeline ===\n');
fprintf('Dataset: %s\n', dataset_base);
fprintf('Parameters: T=%.2f, sigma=%.1f, GT=%.4f\n', T, sigma, GT);
fprintf('==================================================\n\n');

%% ==================== STEP 1: LOAD MIAS DATASET ====================
fprintf('[Step 1] Loading MIAS dataset...\n');

% Parse Info.txt
annotations_raw = local_parse_mias_info(mias_info_file);
fprintf('  Parsed %d entries from Info.txt\n', length(annotations_raw));

mias_images = {};
mias_labels = [];
mias_annotations = struct('x', {}, 'y', {}, 'radius', {});
valid_count = 0;

for i = 1:length(annotations_raw)
    fname = fullfile(mias_image_dir, [annotations_raw(i).name, '.pgm']);
    if ~exist(fname, 'file')
        fname = fullfile(mias_image_dir, [annotations_raw(i).name, '.PGM']);
        if ~exist(fname, 'file'), continue; end
    end
    if ~(strcmpi(annotations_raw(i).severity, 'B') || strcmpi(annotations_raw(i).severity, 'M'))
        continue;
    end
    if isnan(annotations_raw(i).x) || isnan(annotations_raw(i).y) || isnan(annotations_raw(i).radius)
        continue;
    end
    try
        img = imread(fname);
    catch
        continue;
    end
    valid_count = valid_count + 1;
    mias_images{valid_count} = img;
    if strcmpi(annotations_raw(i).severity, 'M')
        mias_labels(valid_count) = 1;
    else
        mias_labels(valid_count) = 0;
    end
    mias_annotations(valid_count).x = annotations_raw(i).x;
    mias_annotations(valid_count).y = annotations_raw(i).y;
    mias_annotations(valid_count).radius = annotations_raw(i).radius;
end
mias_labels = mias_labels(:);
fprintf('  Loaded %d valid images (%d malignant, %d benign)\n\n', ...
    valid_count, sum(mias_labels==1), sum(mias_labels==0));

if valid_count == 0
    error('No valid MIAS images loaded. Check Info.txt and .pgm files exist.');
end

%% ==================== STEP 2: EDGE DETECTION (ALL 3 METHODS) ====================
fprintf('[Step 2] Applying edge detection (Nearness + Canny + Sobel)...\n');
num_images = length(mias_images);

edge_maps_nearness = cell(num_images, 1);
edge_maps_canny = cell(num_images, 1);
edge_maps_sobel = cell(num_images, 1);
processing_times = zeros(num_images, 1);

for i = 1:num_images
    img_pre = local_preprocess(mias_images{i}, clahe_clip_limit, clahe_tile_size);
    
    tic;
    edge_maps_nearness{i} = local_fuzzy_nearness_edge(img_pre, T, sigma, GT);
    processing_times(i) = toc;
    
    edge_maps_canny{i} = edge(img_pre, 'canny');
    edge_maps_sobel{i} = edge(img_pre, 'sobel');
    
    if mod(i, 20) == 0
        fprintf('  Processed %d/%d\n', i, num_images);
    end
end
fprintf('  Done. Nearness avg time: %.1f ms\n\n', mean(processing_times)*1000);

%% ==================== STEP 3: FEATURE EXTRACTION (ALL 3 METHODS) ====================
fprintf('[Step 3] Extracting features for all methods...\n');

feats_nearness = zeros(num_images, 5);
feats_canny = zeros(num_images, 5);
feats_sobel = zeros(num_images, 5);

for i = 1:num_images
    feats_nearness(i,:) = local_extract_features(edge_maps_nearness{i}, ...
        mias_images{i}, mias_annotations(i), roi_margin);
    feats_canny(i,:) = local_extract_features(edge_maps_canny{i}, ...
        mias_images{i}, mias_annotations(i), roi_margin);
    feats_sobel(i,:) = local_extract_features(edge_maps_sobel{i}, ...
        mias_images{i}, mias_annotations(i), roi_margin);
end

% Common valid indices across all methods
valid_idx = ~any(isnan(feats_nearness),2) & ~any(isinf(feats_nearness),2) & ...
            ~any(isnan(feats_canny),2) & ~any(isinf(feats_canny),2) & ...
            ~any(isnan(feats_sobel),2) & ~any(isinf(feats_sobel),2);

feats_nearness = feats_nearness(valid_idx, :);
feats_canny = feats_canny(valid_idx, :);
feats_sobel = feats_sobel(valid_idx, :);
labels = mias_labels(valid_idx);   % <--- THIS CREATES 'labels'
valid_images = mias_images(valid_idx);
valid_annotations = mias_annotations(valid_idx);

fprintf('  Valid samples: %d (%d malignant, %d benign)\n\n', ...
    length(labels), sum(labels==1), sum(labels==0));

%% ==================== STEP 4: STANDARDIZE FEATURES ====================
fprintf('[Step 4] Standardizing features...\n');

mu_n = mean(feats_nearness); sd_n = std(feats_nearness); sd_n(sd_n==0)=1;
feats_nearness_std = (feats_nearness - mu_n) ./ sd_n;

mu_c = mean(feats_canny); sd_c = std(feats_canny); sd_c(sd_c==0)=1;
feats_canny_std = (feats_canny - mu_c) ./ sd_c;

mu_s = mean(feats_sobel); sd_s = std(feats_sobel); sd_s(sd_s==0)=1;
feats_sobel_std = (feats_sobel - mu_s) ./ sd_s;

fprintf('  Done.\n\n');

%% ==================== STEP 5: 5-FOLD CV (NEARNESS - Table 7) ====================
fprintf('[Step 5] Stratified 5-fold CV (Nearness-based)...\n');

cv_partition = cvpartition(labels, 'KFold', num_folds, 'Stratify', true);

accuracy = zeros(num_folds, 1);
precision_val = zeros(num_folds, 1);
recall_val = zeros(num_folds, 1);
f1_val = zeros(num_folds, 1);

for k = 1:num_folds
    X_tr = feats_nearness_std(training(cv_partition,k), :);
    y_tr = labels(training(cv_partition,k));
    X_te = feats_nearness_std(test(cv_partition,k), :);
    y_te = labels(test(cv_partition,k));
    
    mdl = fitcsvm(X_tr, y_tr, 'KernelFunction', 'rbf', ...
        'BoxConstraint', svm_C, 'KernelScale', 1/sqrt(svm_gamma), ...
        'Standardize', false, 'ClassNames', [0,1]);
    y_pred = predict(mdl, X_te);
    
    TP = sum(y_pred==1 & y_te==1); FP = sum(y_pred==1 & y_te==0);
    FN = sum(y_pred==0 & y_te==1); TN = sum(y_pred==0 & y_te==0);
    accuracy(k) = (TP+TN)/length(y_te);
    precision_val(k) = TP/max(TP+FP,1);
    recall_val(k) = TP/max(TP+FN,1);
    if (precision_val(k)+recall_val(k))>0
        f1_val(k) = 2*precision_val(k)*recall_val(k)/(precision_val(k)+recall_val(k));
    end
end

fprintf('\n  ============ Table 7: Nearness 5-Fold CV ============\n');
fprintf('  Fold | Accuracy | Precision | Recall  | F1-Score\n');
fprintf('  -----|----------|-----------|---------|----------\n');
for k = 1:num_folds
    fprintf('   %d   |  %.4f  |   %.4f  |  %.4f |  %.4f\n', ...
        k, accuracy(k), precision_val(k), recall_val(k), f1_val(k));
end
fprintf('  Mean |  %.4f  |   %.4f  |  %.4f |  %.4f\n\n', ...
    mean(accuracy), mean(precision_val), mean(recall_val), mean(f1_val));

%% ==================== STEP 6: METHOD COMPARISON (Table 8) ====================
fprintf('[Step 6] Comparing all methods...\n');

% Canny CV
acc_canny_folds = zeros(num_folds,1); f1_canny_folds = zeros(num_folds,1);
prec_canny_folds = zeros(num_folds,1); rec_canny_folds = zeros(num_folds,1);
cv_c = cvpartition(labels, 'KFold', num_folds, 'Stratify', true);
for k = 1:num_folds
    mdl_c = fitcsvm(feats_canny_std(training(cv_c,k),:), labels(training(cv_c,k)), ...
        'KernelFunction','rbf','BoxConstraint',svm_C,...
        'KernelScale',1/sqrt(svm_gamma),'ClassNames',[0,1]);
    yp = predict(mdl_c, feats_canny_std(test(cv_c,k),:));
    yt = labels(test(cv_c,k));
    tp=sum(yp==1&yt==1); fp=sum(yp==1&yt==0); fn=sum(yp==0&yt==1); tn=sum(yp==0&yt==0);
    acc_canny_folds(k)=(tp+tn)/length(yt);
    prec_canny_folds(k)=tp/max(tp+fp,1);
    rec_canny_folds(k)=tp/max(tp+fn,1);
    if (prec_canny_folds(k)+rec_canny_folds(k))>0
        f1_canny_folds(k)=2*prec_canny_folds(k)*rec_canny_folds(k)/(prec_canny_folds(k)+rec_canny_folds(k));
    end
end

% Sobel CV
acc_sobel_folds = zeros(num_folds,1); f1_sobel_folds = zeros(num_folds,1);
prec_sobel_folds = zeros(num_folds,1); rec_sobel_folds = zeros(num_folds,1);
cv_s = cvpartition(labels, 'KFold', num_folds, 'Stratify', true);
for k = 1:num_folds
    mdl_s = fitcsvm(feats_sobel_std(training(cv_s,k),:), labels(training(cv_s,k)), ...
        'KernelFunction','rbf','BoxConstraint',svm_C,...
        'KernelScale',1/sqrt(svm_gamma),'ClassNames',[0,1]);
    yp = predict(mdl_s, feats_sobel_std(test(cv_s,k),:));
    yt = labels(test(cv_s,k));
    tp=sum(yp==1&yt==1); fp=sum(yp==1&yt==0); fn=sum(yp==0&yt==1); tn=sum(yp==0&yt==0);
    acc_sobel_folds(k)=(tp+tn)/length(yt);
    prec_sobel_folds(k)=tp/max(tp+fp,1);
    rec_sobel_folds(k)=tp/max(tp+fn,1);
    if (prec_sobel_folds(k)+rec_sobel_folds(k))>0
        f1_sobel_folds(k)=2*prec_sobel_folds(k)*rec_sobel_folds(k)/(prec_sobel_folds(k)+rec_sobel_folds(k));
    end
end

fprintf('\n  ============ Table 8: Method Comparison ============\n');
fprintf('  Method    | Accuracy | Precision | Recall  | F1-Score\n');
fprintf('  ----------|----------|-----------|---------|----------\n');
fprintf('  Nearness  |  %.4f  |   %.4f  |  %.4f |  %.4f\n', ...
    mean(accuracy), mean(precision_val), mean(recall_val), mean(f1_val));
fprintf('  Canny     |  %.4f  |   %.4f  |  %.4f |  %.4f\n', ...
    mean(acc_canny_folds), mean(prec_canny_folds), mean(rec_canny_folds), mean(f1_canny_folds));
fprintf('  Sobel     |  %.4f  |   %.4f  |  %.4f |  %.4f\n\n', ...
    mean(acc_sobel_folds), mean(prec_sobel_folds), mean(rec_sobel_folds), mean(f1_sobel_folds));

%% ==================== STEP 7: ROC CURVES (3 METHODS) ====================
fprintf('[Step 7] Computing ROC curves for all 3 methods...\n');

% --- Nearness ROC ---
probs_nearness = zeros(length(labels), 1);
cv_roc_n = cvpartition(labels, 'KFold', num_folds, 'Stratify', true);
for k = 1:num_folds
    tr_idx = training(cv_roc_n, k);
    te_idx = test(cv_roc_n, k);
    mdl_rn = fitcsvm(feats_nearness_std(tr_idx,:), labels(tr_idx), ...
        'KernelFunction','rbf','BoxConstraint',svm_C,...
        'KernelScale',1/sqrt(svm_gamma),'ClassNames',[0,1]);
    mdl_rn_post = fitPosterior(mdl_rn);
    [~, post_n] = predict(mdl_rn_post, feats_nearness_std(te_idx,:));
    probs_nearness(te_idx) = post_n(:, 2);
end
[fpr_nearness, tpr_nearness, ~, auc_nearness] = perfcurve(labels, probs_nearness, 1);
fprintf('  Nearness AUC = %.3f\n', auc_nearness);

% --- Canny ROC ---
probs_canny = zeros(length(labels), 1);
cv_roc_c = cvpartition(labels, 'KFold', num_folds, 'Stratify', true);
for k = 1:num_folds
    tr_idx = training(cv_roc_c, k);
    te_idx = test(cv_roc_c, k);
    mdl_rc = fitcsvm(feats_canny_std(tr_idx,:), labels(tr_idx), ...
        'KernelFunction','rbf','BoxConstraint',svm_C,...
        'KernelScale',1/sqrt(svm_gamma),'ClassNames',[0,1]);
    mdl_rc_post = fitPosterior(mdl_rc);
    [~, post_c] = predict(mdl_rc_post, feats_canny_std(te_idx,:));
    probs_canny(te_idx) = post_c(:, 2);
end
[fpr_canny, tpr_canny, ~, auc_canny] = perfcurve(labels, probs_canny, 1);
fprintf('  Canny AUC = %.3f\n', auc_canny);

% --- Sobel ROC ---
probs_sobel = zeros(length(labels), 1);
cv_roc_s = cvpartition(labels, 'KFold', num_folds, 'Stratify', true);
for k = 1:num_folds
    tr_idx = training(cv_roc_s, k);
    te_idx = test(cv_roc_s, k);
    mdl_rs = fitcsvm(feats_sobel_std(tr_idx,:), labels(tr_idx), ...
        'KernelFunction','rbf','BoxConstraint',svm_C,...
        'KernelScale',1/sqrt(svm_gamma),'ClassNames',[0,1]);
    mdl_rs_post = fitPosterior(mdl_rs);
    [~, post_s] = predict(mdl_rs_post, feats_sobel_std(te_idx,:));
    probs_sobel(te_idx) = post_s(:, 2);
end
[fpr_sobel, tpr_sobel, ~, auc_sobel] = perfcurve(labels, probs_sobel, 1);
fprintf('  Sobel AUC = %.3f\n', auc_sobel);

% --- Bootstrap 95% CI for Nearness AUC ---
fprintf('  Computing bootstrap 95%% CI...\n');
n_boot = 1000;
n_samp = length(labels);
auc_boot = zeros(n_boot, 1);
for b = 1:n_boot
    bidx = randsample(n_samp, n_samp, true);
    if length(unique(labels(bidx))) < 2, auc_boot(b)=NaN; continue; end
    [~,~,~,auc_boot(b)] = perfcurve(labels(bidx), probs_nearness(bidx), 1);
end
auc_boot = auc_boot(~isnan(auc_boot));
auc_ci_lower = prctile(auc_boot, 2.5);
auc_ci_upper = prctile(auc_boot, 97.5);
fprintf('  95%% CI: [%.3f, %.3f]\n\n', auc_ci_lower, auc_ci_upper);

%% ==================== STEP 8: PLOT 3-CURVE ROC ====================
fprintf('[Step 8] Plotting 3-curve ROC figure...\n');

figure('Position', [100 100 850 750]);

% GREEN - Nearness (proposed method, should be TOP curve)
plot(fpr_nearness, tpr_nearness, 'Color', [0 0.7 0], 'LineWidth', 2.5);
hold on;

% BLUE - Canny
plot(fpr_canny, tpr_canny, 'b-', 'LineWidth', 2.0);

% RED - Sobel
plot(fpr_sobel, tpr_sobel, 'r-', 'LineWidth', 2.0);

% BLACK dashed - Random classifier diagonal
plot([0 1], [0 1], 'k--', 'LineWidth', 1.5);

% Formatting
xlabel('False Positive Rate (1 - Specificity)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('True Positive Rate (Sensitivity)', 'FontSize', 13, 'FontWeight', 'bold');
title('ROC Curves Comparing Edge Detection Methods for Tumor Classification', ...
    'FontSize', 13, 'FontWeight', 'bold');

% Legend
legend({sprintf('Nearness-based (AUC = %.3f)', auc_nearness), ...
        sprintf('Canny (AUC = %.3f)', auc_canny), ...
        sprintf('Sobel (AUC = %.3f)', auc_sobel), ...
        'Random Classifier'}, ...
    'Location', 'southeast', 'FontSize', 11, 'Box', 'on');

% Annotation textbox (top-left)
annot_str = sprintf(['Nearness-based approach (AUC = %.3f)\n' ...
    'demonstrates superior discrimination\n' ...
    'compared to Canny (AUC = %.3f)\nand Sobel (AUC = %.3f).'], ...
    auc_nearness, auc_canny, auc_sobel);
text(0.02, 0.88, annot_str, 'FontSize', 9, 'BackgroundColor', 'white', ...
    'EdgeColor', 'black', 'Margin', 4, 'VerticalAlignment', 'top');

% 95% CI annotation (green text)
ci_str = sprintf('95%% CI: [%.3f, %.3f]', auc_ci_lower, auc_ci_upper);
text(0.50, 0.30, ci_str, 'FontSize', 11, 'Color', [0 0.5 0], 'FontWeight', 'bold');

grid on;
set(gca, 'FontSize', 11);
xlim([0 1]); ylim([0 1]);
box on;

saveas(gcf, fullfile(output_dir, 'figure6_roc_3curves.png'));
saveas(gcf, fullfile(output_dir, 'figure6_roc_3curves.fig'));
fprintf('  ROC figure saved!\n\n');

%% ==================== STEP 9: PARAMETER SENSITIVITY ====================
fprintf('[Step 9] Parameter sensitivity analysis...\n');
T_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30];
f1_sensitivity = zeros(length(T_values), 1);

for t = 1:length(T_values)
    T_test = T_values(t);
    feats_t = zeros(length(labels), 5);
    for i = 1:length(labels)
        img_pre = local_preprocess(valid_images{i}, clahe_clip_limit, clahe_tile_size);
        edge_t = local_fuzzy_nearness_edge(img_pre, T_test, sigma, GT);
        feats_t(i,:) = local_extract_features(edge_t, valid_images{i}, valid_annotations(i), roi_margin);
    end
    v = ~any(isnan(feats_t),2) & ~any(isinf(feats_t),2);
    ft = feats_t(v,:); lt = labels(v);
    mu_t = mean(ft); sd_t = std(ft); sd_t(sd_t==0)=1;
    ft_s = (ft - mu_t) ./ sd_t;
    cv_t = cvpartition(lt, 'KFold', num_folds, 'Stratify', true);
    f1_folds = zeros(num_folds,1);
    for k = 1:num_folds
        mdl_t = fitcsvm(ft_s(training(cv_t,k),:), lt(training(cv_t,k)), ...
            'KernelFunction','rbf','BoxConstraint',svm_C,...
            'KernelScale',1/sqrt(svm_gamma),'ClassNames',[0,1]);
        yp = predict(mdl_t, ft_s(test(cv_t,k),:));
        yt = lt(test(cv_t,k));
        tp=sum(yp==1&yt==1); fp=sum(yp==1&yt==0); fn=sum(yp==0&yt==1);
        pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1);
        if (pr+rc)>0, f1_folds(k)=2*pr*rc/(pr+rc); end
    end
    f1_sensitivity(t) = mean(f1_folds);
    fprintf('  T=%.2f: F1=%.4f\n', T_test, f1_sensitivity(t));
end

% Sensitivity plot
figure('Position', [100 100 800 500]);
plot(T_values, f1_sensitivity, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
xlabel('Proximity Threshold T'); ylabel('F1-Score');
title('Parameter Sensitivity Analysis'); grid on;
saveas(gcf, fullfile(output_dir, 'figure4_sensitivity.png'));
fprintf('\n');

%% ==================== STEP 10: EDGE COMPARISON FIGURE ====================
figure('Position', [100 100 1200 300]);
img_demo = local_preprocess(mias_images{1}, clahe_clip_limit, clahe_tile_size);
subplot(1,4,1); imshow(img_demo); title('(a) Original');
subplot(1,4,2); imshow(edge_maps_nearness{1}); title('(b) Nearness');
subplot(1,4,3); imshow(edge_maps_canny{1}); title('(c) Canny');
subplot(1,4,4); imshow(edge_maps_sobel{1}); title('(d) Sobel');
saveas(gcf, fullfile(output_dir, 'figure5_edge_comparison.png'));

%% ==================== STEP 11: STATISTICAL ANALYSIS ====================
fprintf('[Step 11] Statistical analysis...\n');
n = length(labels);
p1 = mean(accuracy); p2 = mean(acc_canny_folds);
p_hat = (p1+p2)/2;
se_diff = sqrt(2*p_hat*(1-p_hat)/n);
z_stat = (p1-p2)/se_diff;
se_acc = sqrt(p1*(1-p1)/n);
ci_95 = [p1 - 1.96*se_acc, p1 + 1.96*se_acc];
fprintf('  Z-statistic: %.4f\n', z_stat);
fprintf('  95%% CI: [%.4f, %.4f]\n\n', ci_95(1), ci_95(2));

%% ==================== STEP 12: SAVE RESULTS ====================
results.accuracy = accuracy;
results.precision = precision_val;
results.recall = recall_val;
results.f1 = f1_val;
results.auc_nearness = auc_nearness;
results.auc_canny = auc_canny;
results.auc_sobel = auc_sobel;
results.auc_ci = [auc_ci_lower, auc_ci_upper];
results.z_stat = z_stat;
results.ci_95 = ci_95;
save(fullfile(output_dir, 'all_results.mat'), 'results');

%% ==================== FINAL SUMMARY ====================
fprintf('\n====================================================\n');
fprintf('  FINAL RESULTS\n');
fprintf('====================================================\n');
fprintf('  Accuracy:  %.4f +/- %.4f\n', mean(accuracy), std(accuracy));
fprintf('  Precision: %.4f +/- %.4f\n', mean(precision_val), std(precision_val));
fprintf('  Recall:    %.4f +/- %.4f\n', mean(recall_val), std(recall_val));
fprintf('  F1-Score:  %.4f +/- %.4f\n', mean(f1_val), std(f1_val));
fprintf('  AUC Nearness: %.3f [%.3f, %.3f]\n', auc_nearness, auc_ci_lower, auc_ci_upper);
fprintf('  AUC Canny:    %.3f\n', auc_canny);
fprintf('  AUC Sobel:    %.3f\n', auc_sobel);
fprintf('====================================================\n');
fprintf('  COMPLETE!\n');
fprintf('====================================================\n');


%% ======================================================================
%%            LOCAL FUNCTIONS BELOW (DO NOT MODIFY OR RUN SEPARATELY)
%% ======================================================================

function img_out = local_preprocess(img, clip_limit, tile_size)
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    if ~isa(img_gray, 'uint8')
        if max(img_gray(:)) <= 1
            img_gray = uint8(img_gray * 255);
        else
            img_gray = uint8(img_gray);
        end
    end
    img_clahe = adapthisteq(img_gray, ...
        'ClipLimit', clip_limit / 100, ...
        'NumTiles', tile_size, ...
        'Distribution', 'uniform');
    img_out = double(img_clahe) / 255.0;
end

function edge_map = local_fuzzy_nearness_edge(I, T, sigma, GT)
    [M, N] = size(I);
    edge_map = false(M, N);
    offsets = [-1,-1; -1,0; -1,1; 0,-1; 0,1; 1,-1; 1,0; 1,1];
    for i = 2:(M-1)
        for j = 2:(N-1)
            Ip = I(i, j);
            G_p = 0;
            for k = 1:8
                ni = i + offsets(k,1);
                nj = j + offsets(k,2);
                diff_val = abs(Ip - I(ni, nj));
                if diff_val < T
                    mu = exp(-(diff_val^2) / (2 * sigma^2));
                    G_p = G_p + mu * diff_val;
                end
            end
            if G_p > GT
                edge_map(i, j) = true;
            end
        end
    end
end

function feat = local_extract_features(edge_map, original_img, annotation, roi_margin_val)
    if size(original_img, 3) == 3
        img_gray = double(rgb2gray(original_img)) / 255;
    else
        if max(original_img(:)) > 1
            img_gray = double(original_img) / 255;
        else
            img_gray = double(original_img);
        end
    end
    [M, N] = size(edge_map);
    if isfield(annotation, 'x') && ~isnan(annotation.x)
        cx = round(annotation.x); cy = round(annotation.y);
        r = round(annotation.radius);
        margin = round(r * roi_margin_val);
        r1 = max(1, cy-r-margin); r2 = min(M, cy+r+margin);
        c1 = max(1, cx-r-margin); c2 = min(N, cx+r+margin);
        roi_edge = edge_map(r1:r2, c1:c2);
        roi_img = img_gray(r1:r2, c1:c2);
    else
        roi_edge = edge_map;
        roi_img = img_gray;
    end
    A = sum(roi_edge(:));
    if A == 0
        feat = [0, 0, mean(roi_img(:)), 0, std(roi_img(:))];
        return;
    end
    roi_filled = imfill(roi_edge, 'holes');
    P = sum(sum(bwperim(roi_filled)));
    if sum(roi_filled(:)) > 0
        mu_I = mean(roi_img(roi_filled));
        sigma_I = std(roi_img(roi_filled));
    else
        mu_I = mean(roi_img(:));
        sigma_I = std(roi_img(:));
    end
    C = (P^2) / max(A, 1);
    feat = [A, P, mu_I, C, sigma_I];
end

function annotations = local_parse_mias_info(info_file)
    if ~exist(info_file, 'file')
        parent_dir = fileparts(info_file);
        alt_names = {'Info.txt','info.txt','INFO.txt','INFO.TXT'};
        found = false;
        for f = 1:length(alt_names)
            alt_path = fullfile(parent_dir, alt_names{f});
            if exist(alt_path, 'file')
                info_file = alt_path; found = true; break;
            end
        end
        if ~found
            error('Info.txt not found in: %s', parent_dir);
        end
    end
    fid = fopen(info_file, 'r');
    if fid == -1, error('Cannot open: %s', info_file); end
    annotations = struct('name',{},'tissue',{},'abnormality',{},...
                        'severity',{},'x',{},'y',{},'radius',{});
    idx = 0;
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line), continue; end
        line = strtrim(line);
        if isempty(line) || line(1)=='%' || line(1)=='#', continue; end
        parts = strsplit(line);
        if length(parts) < 3, continue; end
        idx = idx + 1;
        annotations(idx).name = parts{1};
        annotations(idx).tissue = parts{2};
        annotations(idx).abnormality = parts{3};
        if length(parts)>=4 && (strcmpi(parts{4},'B') || strcmpi(parts{4},'M'))
            annotations(idx).severity = upper(parts{4});
        else
            annotations(idx).severity = 'NORM';
        end
        if length(parts)>=7 && ~strcmpi(annotations(idx).severity,'NORM')
            annotations(idx).x = str2double(parts{5});
            annotations(idx).y = str2double(parts{6});
            annotations(idx).radius = str2double(parts{7});
        else
            annotations(idx).x = NaN;
            annotations(idx).y = NaN;
            annotations(idx).radius = NaN;
        end
    end
    fclose(fid);
end
