function stats = statistical_analysis(cv_results, comparison_results, cfg)
% STATISTICAL_ANALYSIS - Z-test, McNemar's, Cohen's d (Section 6.3)

    pred_nearness = cv_results.all_predictions;
    true_labels = cv_results.all_true_labels;
    pred_canny = comparison_results.Canny.predictions;
    n = length(true_labels);
    
    % Z-statistic
    p1 = mean(cv_results.accuracy);
    p2 = comparison_results.Canny.accuracy;
    p_hat = (p1 + p2) / 2;
    se_diff = sqrt(2 * p_hat * (1 - p_hat) / n);
    stats.z_stat = (p1 - p2) / se_diff;
    stats.z_pvalue = 2 * (1 - normcdf(abs(stats.z_stat)));
    
    % McNemar's Test
    correct_nearness = (pred_nearness == true_labels);
    correct_canny = (pred_canny == true_labels);
    b = sum(correct_nearness & ~correct_canny);
    c = sum(~correct_nearness & correct_canny);
    
    if (b + c) > 0
        stats.mcnemar_chi2 = (abs(b - c) - 1)^2 / (b + c);
    else
        stats.mcnemar_chi2 = 0;
    end
    stats.mcnemar_p = 1 - chi2cdf(stats.mcnemar_chi2, 1);
    
    % Cohen's d
    acc_nearness = cv_results.accuracy;
    acc_canny = comparison_results.Canny.accuracy * ones(cfg.cv.num_folds, 1);
    pooled_std = sqrt((std(acc_nearness)^2 + std(acc_canny)^2) / 2);
    if pooled_std > 0
        stats.cohens_d = (mean(acc_nearness) - mean(acc_canny)) / pooled_std;
    else
        stats.cohens_d = Inf;
    end
    
    % Odds Ratio
    a_n = sum(correct_nearness);
    b_n = sum(~correct_nearness);
    a_c = sum(correct_canny);
    b_c = sum(~correct_canny);
    stats.odds_ratio = (a_n * b_c) / max(b_n * a_c, 1);
    se_log_or = sqrt(1/max(a_n,1) + 1/max(b_n,1) + 1/max(a_c,1) + 1/max(b_c,1));
    log_or = log(stats.odds_ratio);
    stats.odds_ratio_ci = [exp(log_or - 1.96*se_log_or), exp(log_or + 1.96*se_log_or)];
end
