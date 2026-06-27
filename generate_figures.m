function generate_figures(images, edge_maps, cv_results, comparison_results, ...
    roc_data, sensitivity_results, cfg)
% GENERATE_FIGURES - Generate all paper figures

    output_dir = cfg.output_dir;
    
    %% Figure 4: Parameter Sensitivity
    fig4 = figure('Position', [100 100 800 500]);
    plot(sensitivity_results.T_values, sensitivity_results.f1_scores, ...
        'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
    xlabel('Proximity Threshold T'); ylabel('F1-Score');
    title('Parameter Sensitivity Analysis');
    grid on;
    saveas(fig4, fullfile(output_dir, 'figure4_sensitivity.png'));
    
    %% Figure 5: Edge Comparison
    if ~isempty(images)
        fig5 = figure('Position', [100 100 1200 300]);
        img_pre = preprocess_image(images{1}, cfg);
        
        subplot(1,4,1); imshow(img_pre); title('(a) Original');
        subplot(1,4,2); imshow(edge_maps{1}); title('(b) Nearness');
        subplot(1,4,3); imshow(edge(img_pre,'canny')); title('(c) Canny');
        subplot(1,4,4); imshow(edge(img_pre,'sobel')); title('(d) Sobel');
        
        saveas(fig5, fullfile(output_dir, 'figure5_edge_comparison.png'));
    end
    
    %% Figure 6: ROC Curve
    fig6 = figure('Position', [100 100 700 600]);
    plot(roc_data.fpr, roc_data.tpr, 'b-', 'LineWidth', 2.5); hold on;
    plot([0 1], [0 1], 'k--');
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title('ROC Curve');
    legend(sprintf('Nearness (AUC=%.3f)', roc_data.nearness_auc), 'Random');
    grid on;
    saveas(fig6, fullfile(output_dir, 'figure6_roc.png'));
    
    close all;
    fprintf('  Figures saved to: %s\n', output_dir);
end
