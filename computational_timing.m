%% COMPUTATIONAL_TIMING - Reproduce Table 9 timing results
% Measures execution time for all edge detection methods on 1024x1024 images

clear; clc;
cfg = config();

fprintf('=== Computational Complexity Analysis (Table 9) ===\n');
fprintf('Hardware: %s\n\n', cfg.hardware.description);

%% Create test image (1024 x 1024)
test_size = 1024;
num_runs = 100;

% Use a real MIAS image if available, otherwise synthetic
mias_dir = cfg.mias.image_dir;
pgm_files = dir(fullfile(mias_dir, '*.pgm'));

if ~isempty(pgm_files)
    img = imread(fullfile(mias_dir, pgm_files(1).name));
    img = imresize(img, [test_size, test_size]);
    img_pre = preprocess_image(img, cfg);
    fprintf('Using real MIAS image (mdb001)\n');
else
    img_pre = rand(test_size);
    fprintf('Using synthetic 1024x1024 image\n');
end

fprintf('Image size: %d x %d\n', test_size, test_size);
fprintf('Number of runs: %d\n\n', num_runs);

%% Method 1: Nearness-based
times_nearness = zeros(num_runs, 1);
for r = 1:num_runs
    tic;
    edge_nearness = fuzzy_nearness_edge_detection(img_pre, ...
        cfg.params.T, cfg.params.sigma, cfg.params.GT_normalized);
    times_nearness(r) = toc;
end

%% Method 2: Canny
times_canny = zeros(num_runs, 1);
for r = 1:num_runs
    tic;
    edge_canny = edge(img_pre, 'canny');
    times_canny(r) = toc;
end

%% Method 3: Sobel
times_sobel = zeros(num_runs, 1);
for r = 1:num_runs
    tic;
    edge_sobel = edge(img_pre, 'sobel');
    times_sobel(r) = toc;
end

%% Method 4: LoG
times_log = zeros(num_runs, 1);
for r = 1:num_runs
    tic;
    edge_log = edge(img_pre, 'log');
    times_log(r) = toc;
end

%% Display Results (Table 9)
fprintf('\n=== Table 9: Computational Complexity Comparison ===\n');
fprintf('%-12s | %-16s | %-16s | %-12s | %-8s\n', ...
    'Algorithm', 'Time Complexity', 'Space Complexity', 'Avg Time(ms)', 'FPS');
fprintf('-------------|------------------|------------------|--------------|--------\n');
fprintf('%-12s | %-16s | %-16s | %10.1f   | %.2f\n', ...
    'Nearness', 'O(M×N)', 'O(M×N)', mean(times_nearness)*1000, ...
    1/mean(times_nearness));
fprintf('%-12s | %-16s | %-16s | %10.1f   | %.2f\n', ...
    'Canny', 'O(M×N)', 'O(M×N)', mean(times_canny)*1000, ...
    1/mean(times_canny));
fprintf('%-12s | %-16s | %-16s | %10.1f   | %.2f\n', ...
    'Sobel', 'O(M×N)', 'O(M×N)', mean(times_sobel)*1000, ...
    1/mean(times_sobel));
fprintf('%-12s | %-16s | %-16s | %10.1f   | %.2f\n', ...
    'LoG', 'O(M×N log(M×N))', 'O(M×N)', mean(times_log)*1000, ...
    1/mean(times_log));

%% Save timing results
timing_results.nearness = times_nearness;
timing_results.canny = times_canny;
timing_results.sobel = times_sobel;
timing_results.log = times_log;
save(fullfile(cfg.output_dir, 'timing_results.mat'), 'timing_results');

fprintf('\nTiming results saved.\n');
