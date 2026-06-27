function [images, labels, split_info] = load_chestxray_dataset(cfg)
% LOAD_CHESTXRAY_DATASET - Load Chest X-ray Pneumonia dataset
%
% Loads .jpeg images from:
%   X:\DKN PAPERS\Rashmi\Dataset 04 jan\chest_xray\{test,train,val}\{NORMAL,PNEUMONIA}
%
% THIS MUST BE A SEPARATE .m FILE FROM load_mias_dataset.m
%
% Input:
%   cfg - Configuration structure
%
% Output:
%   images     - Cell array of images
%   labels     - Binary labels (1=pneumonia, 0=normal)
%   split_info - Structure with train/test/val indices

    fprintf('  Loading Chest X-ray dataset...\n');
    fprintf('  Base path: %s\n', cfg.dataset_base);
    
    images = {};
    labels = [];
    split_info = struct();
    split_info.train_idx = [];
    split_info.test_idx = [];
    split_info.val_idx = [];
    count = 0;
    
    % Training - Normal
    [images, labels, count, split_info] = load_chest_dir(...
        cfg.chestxray.train_normal, 0, images, labels, count, 'train', split_info);
    
    % Training - Pneumonia
    [images, labels, count, split_info] = load_chest_dir(...
        cfg.chestxray.train_pneumonia, 1, images, labels, count, 'train', split_info);
    
    % Test - Normal
    [images, labels, count, split_info] = load_chest_dir(...
        cfg.chestxray.test_normal, 0, images, labels, count, 'test', split_info);
    
    % Test - Pneumonia
    [images, labels, count, split_info] = load_chest_dir(...
        cfg.chestxray.test_pneumonia, 1, images, labels, count, 'test', split_info);
    
    % Validation - Normal
    [images, labels, count, split_info] = load_chest_dir(...
        cfg.chestxray.val_normal, 0, images, labels, count, 'val', split_info);
    
    % Validation - Pneumonia
    [images, labels, count, split_info] = load_chest_dir(...
        cfg.chestxray.val_pneumonia, 1, images, labels, count, 'val', split_info);
    
    labels = labels(:);
    fprintf('  Total: %d images (%d pneumonia, %d normal)\n', ...
        count, sum(labels==1), sum(labels==0));
end

%% ===== LOCAL HELPER FUNCTION =====
function [images, labels, count, split_info] = load_chest_dir(...
    dir_path, label, images, labels, count, split_type, split_info)
% Load all .jpeg images from one directory

    if ~exist(dir_path, 'dir')
        warning('Directory not found: %s', dir_path);
        return;
    end
    
    % Find .jpeg files (primary format in this dataset)
    files = [dir(fullfile(dir_path, '*.jpeg'));
             dir(fullfile(dir_path, '*.jpg'));
             dir(fullfile(dir_path, '*.png'))];
    
    if isempty(files)
        warning('No images found in: %s', dir_path);
        return;
    end
    
    start_idx = count + 1;
    
    for f = 1:length(files)
        filepath = fullfile(dir_path, files(f).name);
        try
            img = imread(filepath);
            count = count + 1;
            images{count} = img;
            labels(count) = label;
        catch
            continue;
        end
    end
    
    end_idx = count;
    
    if end_idx >= start_idx
        new_indices = start_idx:end_idx;
        switch lower(split_type)
            case 'train'
                split_info.train_idx = [split_info.train_idx, new_indices];
            case 'test'
                split_info.test_idx = [split_info.test_idx, new_indices];
            case 'val'
                split_info.val_idx = [split_info.val_idx, new_indices];
        end
        fprintf('    %s: loaded %d images\n', dir_path, end_idx - start_idx + 1);
    end
end
