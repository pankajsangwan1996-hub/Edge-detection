function cfg = config()
% CONFIG - Configuration parameters for Fuzzy Nearness Edge Detection
% Dataset paths verified from actual directory structure.

    %% Dataset Base Path (VERIFIED from your directory)
    cfg.dataset_base = 'X:\DKN PAPERS\Rashmi\Dataset 04 jan';
    
    %% Chest X-ray Dataset Paths (VERIFIED - all .jpeg files)
    cfg.chestxray.test_normal = fullfile(cfg.dataset_base, 'chest_xray', 'test', 'NORMAL');
    cfg.chestxray.test_pneumonia = fullfile(cfg.dataset_base, 'chest_xray', 'test', 'PNEUMONIA');
    cfg.chestxray.train_normal = fullfile(cfg.dataset_base, 'chest_xray', 'train', 'NORMAL');
    cfg.chestxray.train_pneumonia = fullfile(cfg.dataset_base, 'chest_xray', 'train', 'PNEUMONIA');
    cfg.chestxray.val_normal = fullfile(cfg.dataset_base, 'chest_xray', 'val', 'NORMAL');
    cfg.chestxray.val_pneumonia = fullfile(cfg.dataset_base, 'chest_xray', 'val', 'PNEUMONIA');
    
    %% MIAS Dataset Path (VERIFIED - .pgm files mdb001 to mdb322)
    cfg.mias.image_dir = fullfile(cfg.dataset_base, 'all-mias');
    cfg.mias.info_file = fullfile(cfg.dataset_base, 'all-mias', 'Info.txt');
    
    %% Algorithm Parameters (Table 5 in paper)
    cfg.params.T = 0.15;
    cfg.params.sigma = 2.0;
    cfg.params.GT = 10;
    cfg.params.GT_normalized = cfg.params.GT / 255;
    
    %% Preprocessing Parameters (Table 3)
    cfg.preprocess.clahe_clip_limit = 2.0;
    cfg.preprocess.clahe_tile_size = [8 8];
    cfg.preprocess.roi_margin = 0.20;
    
    %% SVM Parameters (Section 5.2)
    cfg.svm.C = 10;
    cfg.svm.gamma = 0.01;
    cfg.svm.kernel = 'rbf';
    
    %% Cross-Validation Settings
    cfg.cv.num_folds = 5;
    cfg.cv.random_seed = 42;
    
    %% Bootstrap Settings
    cfg.bootstrap.num_samples = 10000;
    
    %% Output Directory
    cfg.output_dir = fullfile(pwd, 'results');
    if ~exist(cfg.output_dir, 'dir')
        mkdir(cfg.output_dir);
    end
end
