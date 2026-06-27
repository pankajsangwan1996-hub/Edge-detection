function img_processed = preprocess_image(img, cfg)
% PREPROCESS_IMAGE - Preprocessing pipeline (Table 3)
%
% Input:
%   img - Input image (RGB or grayscale, any format)
%   cfg - Configuration structure
%
% Output:
%   img_processed - Grayscale image normalized to [0,1]

    % Step 1: Convert to grayscale
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    
    % Ensure uint8
    if ~isa(img_gray, 'uint8')
        if max(img_gray(:)) <= 1
            img_gray = uint8(img_gray * 255);
        else
            img_gray = uint8(img_gray);
        end
    end
    
    % Step 2: CLAHE
    img_clahe = adapthisteq(img_gray, ...
        'ClipLimit', cfg.preprocess.clahe_clip_limit / 100, ...
        'NumTiles', cfg.preprocess.clahe_tile_size, ...
        'Distribution', 'uniform');
    
    % Step 3: Normalize to [0, 1]
    img_processed = double(img_clahe) / 255.0;
end
