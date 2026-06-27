function feat = extract_features(edge_map, original_img, annotation, cfg)
% EXTRACT_FEATURES - Feature extraction (Section 5.1)
%
% Output: [Area, Perimeter, MeanIntensity, Compactness, StdIntensity]

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
    
    % Extract ROI based on annotation
    if isfield(annotation, 'x') && ~isnan(annotation.x)
        cx = round(annotation.x);
        cy = round(annotation.y);
        r = round(annotation.radius);
        margin = round(r * cfg.preprocess.roi_margin);
        
        row_start = max(1, cy - r - margin);
        row_end = min(M, cy + r + margin);
        col_start = max(1, cx - r - margin);
        col_end = min(N, cx + r + margin);
        
        roi_edge = edge_map(row_start:row_end, col_start:col_end);
        roi_img = img_gray(row_start:row_end, col_start:col_end);
    else
        roi_edge = edge_map;
        roi_img = img_gray;
    end
    
    % Feature 1: Area
    A = sum(roi_edge(:));
    
    if A == 0
        feat = [0, 0, mean(roi_img(:)), 0, std(roi_img(:))];
        return;
    end
    
    % Feature 2: Perimeter
    roi_filled = imfill(roi_edge, 'holes');
    P = sum(sum(bwperim(roi_filled)));
    
    % Feature 3: Mean Intensity
    region_mask = roi_filled;
    if sum(region_mask(:)) > 0
        mu_I = mean(roi_img(region_mask));
    else
        mu_I = mean(roi_img(:));
    end
    
    % Feature 4: Compactness (P^2 / A)
    C = (P^2) / A;
    
    % Feature 5: Intensity Std
    if sum(region_mask(:)) > 0
        sigma_I = std(roi_img(region_mask));
    else
        sigma_I = std(roi_img(:));
    end
    
    feat = [A, P, mu_I, C, sigma_I];
end
