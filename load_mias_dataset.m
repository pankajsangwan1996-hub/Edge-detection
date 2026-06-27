function [images, labels, annotations] = load_mias_dataset(cfg)
% LOAD_MIAS_DATASET - Load MIAS mammographic dataset
%
% Loads .pgm images (mdb001 to mdb322) from:
%   X:\DKN PAPERS\Rashmi\Dataset 04 jan\all-mias
%
% Input:
%   cfg - Configuration structure
%
% Output:
%   images      - Cell array of grayscale images
%   labels      - Binary labels (1=malignant, 0=benign)
%   annotations - Structure array with tumor coordinates

    image_dir = cfg.mias.image_dir;
    
    fprintf('  MIAS image directory: %s\n', image_dir);
    
    % Verify directory exists
    if ~exist(image_dir, 'dir')
        error('MIAS directory not found: %s\nPlease verify path in config.m', image_dir);
    end
    
    % Parse MIAS Info.txt annotation file
    annotations_raw = parse_mias_info(cfg.mias.info_file);
    
    images = {};
    labels = [];
    annotations = struct('x', {}, 'y', {}, 'radius', {}, ...
                        'tissue', {}, 'abnormality', {});
    
    valid_count = 0;
    
    for i = 1:length(annotations_raw)
        % Build filename: mdb001.pgm, mdb002.pgm, etc.
        filename = fullfile(image_dir, [annotations_raw(i).name, '.pgm']);
        
        % Try alternative extension if not found
        if ~exist(filename, 'file')
            filename = fullfile(image_dir, [annotations_raw(i).name, '.PGM']);
            if ~exist(filename, 'file')
                continue;
            end
        end
        
        % Read image
        try
            img = imread(filename);
        catch ME
            fprintf('  Warning: Could not read %s (%s), skipping.\n', ...
                filename, ME.message);
            continue;
        end
        
        % Only include images labeled Benign (B) or Malignant (M)
        if ~(strcmpi(annotations_raw(i).severity, 'B') || ...
             strcmpi(annotations_raw(i).severity, 'M'))
            continue;
        end
        
        % Skip if tumor coordinates are missing
        if isnan(annotations_raw(i).x) || isnan(annotations_raw(i).y) || ...
           isnan(annotations_raw(i).radius)
            continue;
        end
        
        valid_count = valid_count + 1;
        images{valid_count} = img;
        
        % Binary label: 1 = malignant, 0 = benign
        if strcmpi(annotations_raw(i).severity, 'M')
            labels(valid_count) = 1;
        else
            labels(valid_count) = 0;
        end
        
        annotations(valid_count).x = annotations_raw(i).x;
        annotations(valid_count).y = annotations_raw(i).y;
        annotations(valid_count).radius = annotations_raw(i).radius;
        annotations(valid_count).tissue = annotations_raw(i).tissue;
        annotations(valid_count).abnormality = annotations_raw(i).abnormality;
    end
    
    labels = labels(:);
    fprintf('  Loaded %d valid MIAS images\n', valid_count);
end

%% ===== LOCAL HELPER: Parse Info.txt =====
function annotations = parse_mias_info(info_file)
    if ~exist(info_file, 'file')
        parent_dir = fileparts(info_file);
        alt_names = {'Info.txt', 'info.txt', 'INFO.txt', 'INFO.TXT'};
        found = false;
        for f = 1:length(alt_names)
            alt_path = fullfile(parent_dir, alt_names{f});
            if exist(alt_path, 'file')
                info_file = alt_path;
                found = true;
                break;
            end
        end
        if ~found
            error('MIAS Info.txt not found in: %s', parent_dir);
        end
    end
    
    fid = fopen(info_file, 'r');
    if fid == -1
        error('Cannot open: %s', info_file);
    end
    
    annotations = struct('name', {}, 'tissue', {}, 'abnormality', {}, ...
                        'severity', {}, 'x', {}, 'y', {}, 'radius', {});
    idx = 0;
    
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line), continue; end
        line = strtrim(line);
        if isempty(line) || line(1) == '%' || line(1) == '#'
            continue;
        end
        
        parts = strsplit(line);
        if length(parts) < 3, continue; end
        
        idx = idx + 1;
        annotations(idx).name = parts{1};
        annotations(idx).tissue = parts{2};
        annotations(idx).abnormality = parts{3};
        
        if length(parts) >= 4 && (strcmpi(parts{4}, 'B') || strcmpi(parts{4}, 'M'))
            annotations(idx).severity = upper(parts{4});
        else
            annotations(idx).severity = 'NORM';
        end
        
        if length(parts) >= 7 && ~strcmpi(annotations(idx).severity, 'NORM')
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
    fprintf('  Parsed %d entries from Info.txt\n', idx);
end
