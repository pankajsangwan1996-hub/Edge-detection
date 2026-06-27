function edge_map = fuzzy_nearness_edge_detection(I, T, sigma, GT)
% FUZZY_NEARNESS_EDGE_DETECTION - Core algorithm (Section 3.3)
%
% Input:
%   I     - Preprocessed grayscale image normalized to [0,1]
%   T     - Proximity threshold (default: 0.15)
%   sigma - Gaussian membership spread (default: 2.0)
%   GT    - Gradient threshold normalized (default: 10/255)
%
% Output:
%   edge_map - Binary edge map

    if nargin < 2, T = 0.15; end
    if nargin < 3, sigma = 2.0; end
    if nargin < 4, GT = 10/255; end
    
    [M, N] = size(I);
    edge_map = false(M, N);
    
    % 8-connected neighborhood offsets
    offsets = [-1,-1; -1,0; -1,1; 0,-1; 0,1; 1,-1; 1,0; 1,1];
    num_neighbors = size(offsets, 1);
    
    for i = 2:(M-1)
        for j = 2:(N-1)
            Ip = I(i, j);
            G_p = 0;
            
            for k = 1:num_neighbors
                ni = i + offsets(k, 1);
                nj = j + offsets(k, 2);
                Iq = I(ni, nj);
                
                intensity_diff = abs(double(Ip) - double(Iq));
                
                % Proximity relation (Eq. 5): p delta q iff |I(p)-I(q)| < T
                if intensity_diff < T
                    % Fuzzy membership (Eq. 8)
                    mu = exp(-(intensity_diff^2) / (2 * sigma^2));
                    % Fuzzy gradient (Eq. 9)
                    G_p = G_p + mu * intensity_diff;
                end
            end
            
            % Edge identification (Eq. 10)
            if G_p > GT
                edge_map(i, j) = true;
            end
        end
    end
end
