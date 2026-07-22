function [indexed, palette] = reduce_channels(image, ncol, median_win)
    % reduce_channels(image, ncol, median_win)
    %
    % Standalone, callable transcription of GDS_Code.m's UI_Act2_reduce()
    % color-reduction pipeline (see C:\Technion\physics\ARPES\
    % GDS_Conversion_Code\GDS_code.m), extracted so it can be called
    % programmatically (e.g. via the MATLAB Engine API for Python) instead
    % of only through the original GUI. The algorithm itself is untouched:
    % median-filter -> rgb2ind quantize -> canonical sorted+black-prepended
    % palette -> re-quantize the median-filtered image against that fixed
    % palette -> posterize -> median-filter again -> re-quantize once more.
    %
    % Inputs:
    %   image      - EITHER a path (char/string) to an image file (MATLAB
    %                does its own imread -- much faster from Python than
    %                marshalling a whole image array through the engine
    %                bridge, ~4s for a 2048x1536 image vs ~0.02s for a
    %                path string) OR an HxWx3 uint8 RGB array directly.
    %   ncol       - number of quantization colors (P.Ncol in GDS_Code.m, 8)
    %   median_win - median filter window size (P.Median_win, 5)
    %
    % Outputs:
    %   indexed - HxW uint8, values 0..ncol. 0 = reserved black (index-0
    %             palette entry, generally unused by real pixels); 1..ncol
    %             are the real clusters, sorted ascending by (R,G,B) -- a
    %             "channel" k (1-indexed, matching the existing label
    %             files' convention) is (indexed == k).
    %   palette - (ncol+1)x3 double, row i (1-indexed) = the RGB color of
    %             cluster (i-1). Row 1 is always [0 0 0].

    if ischar(image) || isstring(image)
        rgb_uint8 = imread(char(image));
    else
        rgb_uint8 = image;
    end
    if size(rgb_uint8, 3) > 3
        rgb_uint8 = rgb_uint8(:,:,1:3);
    end
    ncol = double(ncol);
    median_win = double(median_win);

    fws = median_win;
    Im21 = cat(3, medfilt2(rgb_uint8(:,:,1), [fws fws]), ...
                  medfilt2(rgb_uint8(:,:,2), [fws fws]), ...
                  medfilt2(rgb_uint8(:,:,3), [fws fws]));

    [~, mp22] = rgb2ind(Im21, ncol, 'nodither');
    mp24 = sortrows(round([0 0 0 ; mp22*255]));
    Im24 = rgb2ind(Im21, mp24/256, 'nodither');
    Im23 = uint8(round(ind2rgb(Im24, mp24)));

    Im25 = cat(3, medfilt2(Im23(:,:,1), [fws fws]), ...
                  medfilt2(Im23(:,:,2), [fws fws]), ...
                  medfilt2(Im23(:,:,3), [fws fws]));
    Im26 = rgb2ind(Im25, mp24/256, 'nodither');

    indexed = uint8(Im26);
    palette = mp24;
end
