function [ feature_hog, feature_lbp, feature_sift ] = af_feature( img_mb, img_mm, hogargs, lbpargs, siftargs )
% hogargs = [ cellnum, blknum ]
% lbpargs = [ cellnum, neighbor, r]
% siftargs = [class_num]

% Compute HOG
cellnum = hogargs(1);
blknum = hogargs(2);
filename = strcat('feature_hog_', num2str(cellnum), '_', ...
        num2str(blknum), '.mat');
if ~exist( filename )
    feature_hog = [extractHOG(img_mb, cellnum, blknum); ...
        extractHOG(img_mm, cellnum, blknum)];
    eval(char(strcat('save', {' '}, filename, {' '}, ' feature_hog')))
else
    load(filename)
end

% Compute LBP
cellnum = lbpargs(1);
neighbor = lbpargs(2);
r = lbpargs(3);
filename = strcat('feature_lbp_',  num2str(cellnum), '_', ...
        num2str(neighbor), '_', num2str(r), '.mat');
if ~exist(filename)
    feature_lbp = [extractLBP(img_mb, cellnum, neighbor, r); ...
        extractLBP(img_mm, cellnum, neighbor, r)];
    eval(char(strcat('save', {' '}, filename, {' '}, 'feature_lbp')))
else
    load(filename)
end

% Compute SIFT
class_num = siftargs;
filename = strcat('feature_sift_', num2str(class_num), '.mat');

if ~exist(filename)
    feature_sift = extractVLADSIFT( img_mb, img_mm, class_num );
    eval(char(strcat('save', {' '}, filename, {' '}, 'feature_sift')))
else 
    load(filename)
end


end

